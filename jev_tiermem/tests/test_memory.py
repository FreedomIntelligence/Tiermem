import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_tiermem import Config, JevTierMem
from jev_tiermem.cli import load_observations
from jev_tiermem.demo import DemoModel, DemoRouter, run
from jev_tiermem.providers import Decision, JevRouter
from jev_tiermem.store import Store


class StubModel:
    def __init__(self, results):
        self.results = list(results)
        self.tasks = []

    def complete(self, task, instruction, data):
        self.tasks.append((task, data))
        result = self.results.pop(0)
        return result(data) if callable(result) else result


class StubRouter:
    def __init__(self, decisions=(), writeback=True):
        self.decisions = list(decisions)
        self.writeback = writeback
        self.write_calls = 0

    def sufficient(self, query, evidence):
        return Decision(self.decisions.pop(0))

    def allow_writeback(self, candidate, evidence, summaries):
        self.write_calls += 1
        return Decision(self.writeback)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = self.temporary.name
        self.store = Store(self.directory, "alice", chunk_chars=8)

    def test_lossless_paging_idempotence_and_reload(self):
        text = "  第一行\nsecond line\n末尾  "
        ids = self.store.observe(text, "tool", "2026-09-22", "event-1")
        pages = self.store.get(ids)
        self.assertEqual("".join(page["text"] for page in pages), text)
        reopened = Store(self.directory, "alice", chunk_chars=3)
        self.assertEqual(reopened.observe(text, "tool", "2026-09-22", "event-1"), ids)
        self.assertEqual(pages, reopened.get(ids))
        self.assertNotEqual(reopened.observe(text, "tool", "2026-09-22", "event-2"), ids)

    def test_isolation_and_session_path(self):
        ids = self.store.observe("secret")
        other = Store(self.directory, "../../bob")
        self.assertEqual(other.get(ids), [])
        self.assertEqual(other.search("secret"), [])
        self.assertEqual(other.directory.parent, Path(self.directory))

    def test_cjk_search_and_query_syntax(self):
        ids = self.store.observe("回调超时六十秒")
        self.assertEqual(self.store.search('超时 " OR *')[0]["id"], ids[0])
        self.assertEqual(self.store.search('"*'), [])
        self.assertEqual(self.store.search("超时", exclude=ids), [])

    def test_parallel_instances_deduplicate(self):
        other = Store(self.directory, "alice", chunk_chars=8)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda store: store.observe("parallel observation"), [self.store, other]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(self.store.search("parallel", limit=20)), 1)

    def test_plain_markdown_edits_and_budget_preserve_raw(self):
        ids = self.store.observe("original")
        self.assertEqual(self.store.append_note("old note", ids, "summary", 1000), "written")
        path = self.store.memory_path
        path.write_text(path.read_text().replace("old note", "edited note"), encoding="utf-8")
        self.assertEqual(self.store.summary_hits("edited", 1)[0]["text"], "edited note")
        before = path.read_bytes()
        self.assertEqual(self.store.append_note("new note", ids, "writeback", len(before)), "summary_budget")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.get(ids)[0]["text"], "original")

    def test_invalid_source_and_markers_do_not_change_memory(self):
        ids = self.store.observe("original")
        self.assertEqual(self.store.append_note("note", ["r-missing"], "summary", 1000), "invalid_sources")
        self.assertEqual(self.store.append_note("<!-- forged -->", ids, "summary", 1000), "invalid_markers")
        self.assertFalse(self.store.memory_path.exists())

    def test_import_rejects_bad_record_before_any_writes(self):
        path = Path(self.directory) / "bad.json"
        path.write_text(json.dumps([{"text": "valid"}, {"text": 42}]))
        with self.assertRaises(ValueError):
            load_observations(path)


class RouterTests(unittest.TestCase):
    def router(self, values):
        client = SimpleNamespace(system_one=lambda **kwargs: SimpleNamespace(
            nouls={key: SimpleNamespace(noul=value) for key, value in values.items()},
            usage=SimpleNamespace(input_tokens=12, output_tokens=0),
        ))
        return JevRouter(Config(), client=client)

    def test_threshold_and_usage(self):
        router = self.router({"sufficient": 0.8})
        self.assertTrue(router.sufficient("query", [{"text": "evidence"}]).accepted)
        self.assertEqual((router.calls, router.input_tokens), (1, 12))
        self.assertFalse(self.router({"sufficient": 0.79}).sufficient("query", [{}]).accepted)

    def test_invalid_or_missing_probability_fails_closed(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.99", None):
            with self.subTest(value=value):
                result = self.router({"sufficient": value}).sufficient("query", [{}])
                self.assertFalse(result.accepted)
                self.assertEqual(result.error, "ValueError")
        self.assertFalse(self.router({}).sufficient("query", [{}]).accepted)

    def test_network_failure_fails_closed(self):
        def fail(**kwargs):
            raise TimeoutError("must not appear in trace")
        router = JevRouter(Config(), client=SimpleNamespace(system_one=fail))
        result = router.sufficient("query", [{}])
        self.assertFalse(result.accepted)
        self.assertEqual(result.error, "TimeoutError")

    def test_all_write_conditions_required(self):
        for failed in ("supported", "useful", "novel"):
            probabilities = {"supported": .99, "useful": .99, "novel": .99, failed: .89}
            self.assertFalse(self.router(probabilities).allow_writeback({}, [], []).accepted)
        self.assertTrue(self.router({"supported": .99, "useful": .99, "novel": .99})
                        .allow_writeback({}, [], []).accepted)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = self.temporary.name

    def memory(self, model, router, **config):
        return JevTierMem(self.directory, "test", Config(**config), model=model, router=router)

    def test_summary_path_never_reads_raw(self):
        model = StubModel([lambda data: {"answer": "email", "supported": True,
                                         "citations": [data["evidence"][0]["id"]]}])
        memory = self.memory(model, StubRouter([True]))
        ids = memory.observe("email")
        memory.add_summary("email", ids)
        with patch.object(memory.store, "get", side_effect=AssertionError("raw read")), \
                patch.object(memory.store, "search", side_effect=AssertionError("raw search")):
            result = memory.answer("channel?")
        self.assertEqual(result.route, "S")
        self.assertEqual(model.tasks[0][0], "answer")
        self.assertEqual(result.trace["writeback"]["status"], "not_applicable")

    def test_retrieve_uses_host_answer_without_text_model(self):
        model = StubModel([])
        memory = self.memory(model, StubRouter([True]))
        memory.add_summary("email", memory.observe("email"))
        result = memory.retrieve("channel?")
        self.assertEqual(result["route"], "S")
        self.assertEqual(model.tasks, [])

    def test_deepsearch_writeback_makes_next_query_fast(self):
        result = run()
        self.assertEqual(result["summary_path"]["route"], "S")
        self.assertEqual(result["deepsearch_path"]["route"], "R")
        self.assertEqual(len(result["deepsearch_path"]["trace"]["rounds"]), 2)
        self.assertEqual(result["deepsearch_path"]["trace"]["writeback"]["status"], "written")
        self.assertEqual(result["after_writeback"]["route"], "S")

    def test_forced_deepsearch_skips_summary_router(self):
        memory = self.memory(StubModel([]), StubRouter([True]))
        memory.add_summary("email", memory.observe("email"))
        result = memory.retrieve("email", deepsearch=True)
        self.assertEqual(result["route"], "R")
        self.assertEqual([d["stage"] for d in result["trace"]["decisions"]], ["raw"])

    def test_page_and_round_budgets(self):
        memory = self.memory(StubModel([]), StubRouter([False, False]), max_raw_pages=1, max_search_rounds=1)
        memory.observe("topic one")
        memory.observe("topic two")
        result = memory.retrieve("topic")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["trace"]["stop_reason"], "raw_budget")

    def test_no_results_stops_without_fabricated_answer(self):
        model = StubModel([{"queries": ["another topic"]}])
        memory = self.memory(model, StubRouter([False, False]))
        result = memory.answer("missing")
        self.assertFalse(result.supported)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.trace["stop_reason"], "no_new_evidence")
        self.assertEqual([task for task, _ in model.tasks], ["search"])

    def test_writeback_disabled_leaves_summary_unchanged(self):
        memory = self.memory(DemoModel(), DemoRouter())
        ids = memory.observe("Mira prefers reminders by email. Training departure is 17:30.")
        memory.add_summary("Mira prefers email reminders.", ids)
        before = memory.store.memory_path.read_bytes()
        result = memory.answer("What time does Mira leave?", writeback=False)
        self.assertEqual(result.trace["writeback"]["status"], "disabled")
        self.assertEqual(memory.store.memory_path.read_bytes(), before)

    def test_promote_rejects_hallucinated_quotes_before_jev(self):
        router = StubRouter()
        memory = self.memory(StubModel([]), router)
        ids = memory.observe("timeout 60")
        for quotes in ([{"raw_id": ids[0], "quote": "timeout 100"}],
                       [{"raw_id": "r-other-session", "quote": "timeout 60"}]):
            self.assertEqual(memory.promote("fact", quotes)["status"], "unverified_quote")
        self.assertEqual(router.write_calls, 0)
        self.assertEqual(memory.store.notes(), [])

    def test_promote_rejection_duplicates_and_reload(self):
        router = StubRouter(writeback=False)
        memory = self.memory(StubModel([]), router)
        ids = memory.observe("timeout 60")
        quotes = [{"raw_id": ids[0], "quote": "timeout 60"}]
        self.assertEqual(memory.promote("timeout 60", quotes)["status"], "rejected")
        self.assertEqual(memory.store.notes(), [])
        router.writeback = True
        self.assertEqual(memory.promote("timeout 60", quotes)["status"], "written")
        self.assertEqual(memory.promote("timeout 60", quotes)["status"], "duplicate")
        self.assertEqual(router.write_calls, 2)
        self.assertEqual(Store(self.directory, "test").notes()[0]["raw_ids"], ids)

    def test_invalid_answer_citation_is_not_returned(self):
        memory = self.memory(StubModel([{"answer": "invented", "supported": True,
                                        "citations": ["r-forged"]}]), StubRouter([True]))
        memory.add_summary("email", memory.observe("email"))
        with self.assertRaises(ValueError):
            memory.answer("channel?")

    def test_compact_is_incremental_and_keeps_raw(self):
        model = StubModel([{"summary": "first note"}, {"summary": "second note"}])
        memory = self.memory(model, StubRouter())
        first = memory.observe("first original")
        self.assertEqual(memory.compact()["batches"], 1)
        self.assertEqual(memory.compact()["batches"], 0)
        second = memory.observe("second original")
        self.assertEqual(memory.compact()["batches"], 1)
        self.assertEqual(len(memory.store.get(first + second)), 2)
        self.assertEqual(len(model.tasks), 2)

    def test_bad_summary_preserves_pending_records(self):
        memory = self.memory(StubModel([{"summary": "too long"}]), StubRouter(), note_max_chars=2)
        ids = memory.observe("original")
        with self.assertRaises(ValueError):
            memory.compact()
        self.assertEqual(memory.store.notes(), [])
        self.assertEqual([p["id"] for p in memory.store.pending_batch(100)], ids)

    def test_identical_summary_for_new_records_preserves_source_coverage(self):
        model = StubModel([{"summary": "same fact"}, {"summary": "same fact"}])
        memory = self.memory(model, StubRouter())
        first = memory.observe("same fact", dia_id="one")
        memory.compact()
        second = memory.observe("same fact", dia_id="two")
        self.assertEqual(memory.compact()["stop_reason"], "complete")
        self.assertEqual(memory.compact()["batches"], 0)
        self.assertEqual(len(memory.store.notes()), 1)
        self.assertEqual(memory.store.notes()[0]["raw_ids"], first + second)


class IsolationTests(unittest.TestCase):
    def test_standalone_import_does_not_load_paper_or_provider_dependencies(self):
        code = """
import sys
import jev_tiermem
assert not any(name.split('.')[0] in {'src', 'core', 'torch', 'mem0', 'qdrant_client',
                                     'openai', 'typesafe_sdk', 'mcp'} for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_config(self):
        for kwargs in ({"max_search_rounds": 0}, {"sufficient_threshold": float("nan")},
                       {"writeback_threshold": 1.1}, {"raw_chunk_chars": 30000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
