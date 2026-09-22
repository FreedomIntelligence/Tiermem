import tempfile
import unittest
from pathlib import Path

from jev_tiermem import Config, JevTierMem
from jev_tiermem.agent_loop import AgentLoop
from jev_tiermem.providers import Decision


class ScriptModel:
    def __init__(self, actions):
        self.actions = list(actions)
        self.inputs = []

    def complete(self, task, instruction, data):
        self.inputs.append((task, data))
        if task == "agent_notes":
            return {"summary": "The diagnostic file was read. Timeout adjustment passed; original details remain in raw history."}
        action = self.actions.pop(0)
        return action(data) if callable(action) else action


class RawRouter:
    def sufficient(self, query, evidence):
        return Decision(any("speaker" in item for item in evidence))


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "workspace"
        self.workspace.mkdir()
        self.directory = Path(self.temporary.name) / "memory"

    def test_eviction_preserves_raw_and_fresh_loop_recovers_details(self):
        text = "routine line\n" * 300 + "opaque_tag=EXACT-7q9; timeout=83"
        (self.workspace / "diagnostic.log").write_text(text)
        model = ScriptModel([
            {"action": "tool", "name": "read_file", "arguments": {"path": "diagnostic.log"}},
            {"action": "final", "answer": "Done."},
        ])
        memory = JevTierMem(self.directory, "test", Config(), model=model, router=RawRouter())
        first = AgentLoop(memory, self.workspace, context_chars=1000)
        result = first.run("Read diagnostic.log")
        self.assertEqual(result["status"], "complete")
        self.assertTrue(any(event["event"] == "checkpoint" for event in first.events))
        after_checkpoint = [data for task, data in model.inputs if task == "agent_step"][-1]
        self.assertEqual(after_checkpoint["completed_tools"][0]["name"], "read_file")
        self.assertEqual(after_checkpoint["completed_tools"][0]["status"], "complete")
        self.assertNotIn(text, str(after_checkpoint["completed_tools"]))
        first.checkpoint()
        note_inputs = [data for task, data in model.inputs if task == "agent_notes"]
        self.assertTrue(any("EXACT-7q9" in raw["text"] for data in note_inputs for raw in data["raw"]))
        self.assertFalse(any("EXACT-7q9" in note["text"] for note in memory.store.notes()))

        def respond(data):
            tool_result = next(item["text"] for item in data["recent"] if item.get("tool") == "recall")
            self.assertIn("EXACT-7q9", tool_result)
            return {"action": "final", "answer": "EXACT-7q9"}

        model.actions.extend([
            {"action": "tool", "name": "recall", "arguments": {"query": "opaque_tag timeout"}}, respond,
        ])
        second = AgentLoop(memory, self.workspace, context_chars=20000)
        self.assertEqual(second.recent, [])
        self.assertEqual(second.run("What was the opaque tag?")["answer"], "EXACT-7q9")
        self.assertTrue(any(e["event"] == "recall" and e["route"] == "R" for e in second.events))

    def test_checkpoint_does_not_resummarize_notes_or_recalled_evidence(self):
        model = ScriptModel([])
        memory = JevTierMem(self.directory, "test", model=model, router=RawRouter())
        loop = AgentLoop(memory, self.workspace)
        loop._record("user", "Original instruction A", summarize=True)
        loop.checkpoint()
        loop._record("tool", "Already compressed note should not be summarized again", summarize=False, name="recall")
        loop._record("user", "Original instruction B", summarize=True)
        loop.checkpoint()
        raw = model.inputs[1][1]["raw"]
        self.assertEqual([item["text"] for item in raw], ["Original instruction B"])
        self.assertTrue(memory.search("Already compressed"))

    def test_failed_note_write_keeps_window_and_pending_sources(self):
        model = ScriptModel([])
        memory = JevTierMem(self.directory, "test", Config(memory_max_chars=100, note_max_chars=100),
                            model=model, router=RawRouter())
        loop = AgentLoop(memory, self.workspace)
        ids = loop._record("user", "original fact", summarize=True)
        with self.assertRaises((ValueError, RuntimeError)):
            loop.checkpoint()
        self.assertEqual(loop.pending, ids)
        self.assertEqual(loop.recent[0]["text"], "original fact")

    def test_workspace_tool_cannot_read_outside_root(self):
        memory = JevTierMem(self.directory, "test", model=ScriptModel([]), router=RawRouter())
        loop = AgentLoop(memory, self.workspace)
        with self.assertRaises(ValueError):
            loop._tool("read_file", {"path": "../outside.txt"})

    def test_step_budget_and_tool_errors_are_explicit(self):
        model = ScriptModel([{"action": "tool", "name": "unknown", "arguments": {}}])
        memory = JevTierMem(self.directory, "test", model=model, router=RawRouter())
        loop = AgentLoop(memory, self.workspace, max_steps=1)
        self.assertEqual(loop.run("try a tool")["status"], "step_limit")
        self.assertTrue(any(event["event"] == "tool_error" for event in loop.events))


if __name__ == "__main__":
    unittest.main()
