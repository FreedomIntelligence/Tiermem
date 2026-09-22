"""Explicit scripted demo: no provider calls, no accuracy/latency claims."""

from tempfile import TemporaryDirectory

from .providers import Decision
from .system import JevTierMem


class DemoRouter:
    def sufficient(self, query, evidence):
        needle = "17:30" if "time" in query else "email"
        enough = any(needle in item["text"] for item in evidence)
        return Decision(enough, {"sufficient": 1.0 if enough else 0.0})

    def allow_writeback(self, candidate, evidence, summaries):
        return Decision(True, {"supported": 1.0, "useful": 1.0, "novel": 1.0})


class DemoModel:
    def complete(self, task, instruction, data):
        if task == "summarize":
            return {"summary": "Mira prefers email reminders."}
        if task == "search":
            return {"queries": ["training departure"]}
        needle = "17:30" if "time" in data["query"] else "email"
        source = next(item for item in data["evidence"] if needle in item["text"])
        memory = None
        if needle == "17:30" and source["id"].startswith("r-"):
            memory = {"text": "Mira leaves for training at 17:30.",
                      "quotes": [{"raw_id": source["id"], "quote": source["text"]}]}
        return {"answer": "17:30" if needle == "17:30" else "Email",
                "supported": True, "citations": [source["id"]], "memory": memory}


def run():
    with TemporaryDirectory(prefix="jev-tiermem-demo-") as directory:
        with JevTierMem(directory, "demo", model=DemoModel(), router=DemoRouter()) as memory:
            first = memory.observe("Mira prefers reminders by email.")
            memory.observe("Training departure is 17:30 every Thursday.")
            memory.add_summary("Mira prefers email reminders.", first)
            return {
                "mode": "scripted offline demo; not live Jev and not a benchmark",
                "summary_path": memory.answer("Which reminder channel does Mira prefer?").to_dict(),
                "deepsearch_path": memory.answer("What time does Mira leave?").to_dict(),
                "after_writeback": memory.answer("What time does Mira leave?").to_dict(),
            }
