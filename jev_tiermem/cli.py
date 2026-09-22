import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config
from .system import JevTierMem


def load_observations(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("JSON input must be a list of strings or observation objects")
    else:
        records = [text]
    observations = []
    # Validate the complete import before writing its first observation.
    for record in records:
        if isinstance(record, str):
            record = {"text": record}
        if not isinstance(record, dict):
            raise ValueError("Each observation must be a string or object")
        content = record.get("text", record.get("content"))
        speaker = record.get("speaker", record.get("role", "user"))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each observation needs nonempty text/content")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker/role must be nonempty")
        observation = {"text": content, "speaker": speaker,
                       "timestamp": record.get("timestamp"), "dia_id": record.get("dia_id")}
        for key in ("timestamp", "dia_id"):
            if observation[key] is not None and not isinstance(observation[key], str):
                raise ValueError(f"{key} must be a string or null")
        observations.append(observation)
    return observations


def parser():
    cli = argparse.ArgumentParser(description="Independent lightweight TierMem with a Jev S/R router")
    cli.add_argument("--store", default=".jev_tiermem", help="Local memory directory")
    cli.add_argument("--session", default="default", help="Isolated memory session")
    cli.add_argument("--model", default=Config.model, help="OpenAI-compatible text model")
    cli.add_argument("--jev-model", default=Config.jev_model)
    cli.add_argument("--jev-api-url", default=os.getenv("JEV_API_URL"))
    sub = cli.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Append raw observations and create ordinary summaries")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON/JSONL observations, or plain text/Markdown")
    source.add_argument("--text")
    ingest.add_argument("--no-summary", action="store_true", help="Store raw data without a model call")
    sub.add_parser("summarize", help="Summarize pending raw pages")
    sub.add_parser("show", help="Display MEMORY.md")
    search = sub.add_parser("search", help="Search raw data locally without model calls")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=4)
    ask = sub.add_parser("ask", help="Summary first, Jev routing, bounded raw deepsearch if needed")
    ask.add_argument("query")
    ask.add_argument("--deepsearch", action="store_true", help="Start directly from raw evidence")
    ask.add_argument("--no-writeback", action="store_true")
    ask.add_argument("--max-rounds", type=int, default=Config.max_search_rounds)
    ask.add_argument("--max-raw-pages", type=int, default=Config.max_raw_pages)
    sub.add_parser("demo", help="Run a deterministic offline demo, without API keys")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "demo":
        from .demo import run
        print(json.dumps(run(), ensure_ascii=False, indent=2))
        return 0
    try:
        config = Config(model=args.model, jev_model=args.jev_model, jev_api_url=args.jev_api_url,
                        max_search_rounds=getattr(args, "max_rounds", Config.max_search_rounds),
                        max_raw_pages=getattr(args, "max_raw_pages", Config.max_raw_pages))
        with JevTierMem(args.store, args.session, config) as memory:
            if args.command == "ingest":
                observations = load_observations(args.input) if args.input else [{"text": args.text}]
                ids = [raw_id for record in observations for raw_id in memory.observe(**record)]
                result = {"raw_ids": ids}
                if not args.no_summary:
                    result["summary"] = memory.compact()
            elif args.command == "summarize":
                result = memory.compact()
            elif args.command == "ask":
                result = memory.answer(args.query, deepsearch=args.deepsearch,
                                       writeback=not args.no_writeback).to_dict()
            elif args.command == "search":
                if args.limit < 1:
                    raise ValueError("--limit must be positive")
                result = memory.search(args.query, args.limit)
            else:
                path = memory.store.memory_path
                print(path.read_text(encoding="utf-8") if path.exists() else "# Memory\n", end="")
                return 0
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except Exception as exc:
        print(f"jev_tiermem: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
