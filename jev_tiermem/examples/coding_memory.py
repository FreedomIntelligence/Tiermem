"""Reproduce and fix a CSV import bug, then recall the work in a new client.

From the Tiermem checkout, after installing jev_tiermem[live]:
    python jev_tiermem/examples/coding_memory.py
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

if __name__ == "__main__":
    if __package__:
        from ._bootstrap import ensure_runtime
    else:
        from _bootstrap import ensure_runtime
    ensure_runtime(__file__)

from jev_tiermem import Config, JevTierMem
from jev_tiermem.examples._coding_scenario import CodingScenario, QUESTION, answer_has_details


ROOT = Path(__file__).resolve().parents[2]


def run(args):
    config = Config(
        model=args.model, jev_model=args.jev_model, jev_api_url=args.jev_api_url,
        note_max_chars=240, timeout=60, max_output_tokens=2048,
    )
    session = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    directory = args.store.resolve() / session
    directory.mkdir(parents=True)
    report_path = directory / "result.json"
    report = {"session": session, "question": QUESTION, "model": config.model}

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[1/5] 复现 CSV 导入失败 → 脚本应用一行修复 → 运行回归测试", flush=True)
    scenario = CodingScenario(directory)
    scenario.call("inspect_project", {})
    before = scenario.call("run_import_tests", {})
    print("修复前：", flush=True)
    print(before["text"], flush=True)
    source = (scenario.workspace / "contacts.py").read_text(encoding="utf-8")
    patch = scenario.call("write_importer", {"source": source.replace('encoding="utf-8"', 'encoding="utf-8-sig"')})
    print(patch["text"], flush=True)
    after = scenario.call("run_import_tests", {})
    print("修复后：", flush=True)
    print(after["text"], flush=True)
    log = scenario.call("coding_history", {})["text"]
    (directory / "coding.log").write_text(log, encoding="utf-8")
    report["workspace"] = str(scenario.workspace)
    report["coding_checks"] = scenario.checks()
    save()
    if not all(report["coding_checks"].values()):
        print(f"复现或修复检查未通过，请查看 {directory / 'coding.log'}")
        return 2

    print("[2/5] 写入原始工具输出 → raw history", flush=True)
    with JevTierMem(directory / "memory", session, config) as memory:
        raw_ids = memory.observe(log, speaker="tool", dia_id="csv-import-fix")
        report["raw_ids"] = raw_ids
        report["stored_raw"] = memory.store.get(raw_ids)
        report["raw_preserved"] = "".join(page["text"] for page in report["stored_raw"]) == log
        print("raw_ids:", ", ".join(raw_ids), flush=True)
        print("已保存的原文（从 raw.sqlite3 读回）：", flush=True)
        for page in report["stored_raw"]:
            print(f"[{page['id']}]", flush=True)
            print(page["text"], end="" if page["text"].endswith("\n") else "\n", flush=True)

        print("[3/5] 普通文本模型生成 summary，并绑定原文索引", flush=True)
        report["summary_write"] = memory.compact()
        report["notes"] = memory.store.notes()
        report["memory_path"] = str(memory.store.memory_path)
        report["memory_markdown"] = memory.store.memory_path.read_text(encoding="utf-8")
        covered = {raw_id for note in report["notes"] for raw_id in note["raw_ids"]}
        report["sources_indexed"] = set(raw_ids) <= covered
        print("已保存的 MEMORY.md（包含 summary 和原文索引）：", flush=True)
        print(report["memory_markdown"], flush=True)
        save()
        if not report["raw_preserved"] or not report["sources_indexed"]:
            print(f"记忆写入未完成，详情：{report_path}")
            return 2

    print("[4/5] 关闭旧客户端，重新打开相同 session，仅从记忆回查", flush=True)
    print("问题：", QUESTION, flush=True)
    with JevTierMem(directory / "memory", session, config) as memory:
        # No log text or previous model conversation is supplied to the new client.
        answer = memory.answer(QUESTION, writeback=False)
        report["answer"] = answer.to_dict()
        report["raw_evidence"] = memory.store.get(answer.trace["raw_ids"])
        report["jev_models"] = sorted(memory.router.models_seen)

    print("[5/5] Jev 路由与回答", flush=True)
    for decision in answer.trace["decisions"]:
        source = "summary" if decision["stage"] == "summary" else "raw history"
        verdict = (f"判断失败：{decision['error']}" if decision.get("error")
                   else "足够" if decision["accepted"] else "不足")
        print(f"{source}: {verdict}", flush=True)
    if any(decision.get("error") for decision in answer.trace["decisions"]):
        print("Jev 判断未完成；请检查接口地址、模型名与 key 是否属于同一服务。", flush=True)
        print("兼容服务需要配置 JEV_API_URL 或 --jev-api-url；未设置时使用 TypeSafe 官方 SDK。", flush=True)
    print("route:", answer.route, flush=True)
    print("最终回答（基于检索返回的记忆证据）：", flush=True)
    print(answer.answer, flush=True)
    print("证据：", ", ".join(answer.citations), flush=True)
    sufficient = bool(answer.trace["decisions"] and answer.trace["decisions"][-1]["accepted"])
    report["passed"] = bool(
        answer.supported and sufficient and answer.citations
        and answer_has_details(answer.answer)
    )
    save()
    print("结果：", "写入和回查均通过" if report["passed"] else "未通过全部检查，请查看实际结果", flush=True)
    print("笔记：", report["memory_path"], flush=True)
    print("完整结果：", report_path, flush=True)
    return 0 if report["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "jev_tiermem/.runs/coding-demo")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", Config.model))
    parser.add_argument("--jev-model", default=os.getenv("TYPESAFE_DEFAULT_MODEL", Config.jev_model))
    parser.add_argument("--jev-api-url", default=os.getenv("JEV_API_URL"))
    args = parser.parse_args()
    missing = [name for name in ("OPENAI_API_KEY", "TYPESAFE_API_KEY") if not os.getenv(name)]
    if missing:
        parser.error("请先按 README 配置环境变量：" + ", ".join(missing))
    try:
        return run(args)
    except Exception as exc:
        # Provider messages can contain request details; never print their bodies.
        print(f"示例未完成：{type(exc).__name__}。已写入的记忆保留在 {args.store}。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
