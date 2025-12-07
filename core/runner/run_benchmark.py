"""
统一评估框架

运行所有baseline系统在所有benchmark上的评估。
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import time
from tqdm import tqdm
from core.systems.base import MemorySystem, Turn, ObserveResult
from core.runner.logging_utils import (
    make_write_record, make_qa_record, save_logs_to_jsonl,
    append_log_to_jsonl, load_processed_sessions
)
from core.runner.scoring import compute_metrics
from core.runner.eval_result import generate_eval_report, print_eval_report


def aggregate_cost_metrics(write_logs: List[Dict[str, Any]], qa_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    汇总写入阶段和QA阶段的cost metrics
    
    Args:
        write_logs: 写入日志列表
        qa_logs: QA日志列表
        
    Returns:
        包含汇总cost metrics的字典
    """
    # 汇总写入阶段的cost metrics
    write_total_latency_ms = 0
    write_total_tokens_in = 0
    write_total_tokens_out = 0
    write_total_api_calls = 0
    
    for log in write_logs:
        cost = log.get("cost_metrics", {})
        write_total_latency_ms += cost.get("total_latency_ms", 0)
        write_total_tokens_in += cost.get("total_tokens_in", 0)
        write_total_tokens_out += cost.get("total_tokens_out", 0)
        write_total_api_calls += cost.get("api_calls_count", 0)
    
    # 汇总QA阶段的cost metrics
    qa_total_latency_ms = 0
    qa_retrieval_latency_ms = 0
    qa_generation_latency_ms = 0
    qa_total_tokens_in = 0
    qa_total_tokens_out = 0
    qa_total_api_calls = 0
    
    for log in qa_logs:
        cost = log.get("cost_metrics", {})
        qa_total_latency_ms += cost.get("online_total_latency_ms", 0)
        qa_retrieval_latency_ms += cost.get("online_retrieval_latency_ms", 0)
        qa_generation_latency_ms += cost.get("online_generation_latency_ms", 0)
        qa_total_tokens_in += cost.get("online_tokens_in", 0)
        qa_total_tokens_out += cost.get("online_tokens_out", 0)
        qa_total_api_calls += cost.get("online_api_calls", 0)
    
    # 计算总体统计
    total_latency_ms = write_total_latency_ms + qa_total_latency_ms
    total_tokens_in = write_total_tokens_in + qa_total_tokens_in
    total_tokens_out = write_total_tokens_out + qa_total_tokens_out
    total_api_calls = write_total_api_calls + qa_total_api_calls
    
    return {
        "write_phase": {
            "total_latency_ms": write_total_latency_ms,
            "total_latency_sec": write_total_latency_ms / 1000.0,
            "total_tokens_in": write_total_tokens_in,
            "total_tokens_out": write_total_tokens_out,
            "total_api_calls": write_total_api_calls,
            "avg_latency_per_turn_ms": write_total_latency_ms / len(write_logs) if write_logs else 0,
            "avg_tokens_in_per_turn": write_total_tokens_in / len(write_logs) if write_logs else 0,
            "avg_tokens_out_per_turn": write_total_tokens_out / len(write_logs) if write_logs else 0
        },
        "qa_phase": {
            "total_latency_ms": qa_total_latency_ms,
            "total_latency_sec": qa_total_latency_ms / 1000.0,
            "retrieval_latency_ms": qa_retrieval_latency_ms,
            "retrieval_latency_sec": qa_retrieval_latency_ms / 1000.0,
            "generation_latency_ms": qa_generation_latency_ms,
            "generation_latency_sec": qa_generation_latency_ms / 1000.0,
            "total_tokens_in": qa_total_tokens_in,
            "total_tokens_out": qa_total_tokens_out,
            "total_api_calls": qa_total_api_calls,
            "avg_latency_per_qa_ms": qa_total_latency_ms / len(qa_logs) if qa_logs else 0,
            "avg_tokens_in_per_qa": qa_total_tokens_in / len(qa_logs) if qa_logs else 0,
            "avg_tokens_out_per_qa": qa_total_tokens_out / len(qa_logs) if qa_logs else 0
        },
        "overall": {
            "total_latency_ms": total_latency_ms,
            "total_latency_sec": total_latency_ms / 1000.0,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "total_api_calls": total_api_calls
        }
    }


def run_benchmark(
    system: MemorySystem,
    dataset_module,
    benchmark_name: str,
    run_id: str,
    config: Dict[str, Any],
    output_dir: str = "results",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    运行benchmark评估
    
    Args:
        system: MemorySystem实例
        dataset_module: 数据集模块（包含iter_sessions函数）
        benchmark_name: benchmark名称
        run_id: 运行ID
        config: 配置字典
        output_dir: 输出目录
        limit: 限制处理的session数量（用于测试）
        
    Returns:
        包含评估结果的字典
    """
    model_name = config.get("model_name", "unknown")
    split = config.get("split", "test")
    
    # 准备输出路径（提前创建，用于实时写入）
    output_path = Path(output_dir) / benchmark_name / system.get_system_name() / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    write_logs_path = str(output_path / "write_logs.jsonl")
    qa_logs_path = str(output_path / "qa_logs.jsonl")
    
    # 加载已处理的session（断点续跑）
    processed_sessions = load_processed_sessions(qa_logs_path, session_key="session_id")
    if processed_sessions:
        print(f"发现 {len(processed_sessions)} 个已处理的session，将跳过...")
    
    # 迭代所有sessions
    sessions = list(dataset_module.iter_sessions(split=split, limit=limit))
    print(f"Processing {len(sessions)} sessions...")
    
    # Session级别的进度条
    session_pbar = tqdm(sessions, desc="Sessions", unit="session", ncols=100)
    for session in session_pbar:
        session_id = session["session_id"]
        session_pbar.set_description(f"Session: {session_id[:30]}...")
        
        # 检查是否已处理过（断点续跑）
        if session_id in processed_sessions:
            session_pbar.set_postfix({"status": "skipped"})
            continue
        
        # 重置系统状态
        system.reset(session_id)
        
        # 1. 记忆写入阶段
        # 系统可以通过preferred_turns_key指定使用哪个粒度的turns
        # 默认使用"turns"（细粒度），GAM可以使用"session_chunks"（粗粒度）
        turns_key = getattr(system, "preferred_turns_key", "turns")
        turns = session.get(turns_key, session.get("turns", []))
        
        if len(turns) > 0:
            turn_pbar = tqdm(turns, desc="  Writing turns", leave=False, unit="turn", ncols=100)
            for turn_idx, turn_data in enumerate(turn_pbar):
                turn = Turn(
                    speaker=turn_data.get("speaker", "user"),
                    text=turn_data.get("text", ""),
                    timestamp=turn_data.get("timestamp")
                )
                
                # 调用observe（GAM会在observe内部执行memorize）
                observe_result = system.observe(turn)
                
                # 记录写入日志
                write_log = make_write_record(
                    run_id=run_id,
                    system_name=system.get_system_name(),
                    model_name=model_name,
                    benchmark=benchmark_name,
                    session_id=session_id,
                    turn_id=turn_idx,
                    raw_input_text=turn.text,
                    observe_result=observe_result
                )
                # 实时写入到文件（不保存在内存中，节省内存）
                append_log_to_jsonl(write_log, write_logs_path)
            turn_pbar.close()
        
        # 2. QA阶段
        qa_pairs = session["qa_pairs"]
        if len(qa_pairs) > 0:
            qa_pbar = tqdm(qa_pairs, desc="  Answering QAs", leave=False, unit="qa", ncols=100)
            for qa in qa_pbar:
                query_id = qa["query_id"]
                question = qa["question"]
                ground_truth = qa["ground_truth"]
                category = qa.get("category")  # 提取category（用于LoCoMo评估）
                
                # 调用answer（确保category在meta中）
                answer_meta = qa.get("meta", {}).copy()
                if category is not None:
                    answer_meta["category"] = category
                answer_result = system.answer(question, meta=answer_meta)
                
                # 计算score（简单使用exact match）
                from core.runner.scoring import exact_match_score
                score = exact_match_score(answer_result.answer, ground_truth)
                
                # 记录QA日志
                qa_log = make_qa_record(
                    run_id=run_id,
                    system_name=system.get_system_name(),
                    model_name=model_name,
                    benchmark=benchmark_name,
                    query_id=query_id,
                    question=question,
                    ground_truth=ground_truth,
                    answer_result=answer_result,
                    score=score
                )
                # 添加category信息（如果存在）
                if category is not None:
                    qa_log["category"] = category
                # 实时写入到文件（不保存在内存中，节省内存）
                append_log_to_jsonl(qa_log, qa_logs_path)
            qa_pbar.close()
        
        # 标记该session已处理
        processed_sessions.add(session_id)
    
    session_pbar.close()
    
    # 重新加载所有QA日志（包括之前已处理的）用于计算指标
    print("加载所有QA日志用于计算指标...")
    qa_logs = []
    if os.path.exists(qa_logs_path):
        import json
        with open(qa_logs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        qa_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # 3. 计算总体指标
    if len(qa_logs) == 0:
        print("⚠ 警告: 没有QA日志，无法计算指标")
        metrics = {
            "exact_match": 0.0,
            "f1": 0.0,
            "num_samples": 0
        }
        eval_report = None
    else:
        # 提取predictions和ground_truths，添加防御性检查
        predictions = []
        ground_truths = []
        for log in qa_logs:
            if "model_response" not in log:
                print(f"⚠ 警告: QA日志缺少'model_response'字段: {log.get('query_id', 'unknown')}")
                continue
            if "ground_truth" not in log:
                print(f"⚠ 警告: QA日志缺少'ground_truth'字段: {log.get('query_id', 'unknown')}")
                continue
            predictions.append(log["model_response"])
            ground_truths.append(log["ground_truth"])
        
        if len(predictions) == 0:
            print("⚠ 警告: 没有有效的QA记录，无法计算指标")
            metrics = {
                "exact_match": 0.0,
                "f1": 0.0,
                "num_samples": 0
            }
            eval_report = None
        else:
            # 基础指标（兼容现有接口）
            metrics = compute_metrics(predictions, ground_truths, benchmark=benchmark_name)
            
            # 生成详细评估报告（特别是LoCoMo需要按category分类）
            eval_report = generate_eval_report(
                qa_logs=qa_logs,
                benchmark=benchmark_name,
                output_path=None  # 稍后在保存时指定路径
            )
    
    # 4. 日志已经实时写入，这里只需要重新加载write_logs用于统计
    write_logs = []
    if os.path.exists(write_logs_path):
        import json
        with open(write_logs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        write_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # 5. 生成并保存详细评估报告（如果存在）
    if eval_report is not None:
        # 重新生成报告并保存到文件
        eval_report = generate_eval_report(
            qa_logs=qa_logs,
            benchmark=benchmark_name,
            output_path=output_path
        )
        with open(output_path / "eval_report.json", 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)
        
        # 打印评估报告
        print_eval_report(eval_report, benchmark=benchmark_name)
    
    # 6. 汇总cost metrics
    cost_summary = aggregate_cost_metrics(write_logs, qa_logs)
    
    # 7. 保存汇总结果
    summary = {
        "run_id": run_id,
        "benchmark": benchmark_name,
        "system": system.get_system_name(),
        "model": model_name,
        "config": config,
        "metrics": metrics,
        "eval_report": eval_report,  # 包含详细评估报告
        "cost_summary": cost_summary,  # 包含cost metrics汇总
        "num_sessions": len(sessions),
        "num_write_logs": len(write_logs),
        "num_qa_logs": len(qa_logs),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(output_path / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"  Basic Metrics: {metrics}")
    print(f"\n  Cost Summary:")
    print(f"    Write Phase: {cost_summary['write_phase']['total_latency_sec']:.2f}s, "
          f"{cost_summary['write_phase']['total_tokens_in']} in, {cost_summary['write_phase']['total_tokens_out']} out, "
          f"{cost_summary['write_phase']['total_api_calls']} API calls")
    print(f"    QA Phase: {cost_summary['qa_phase']['total_latency_sec']:.2f}s, "
          f"{cost_summary['qa_phase']['total_tokens_in']} in, {cost_summary['qa_phase']['total_tokens_out']} out, "
          f"{cost_summary['qa_phase']['total_api_calls']} API calls")
    print(f"    Overall: {cost_summary['overall']['total_latency_sec']:.2f}s, "
          f"{cost_summary['overall']['total_tokens']} total tokens, "
          f"{cost_summary['overall']['total_api_calls']} total API calls")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--system", type=str, required=True, help="System name (rawllm, mem0, gam, etc.)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (locomo, hotpotqa, memory_agent_bench)")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/validation/test)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generated)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions (for testing)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="Model name")
    
    args = parser.parse_args()
    
    # 生成run_id
    if args.run_id is None:
        args.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 导入数据集模块
    if args.benchmark == "locomo":
        from core.datasets import locomo as dataset_module
    elif args.benchmark == "hotpotqa":
        from core.datasets import hotpotqa as dataset_module
    elif args.benchmark == "memory_agent_bench":
        from core.datasets import memory_agent_bench as dataset_module
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # 导入系统模块
    if args.system == "rawllm":
        from core.systems.rawllm_adapter import RawLLMSystem
        system = RawLLMSystem(config={"model_name": args.model_name})
    elif args.system == "mem0":
        from core.systems.mem0_adapter import Mem0System
        system = Mem0System(config={"model_name": args.model_name})
    elif args.system == "gam":
        from core.systems.gam_adapter import GAMSystem
        system = GAMSystem(config={
            "memory_model": args.model_name,
            "research_model": args.model_name,
            "working_model": args.model_name,
            "max_research_iters": 3
        })
    else:
        raise ValueError(f"Unknown system: {args.system}")
    
    # 运行评估
    config = {
        "model_name": args.model_name,
        "split": args.split
    }
    
    summary = run_benchmark(
        system=system,
        dataset_module=dataset_module,
        benchmark_name=args.benchmark,
        run_id=args.run_id,
        config=config,
        output_dir=args.output_dir,
        limit=args.limit
    )
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    from typing import Optional
    main()

