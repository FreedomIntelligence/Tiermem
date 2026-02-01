"""
总结阶段模块

负责计算评估指标、生成评估报告和汇总结果。
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from core.runner.scoring import compute_metrics
from core.runner.eval_result import generate_eval_report, print_eval_report


def aggregate_cost_metrics(write_logs: List[Dict[str, Any]], qa_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    汇总写入阶段和 QA 阶段的 cost metrics
    
    Args:
        write_logs: 写入日志列表
        qa_logs: QA 日志列表
        
    Returns:
        包含汇总 cost metrics 的字典
    """
    # 汇总写入阶段的 cost metrics
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
    
    # 汇总 QA 阶段的 cost metrics
    qa_total_latency_ms = 0
    qa_retrieval_latency_ms = 0
    qa_generation_latency_ms = 0
    qa_total_tokens_in = 0
    qa_total_tokens_out = 0
    qa_total_api_calls = 0
    # Router 相关统计
    qa_router_tokens_in = 0
    qa_router_tokens_out = 0
    qa_router_api_calls = 0
    qa_router_latency_ms = 0
    
    for log in qa_logs:
        cost = log.get("cost_metrics", {})
        qa_total_latency_ms += cost.get("online_total_latency_ms", 0)
        qa_retrieval_latency_ms += cost.get("online_retrieval_latency_ms", 0)
        qa_generation_latency_ms += cost.get("online_generation_latency_ms", 0)
        qa_total_tokens_in += cost.get("online_tokens_in", 0)
        qa_total_tokens_out += cost.get("online_tokens_out", 0)
        qa_total_api_calls += cost.get("online_api_calls", 0)
        # Router tokens（在 linked_view_system.py 的 cost 中单独统计）
        qa_router_tokens_in += cost.get("router_prompt_tokens", 0)
        qa_router_tokens_out += cost.get("router_completion_tokens", 0)
        qa_router_api_calls += cost.get("router_api_calls", 0)
        qa_router_latency_ms += cost.get("router_latency_ms", 0)
    
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
            # Router token 统计
            "router_tokens_in": qa_router_tokens_in,
            "router_tokens_out": qa_router_tokens_out,
            "router_api_calls": qa_router_api_calls,
            "router_latency_ms": qa_router_latency_ms,
            "router_latency_sec": qa_router_latency_ms / 1000.0,
            "avg_router_tokens_in_per_qa": qa_router_tokens_in / len(qa_logs) if qa_logs else 0,
            "avg_router_tokens_out_per_qa": qa_router_tokens_out / len(qa_logs) if qa_logs else 0,
            "avg_router_latency_per_qa_ms": qa_router_latency_ms / len(qa_logs) if qa_logs else 0,
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


def load_logs_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载所有日志
    
    Args:
        filepath: JSONL 文件路径
        
    Returns:
        日志列表
    """
    logs = []
    if not os.path.exists(filepath):
        return logs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"⚠ 警告: 读取日志文件时出错: {e}")
    
    return logs


def load_logs_from_multiple_files(pattern: str) -> List[Dict[str, Any]]:
    """
    从多个 JSONL 文件中加载所有日志（支持 glob 模式）
    
    Args:
        pattern: glob 模式，例如 "sessions/*_qa.jsonl"
        
    Returns:
        日志列表
    """
    import glob
    logs = []
    files = glob.glob(pattern)
    
    for filepath in files:
        try:
            file_logs = load_logs_from_jsonl(filepath)
            logs.extend(file_logs)
        except Exception as e:
            print(f"⚠ 警告: 读取日志文件 {filepath} 时出错: {e}")
            continue
    
    return logs


def run_summary_phase(
    benchmark_name: str,
    system_name: str,
    model_name: str,
    run_id: str,
    config: Dict[str, Any],
    output_path: Path,
    num_sessions: int
) -> Dict[str, Any]:
    """
    运行总结阶段，计算指标并生成报告
    
    支持两种日志格式：
    1. 旧格式：单个文件 write_logs.jsonl 和 qa_logs.jsonl
    2. 新格式：每个 session 独立的文件 sessions/{session_id}_write.jsonl 和 sessions/{session_id}_qa.jsonl
    
    Args:
        benchmark_name: benchmark 名称
        system_name: 系统名称
        model_name: 模型名称
        run_id: 运行 ID
        config: 配置字典
        output_path: 输出目录路径
        num_sessions: session 数量
        
    Returns:
        包含评估结果的字典
    """
    # 检查是否使用新的日志格式（每个 session 独立文件）
    sessions_dir = output_path / "sessions"
    if sessions_dir.exists() and any(sessions_dir.glob("*_qa.jsonl")):
        # 新格式：从多个 session 文件中加载日志
        print("检测到新的日志格式（每个 session 独立文件），加载所有 QA 日志...")
        qa_logs = load_logs_from_multiple_files(str(sessions_dir / "*_qa.jsonl"))
        print(f"加载了 {len(qa_logs)} 条 QA 日志")
        
        print("加载所有 Write 日志...")
        write_logs = load_logs_from_multiple_files(str(sessions_dir / "*_write.jsonl"))
        print(f"加载了 {len(write_logs)} 条 Write 日志")
    else:
        # 旧格式：从单个文件中加载日志（向后兼容）
        write_logs_path = str(output_path / "write_logs.jsonl")
        qa_logs_path = str(output_path / "qa_logs.jsonl")
        
        print("使用旧格式日志文件，加载所有 QA 日志...")
        qa_logs = load_logs_from_jsonl(qa_logs_path)
        
        print("加载所有 Write 日志...")
        write_logs = load_logs_from_jsonl(write_logs_path)
    
    # 计算总体指标
    if len(qa_logs) == 0:
        print("⚠ 警告: 没有 QA 日志，无法计算指标")
        metrics = {
            "exact_match": 0.0,
            "f1": 0.0,
            "num_samples": 0
        }
        eval_report = None
    else:
        # 提取 predictions 和 ground_truths，添加防御性检查
        predictions = []
        ground_truths = []
        for log in qa_logs:
            if "model_response" not in log:
                print(f"⚠ 警告: QA 日志缺少 'model_response' 字段: {log.get('query_id', 'unknown')}")
                continue
            if "ground_truth" not in log:
                print(f"⚠ 警告: QA 日志缺少 'ground_truth' 字段: {log.get('query_id', 'unknown')}")
                continue
            predictions.append(log["model_response"])
            ground_truths.append(log["ground_truth"])
        
        if len(predictions) == 0:
            print("⚠ 警告: 没有有效的 QA 记录，无法计算指标")
            metrics = {
                "exact_match": 0.0,
                "f1": 0.0,
                "num_samples": 0
            }
            eval_report = None
        else:
            # 基础指标（兼容现有接口）
            metrics = compute_metrics(predictions, ground_truths, benchmark=benchmark_name)
            
            # 生成详细评估报告（特别是 LoCoMo 需要按 category 分类）
            eval_report = generate_eval_report(
                qa_logs=qa_logs,
                benchmark=benchmark_name,
                output_path=output_path
            )
    
    # 汇总 cost metrics
    cost_summary = aggregate_cost_metrics(write_logs, qa_logs)
    
    # 保存汇总结果
    summary = {
        "run_id": run_id,
        "benchmark": benchmark_name,
        "system": system_name,
        "model": model_name,
        "config": config,
        "metrics": metrics,
        "eval_report": eval_report,  # 包含详细评估报告
        "cost_summary": cost_summary,  # 包含 cost metrics 汇总
        "num_sessions": num_sessions,
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
    
    # 打印评估报告
    if eval_report is not None:
        print_eval_report(eval_report, benchmark=benchmark_name)
    
    return summary


