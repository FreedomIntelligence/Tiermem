"""
QA 阶段模块

负责处理 QA 阶段，对每个 session 的问题进行回答。
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from tqdm import tqdm

from core.systems.base import MemorySystem
from core.runner.logging_utils import (
    make_qa_record,
    append_log_to_jsonl,
    load_processed_sessions
)
from core.runner.scoring import exact_match_score


def run_qa_phase(
    system: MemorySystem,
    sessions: List[Dict[str, Any]],
    run_id: str,
    benchmark_name: str,
    model_name: str,
    output_path: Path,
    processed_sessions: Optional[Set[str]] = None
) -> Set[str]:
    """
    运行 QA 阶段，处理所有 sessions 的问题回答
    
    Args:
        system: MemorySystem 实例
        sessions: 会话列表
        run_id: 运行 ID
        benchmark_name: benchmark 名称
        model_name: 模型名称
        output_path: 输出目录路径
        processed_sessions: 已处理的 session 集合（用于断点续跑）
        
    Returns:
        已处理的 session_id 集合
    """
    qa_logs_path = str(output_path / "qa_logs.jsonl")
    
    # 加载已处理的 session（断点续跑）
    if processed_sessions is None:
        processed_sessions = load_processed_sessions(qa_logs_path, session_key="session_id")
    
    if processed_sessions:
        print(f"发现 {len(processed_sessions)} 个已处理的 session，将跳过...")
    
    # Session 级别的进度条
    session_pbar = tqdm(sessions, desc="Sessions (QA)", unit="session", ncols=100)
    session_qa_count = 0  # 用于实时进度显示
    
    for session in session_pbar:
        session_id = session["session_id"]
        session_pbar.set_description(f"Session: {session_id[:30]}...")
        
        # 跳过已处理的 session
        if session_id in processed_sessions:
            session_pbar.set_postfix({"status": "skipped"})
            continue
        
        # 确保系统已加载该 session
        try:
            system.load(session_id)
        except Exception as e:
            print(f"\n⚠ 错误: Session {session_id} 的 load() 失败: {e}")
            print(f"  跳过该 session，继续处理下一个...")
            import traceback
            traceback.print_exc()
            session_pbar.set_postfix({"status": "load_failed"})
            processed_sessions.add(session_id)
            error_log = {
                "run_id": run_id,
                "benchmark": benchmark_name,
                "system": system.get_system_name(),
                "session_id": session_id,
                "error_type": "load_failed",
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            error_log_path = str(output_path / "errors.jsonl")
            append_log_to_jsonl(error_log, error_log_path)
            continue
        
        # 处理 QA pairs
        qa_pairs = session.get("qa_pairs", [])
        if len(qa_pairs) > 0:
            qa_pbar = tqdm(qa_pairs, desc="  Answering QAs", leave=False, unit="qa", ncols=100)
            for qa in qa_pbar:
                query_id = qa["query_id"]
                question = qa["question"]
                ground_truth = qa["ground_truth"]
                category = qa.get("category")  # 提取 category（用于 LoCoMo 评估）
                
                # 调用 answer（确保 category 在 meta 中）
                answer_meta = qa.get("meta", {}).copy()
                if category is not None:
                    answer_meta["category"] = category
                answer_result = system.answer(question, meta=answer_meta)
                
                # 计算 score（简单使用 exact match）
                score = exact_match_score(answer_result.answer, ground_truth)
                
                # 记录 QA 日志
                qa_log = make_qa_record(
                    run_id=run_id,
                    system_name=system.get_system_name(),
                    session_id=session_id,
                    model_name=model_name,
                    benchmark=benchmark_name,
                    query_id=query_id,
                    question=question,
                    ground_truth=ground_truth,
                    answer_result=answer_result,
                    score=score
                )
                # 添加 category 信息（如果存在）
                if category is not None:
                    qa_log["category"] = category
                # 实时写入到文件（不保存在内存中，节省内存）
                append_log_to_jsonl(qa_log, qa_logs_path)
                session_qa_count += 1
            qa_pbar.close()
        
        # 更新进度条显示（显示已完成的 QA 数量）
        session_pbar.set_postfix({"qa_completed": session_qa_count})
        
        # 标记该 session 已处理
        processed_sessions.add(session_id)
    
    session_pbar.close()
    
    return processed_sessions







