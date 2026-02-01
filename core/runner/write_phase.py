"""
写入阶段模块

负责处理记忆写入阶段，将对话 turns 写入到 memory system 中。
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from tqdm import tqdm

from core.systems.base import MemorySystem, Turn
from core.runner.logging_utils import (
    make_write_record,
    append_log_to_jsonl,
    load_processed_sessions
)


def run_write_phase(
    system: MemorySystem,
    sessions: List[Dict[str, Any]],
    run_id: str,
    benchmark_name: str,
    model_name: str,
    output_path: Path,
    processed_sessions: Optional[Set[str]] = None
) -> Set[str]:
    """
    运行写入阶段，处理所有 sessions 的 turns 写入
    
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
    write_logs_path = str(output_path / "write_logs.jsonl")
    
    # 加载已处理的 session（断点续跑）
    if processed_sessions is None:
        processed_sessions = load_processed_sessions(write_logs_path, session_key="session_id")
    
    if processed_sessions:
        print(f"发现 {len(processed_sessions)} 个已处理的 session，将跳过...")
    
    # Session 级别的进度条
    session_pbar = tqdm(sessions, desc="Sessions (Write)", unit="session", ncols=100)
    
    for session in session_pbar:
        session_id = session["session_id"]
        session_pbar.set_description(f"Session: {session_id[:30]}...")
        
        # 跳过已处理的 session
        if session_id in processed_sessions:
            session_pbar.set_postfix({"status": "skipped"})
            continue
        
        try:
            system.reset(session_id)
        except Exception as e:
            print(f"\n⚠ 错误: Session {session_id} 的 load() 失败: {e}")
            print(f"  跳过该 session，继续处理下一个...")
            import traceback
            traceback.print_exc()
            session_pbar.set_postfix({"status": "load_failed"})
            # 标记为已处理（避免重复尝试），但记录错误
            processed_sessions.add(session_id)
            # 记录错误日志
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
            continue  # 跳过当前 session，继续下一个
        
        # 系统可以通过 preferred_turns_key 指定使用哪个粒度的 turns
        # 默认使用 "turns"（细粒度），GAM 可以使用 "session_chunks"（粗粒度）
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
                
                # 调用 observe（GAM 会在 observe 内部执行 memorize）
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
        
        # 尝试执行 auto_summary（如果系统支持）
        try:
            system.auto_summary()
        except Exception as e:
            print(f"\n⚠ 警告: Session {session_id} 的 auto_summary() 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 标记该 session 已处理
        processed_sessions.add(session_id)
        session_pbar.set_postfix({"status": "completed"})
    
    session_pbar.close()
    
    return processed_sessions


