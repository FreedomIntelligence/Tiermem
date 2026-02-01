"""
并发版本的 benchmark 评估框架

采用"每个 Session 作为一个原子任务（Write + QA）"的并发模式：
- 每个 Session 先执行 Write，然后立即执行 QA
- Session 内部的 turns 和 QA pairs 都支持并发处理（通过锁保护）
- 使用 session 级别的锁保护同一 session 内的并发操作（PageStore 和 index）
- 针对 Qdrant 错误加强重试机制（最多5次，指数退避，Qdrant错误延迟加倍）
- 这样可以快速反馈与止损、提高数据完整性与断点续跑能力

使用 ThreadPoolExecutor 或 ProcessPoolExecutor 来并发执行多个 Session。
"""
import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import copy
import glob

from core.systems.base import MemorySystem
from core.runner.write_phase import run_write_phase
from core.runner.qa_phase import run_qa_phase
from core.runner.summary_phase import run_summary_phase

logger = logging.getLogger(__name__)


# 线程安全的日志写入锁
_log_lock = threading.Lock()

# 为每个session创建独立的锁（用于保护同一session内的并发操作）
_session_locks: Dict[str, threading.Lock] = {}
_session_locks_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    """获取或创建session级别的锁"""
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _is_qdrant_error(error: Exception) -> bool:
    """判断是否是Qdrant相关的错误"""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Qdrant相关错误关键词
    qdrant_keywords = [
        "qdrant",
        "connection",
        "timeout",
        "connection timeout",
        "connection refused",
        "connection reset",
        "connection error",
        "network",
        "unavailable",
        "retry",
        "rate limit",
        "too many requests",
        "service unavailable",
        "503",
        "502",
        "500",
    ]
    
    return any(keyword in error_str for keyword in qdrant_keywords) or "connection" in error_type


def _process_single_turn(
    turn_data: tuple,
    session_id: str,
    system: MemorySystem,
    run_id: str,
    benchmark_name: str,
    model_name: str,
    session_write_logs_path: str,
    session_error_log_path: str,
    session_lock: threading.Lock,
) -> tuple[int, bool, Optional[str]]:
    """
    处理单个turn（带锁保护和重试机制）
    
    Args:
        turn_data: (turn_idx, turn_data_dict) 元组
        session_id: session ID
        system: system实例（需要锁保护）
        session_lock: session级别的锁
    
    Returns:
        (turn_idx, success, error_message)
    """
    turn_idx, turn_data_dict = turn_data
    
    from core.systems.base import Turn
    turn = Turn(
        dia_id=turn_data_dict.get("dia_id", ""),
        speaker=turn_data_dict.get("speaker", "user"),
        text=turn_data_dict.get("text", ""),
        timestamp=turn_data_dict.get("timestamp")
    )
    
    # 调用 observe（带重试机制，特别针对Qdrant错误）
    max_retries = 5  # 增加重试次数，特别是Qdrant可能阻塞
    retry_delay = 2  # 初始延迟（秒）
    observe_result = None
    
    for retry_count in range(max_retries):
        try:
            # PageStore 和 index 操作已经在内部有锁保护，不需要外部锁
            # 这样可以实现真正的并发执行
            observe_result = system.observe(turn)
            break  # 成功则退出重试循环
        except Exception as e:
            is_qdrant_err = _is_qdrant_error(e)
            
            if retry_count < max_retries - 1:
                # 指数退避：2秒, 4秒, 8秒, 16秒, 32秒
                # 对于Qdrant错误，使用更长的延迟
                base_delay = retry_delay * (2 ** retry_count)
                wait_time = base_delay * (2 if is_qdrant_err else 1)  # Qdrant错误延迟加倍
                
                error_type = "Qdrant" if is_qdrant_err else "General"
                logger.warning(
                    f"[{session_id}] Turn {turn_idx} observe failed ({error_type} error, attempt {retry_count + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                # 最后一次重试也失败
                error_msg = str(e)
                logger.error(
                    f"[{session_id}] Turn {turn_idx} observe failed after {max_retries} attempts: {e}"
                )
                # 记录错误日志
                error_log = {
                    "run_id": run_id,
                    "benchmark": benchmark_name,
                    "system": system.get_system_name(),
                    "session_id": session_id,
                    "turn_id": turn_idx,
                    "error_type": "observe_failed",
                    "error_message": error_msg,
                    "is_qdrant_error": is_qdrant_err,
                    "retry_count": max_retries,
                    "timestamp": datetime.utcnow().isoformat()
                }
                from core.runner.logging_utils import append_log_to_jsonl
                append_log_to_jsonl(error_log, session_error_log_path)
                
                return (turn_idx, False, error_msg)
    
    # 记录写入日志（写入 session 专属文件，不需要锁）
    from core.runner.logging_utils import make_write_record, append_log_to_jsonl
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
    
    # 不需要锁，因为每个 session 有独立的文件
    append_log_to_jsonl(write_log, session_write_logs_path)
    
    return (turn_idx, True, None)


def _process_single_qa_pair(
    qa: Dict[str, Any],
    session_id: str,
    system: MemorySystem,
    run_id: str,
    benchmark_name: str,
    model_name: str,
    qa_logs_path: str,
    error_log_path: str,
) -> tuple[str, bool, Optional[str]]:
    """
    处理单个 QA pair（复用同一个 system 实例）
    
    注意：直接使用传入的 system 实例，不重新创建或 load。
    这样可以避免重复 load 的开销，因为 Write 阶段已经完成，system 实例已经包含了所有数据。
    
    Args:
        system: 已加载 session 的 system 实例（复用 Write 阶段的实例）
    
    Returns:
        (query_id, success, error_message)
    """
    query_id = qa["query_id"]
    question = qa["question"]
    ground_truth = qa["ground_truth"]
    category = qa.get("category")
    
    # 直接使用传入的 system 实例（不重新创建或 load）
    answer_meta = qa.get("meta", {}).copy()
    if category is not None:
        answer_meta["category"] = category
    
    # 调用 answer（带重试机制，特别针对Qdrant错误）
    max_retries = 5  # 增加重试次数，特别是Qdrant可能阻塞
    retry_delay = 2  # 初始延迟（秒）
    answer_result = None
    
    # 获取session级别的锁（保护system.answer()调用）
    session_lock = _get_session_lock(session_id)
    
    for retry_count in range(max_retries):
        try:
            # 使用锁保护system.answer()调用（防止并发访问同一session的index）
            with session_lock:
                answer_result = system.answer(question, meta=answer_meta)
            break  # 成功则退出重试循环
        except Exception as e:
            is_qdrant_err = _is_qdrant_error(e)
            
            if retry_count < max_retries - 1:
                # 指数退避：2秒, 4秒, 8秒, 16秒, 32秒
                # 对于Qdrant错误，使用更长的延迟
                base_delay = retry_delay * (2 ** retry_count)
                wait_time = base_delay * (2 if is_qdrant_err else 1)  # Qdrant错误延迟加倍
                
                error_type = "Qdrant" if is_qdrant_err else "General"
                logger.warning(
                    f"[{session_id}] QA {query_id} answer failed ({error_type} error, attempt {retry_count + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                # 最后一次重试也失败
                error_msg = str(e)
                logger.error(
                    f"[{session_id}] QA {query_id} answer failed after {max_retries} attempts: {e}"
                )
                # 记录错误日志（不需要锁，因为每个 session 有独立的文件）
                from datetime import datetime
                error_log = {
                    "run_id": run_id,
                    "benchmark": benchmark_name,
                    "system": system.get_system_name(),
                    "session_id": session_id,
                    "query_id": query_id,
                    "error_type": "qa_pair_failed",
                    "error_message": error_msg,
                    "is_qdrant_error": is_qdrant_err,
                    "retry_count": max_retries,
                    "timestamp": datetime.utcnow().isoformat()
                }
                from core.runner.logging_utils import append_log_to_jsonl
                append_log_to_jsonl(error_log, error_log_path)
                
                return (query_id, False, error_msg)
    
    # 计算 score
    try:
        from core.runner.scoring import exact_match_score
        score = exact_match_score(answer_result.answer, ground_truth)
        
        # 记录 QA 日志（线程安全）
        from core.runner.logging_utils import make_qa_record, append_log_to_jsonl
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
        if category is not None:
            qa_log["category"] = category
        
        # 不需要锁，因为每个 session 有独立的文件
        append_log_to_jsonl(qa_log, qa_logs_path)
        
        return (query_id, True, None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{session_id}] QA {query_id} score calculation or logging failed: {e}")
        # 记录错误日志（不需要锁，因为每个 session 有独立的文件）
        from datetime import datetime
        error_log = {
            "run_id": run_id,
            "benchmark": benchmark_name,
            "system": system.get_system_name(),
            "session_id": session_id,
            "query_id": query_id,
            "error_type": "qa_pair_post_processing_failed",
            "error_message": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
        from core.runner.logging_utils import append_log_to_jsonl
        append_log_to_jsonl(error_log, error_log_path)
        
        return (query_id, False, error_msg)


def _process_full_session(
    session: Dict[str, Any],
    system_factory: Callable[[], MemorySystem],
    run_id: str,
    benchmark_name: str,
    model_name: str,
    output_path: Path,
    write_logs_path: Optional[str],  # 已废弃，保留以保持兼容
    qa_logs_path: Optional[str],  # 已废弃，保留以保持兼容
    error_log_path: Optional[str],  # 已废弃，保留以保持兼容
    processed_sessions: set,
    load_only: bool = False,  # 如果为 True，只加载已存在的 session，不存在则跳过
    write_max_workers: Optional[int] = None,  # Session 内部 Write (turns) 的并发数
    qa_max_workers: Optional[int] = None,  # Session 内部 QA 的并发数
) -> tuple[str, bool, Optional[str]]:
    """
    处理完整的 session（Write + QA 原子化执行）
    
    这是推荐的并发模式：每个 Session 作为一个原子任务，先执行 Write，然后立即执行 QA。
    这样可以：
    1. 快速反馈与止损（Fail Fast）
    2. 数据完整性与断点续跑（Resilience）
    3. 状态管理与资源释放
    4. 复用 system 实例，避免重复 load 的开销
    
    Args:
        processed_sessions: 已完成的 session 集合（以 QA 完成为准）
    
    Returns:
        (session_id, success, error_message)
    """
    session_id = session["session_id"]
    
    # 检查是否完全处理过（以 QA 完成为准）
    with _log_lock:
        if session_id in processed_sessions:
            return (session_id, True, "skipped")
    
    # 为每个 session 创建独立的日志文件路径
    session_write_logs_path = str(output_path / "sessions" / f"{session_id}_write.jsonl")
    session_qa_logs_path = str(output_path / "sessions" / f"{session_id}_qa.jsonl")
    session_error_log_path = str(output_path / "sessions" / f"{session_id}_error.jsonl")
    
    # 确保 sessions 目录存在
    os.makedirs(os.path.dirname(session_write_logs_path), exist_ok=True)
    
    # 检查是否存在 qa.jsonl 文件，如果存在且不为空，则跳过整个 session
    if os.path.exists(session_qa_logs_path) and os.path.getsize(session_qa_logs_path) > 0:
        return (session_id, True, "skipped (qa already exists)")
    
    try:
        # 为每个 session 创建独立的 system 实例
        system = system_factory()
        
        # 检查是否存在 write.jsonl 文件，决定使用 load() 还是 reset()
        write_exists = os.path.exists(session_write_logs_path) 
        
        if write_exists:
            # 如果 write.jsonl 存在，使用 load() 加载已有数据（断点续跑）
            system.load(session_id)
            logger.info(f"[{session_id}] Using load() for resume (write.jsonl exists)")
        else:
            # 如果 load_only = True 且 write.jsonl 不存在，跳过这个 session
            if load_only:
                warning_msg = f"[{session_id}] Skipping session: write.jsonl does not exist (load_only=True)"
                logger.warning(warning_msg)
                # 标记为已处理，避免重复尝试
                with _log_lock:
                    processed_sessions.add(session_id)
                return (session_id, True, "skipped (load_only=True, write.jsonl not found)")
            # 如果 write.jsonl 不存在，使用 reset() 从头开始
            system.reset(session_id)
            logger.info(f"[{session_id}] Using reset() for new session")
        
        # ====================
        # Phase 1: Write（并发处理turns）
        # ====================
        # 如果 write.jsonl 已存在，跳过 Write 阶段（断点续跑）
        if not write_exists:
            # 获取 turns
            turns_key = getattr(system, "preferred_turns_key", "turns")
            turns = session.get(turns_key, session.get("turns", []))
            
            # 处理所有 turns（并发处理，带锁保护）
            if turns:
                # 获取session级别的锁
                session_lock = _get_session_lock(session_id)
                
                # 准备turn数据（带索引）
                turn_data_list = [(turn_idx, turn_data) for turn_idx, turn_data in enumerate(turns)]
                
                # 使用ThreadPoolExecutor并发处理turns
                # 注意：PageStore 和 index 操作已经在内部有锁保护，可以实现真正的并发
                # 使用传入的write_max_workers参数，如果没有则默认4
                # 注意：由于 mem0.add(infer=True) 是瓶颈（LLM 调用），并发数不宜过高
                # 建议设置为 4-8，避免太多任务等待 index 锁
                default_write_workers = write_max_workers if write_max_workers is not None else 4
                actual_write_workers = min(len(turns), default_write_workers)  # 限制并发数，避免Qdrant过载
                
                failed_turns = []
                with ThreadPoolExecutor(max_workers=actual_write_workers) as executor:
                    # 提交所有任务
                    futures = {
                        executor.submit(
                            _process_single_turn,
                            turn_data,
                            session_id,
                            system,
                            run_id,
                            benchmark_name,
                            model_name,
                            session_write_logs_path,
                            session_error_log_path,
                            session_lock,
                        ): turn_data[0]  # turn_idx
                        for turn_data in turn_data_list
                    }
                    
                    # 使用tqdm显示进度
                    with tqdm(total=len(futures), desc=f"  [{session_id[:8]}] Write", unit="turn", 
                             leave=False, ncols=80, disable=False) as turn_pbar:
                        for future in as_completed(futures):
                            turn_idx = futures[future]
                            try:
                                result_turn_idx, success, error = future.result()
                                if not success:
                                    failed_turns.append(f"Turn {result_turn_idx}: {error}")
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"[{session_id}] Turn {turn_idx} processing exception: {e}")
                                failed_turns.append(f"Turn {turn_idx}: {error_msg}")
                            finally:
                                turn_pbar.update(1)
                
                # 如果有失败的turns，记录但继续（允许部分成功）
                if failed_turns:
                    error_msg = f"Some turns failed: {', '.join(failed_turns[:5])}"  # 只显示前5个
                    if len(failed_turns) > 5:
                        error_msg += f" ... and {len(failed_turns) - 5} more"
                    logger.warning(f"[{session_id}] Write phase completed with {len(failed_turns)} failed turns: {error_msg}")
                    # 记录警告日志
                    warning_log = {
                        "run_id": run_id,
                        "benchmark": benchmark_name,
                        "system": system.get_system_name(),
                        "session_id": session_id,
                        "error_type": "write_partial_failure",
                        "error_message": error_msg,
                        "failed_count": len(failed_turns),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    from core.runner.logging_utils import append_log_to_jsonl
                    append_log_to_jsonl(warning_log, session_error_log_path)
            
            # 尝试执行 auto_summary
            try:
                system.auto_summary(session_id)
            except Exception as e:
                # 警告但不失败
                logger.error(f"[{session_id}] Auto-summary failed: {e}")
                pass
        else:
            # Write 阶段已存在，跳过（断点续跑）
            logger.info(f"[{session_id}] Skipping Write phase (write.jsonl exists)")
        
        # ====================
        # Phase 2: QA（并发处理，复用同一个 system 实例，带锁保护）
        # ====================
        # 注意：复用 Write 阶段的 system 实例，避免重复 load 的开销
        # 虽然 LinkedViewSystem 不是线程安全的，但通过锁保护可以实现并发
        # 这样可以提高QA阶段的处理速度

        qa_pairs = session.get("qa_pairs", [])
        logger.info(f"[{session_id}] QA pairs: {qa_pairs}")
        if qa_pairs:
            # Session 内部的 QA 并发执行，复用同一个 system 实例，通过锁保护
            failed_queries = []
            
            # 使用ThreadPoolExecutor并发处理QA pairs
            # 注意：虽然并发，但通过session_lock保护同一session的system操作
            # 使用传入的qa_max_workers参数，如果没有则默认4
            default_qa_workers = qa_max_workers if qa_max_workers is not None else 4
            actual_qa_workers = min(len(qa_pairs), default_qa_workers)  # 限制并发数，避免Qdrant过载
            
            with ThreadPoolExecutor(max_workers=actual_qa_workers) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(
                        _process_single_qa_pair,
                        qa=qa,
                        session_id=session_id,
                        system=system,  # 复用 Write 阶段的 system 实例
                        run_id=run_id,
                        benchmark_name=benchmark_name,
                        model_name=model_name,
                        qa_logs_path=session_qa_logs_path,  # 使用 session 专属文件
                        error_log_path=session_error_log_path,  # 使用 session 专属文件
                    ): qa.get("query_id", "unknown")
                    for qa in qa_pairs
                }
                
                # 使用tqdm显示进度
                with tqdm(total=len(futures), desc=f"  [{session_id[:8]}] QA", unit="qa", 
                         leave=False, ncols=80, disable=False) as qa_pbar:
                    for future in as_completed(futures):
                        query_id = futures[future]
                        try:
                            result_query_id, success, error = future.result()
                            if not success:
                                failed_queries.append(f"{result_query_id}: {error}")
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"[{session_id}] QA {query_id} processing exception: {e}")
                            failed_queries.append(f"{query_id}: {error_msg}")
                        finally:
                            qa_pbar.update(1)
            
            # 如果有失败的 QA pairs，记录但不算整个 session 失败（允许部分成功）
            if failed_queries:
                error_msg = f"Some QA pairs failed: {', '.join(failed_queries)}"
                from datetime import datetime
                warning_log = {
                    "run_id": run_id,
                    "benchmark": benchmark_name,
                    "system": system.get_system_name(),
                    "session_id": session_id,
                    "error_type": "qa_partial_failure",
                    "error_message": error_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
                # 不需要锁，因为每个 session 有独立的文件
                from core.runner.logging_utils import append_log_to_jsonl
                append_log_to_jsonl(warning_log, session_error_log_path)

        # ====================
        # 保存 Linked Facts 记录（用于后续分析）
        # ====================
        # 如果 system 支持 linked facts 记录，在 session 结束时保存
        if hasattr(system, 'save_pending_linked_facts'):
            try:
                linked_facts_dir = str(output_path / "linked_facts")
                os.makedirs(linked_facts_dir, exist_ok=True)
                saved_file = system.save_pending_linked_facts(linked_facts_dir)
                if saved_file:
                    logger.info(f"[{session_id}] Saved linked facts records to {saved_file}")
            except Exception as e:
                logger.warning(f"[{session_id}] Failed to save linked facts records: {e}")

        # 标记为完成（以 QA 完成为准）
        with _log_lock:
            processed_sessions.add(session_id)
        
        return (session_id, True, None)
        
    except Exception as e:
        error_msg = str(e)
        # 记录错误日志（使用 session 专属文件，不需要锁）
        from datetime import datetime
        # 确保 sessions 目录存在
        session_error_log_path = str(output_path / "sessions" / f"{session_id}_error.jsonl")
        os.makedirs(os.path.dirname(session_error_log_path), exist_ok=True)
        
        error_log = {
            "run_id": run_id,
            "benchmark": benchmark_name,
            "system": system_factory().get_system_name() if hasattr(system_factory, '__call__') else "unknown",
            "session_id": session_id,
            "error_type": "session_failed",
            "error_message": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
        from core.runner.logging_utils import append_log_to_jsonl
        append_log_to_jsonl(error_log, session_error_log_path)
        
        # 即使失败也标记为已处理，避免重复尝试
        with _log_lock:
            processed_sessions.add(session_id)
        
        return (session_id, False, error_msg)




def run_benchmark_multi(
    system: MemorySystem,
    dataset_module,
    benchmark_name: str,
    run_id: str,
    config: Dict[str, Any],
    output_dir: str = "results",
    limit: Optional[int] = None,
    max_workers: int = 4,
    executor_type: str = "thread",  # "thread" or "process"
    system_config: Optional[Dict[str, Any]] = None,  # 可选：显式传入系统配置
    qa_max_workers: Optional[int] = None,  # Session 内部 QA 的并发数（默认4）
    write_max_workers: Optional[int] = None,  # Session 内部 Write (turns) 的并发数（默认4）
    load_only: bool = False,  # 如果为 True，只加载已存在的 session，不存在则跳过
) -> Dict[str, Any]:
    """
    并发版本的 benchmark 评估（Session 原子化模式）
    
    采用"每个 Session 作为一个原子任务（Write + QA）"的并发模式：
    - 每个 Session 先执行 Write，然后立即执行 QA
    - Session 内部的 turns 和 QA pairs 都支持并发处理（通过 session 级别的锁保护）
    - 复用同一个 system 实例（避免重复 load 的开销）
    - 针对 Qdrant 错误加强重试机制（最多5次，指数退避，Qdrant错误延迟加倍）
    - 这样可以快速反馈与止损、提高数据完整性与断点续跑能力
    
    注意：
    - Session 级别的并发由 max_workers 控制，建议设置为 5-8 个
    - Session 内部的 turns/QA 并发数由 write_max_workers/qa_max_workers 控制（默认4），避免 Qdrant 过载
    
    Args:
        system: MemorySystem 实例（将用于创建多个实例）
        dataset_module: 数据集模块
        benchmark_name: benchmark 名称
        run_id: 运行 ID
        config: 配置字典
        output_dir: 输出目录
        limit: 限制处理的 session 数量
        max_workers: Session 级别的最大并发数（建议 5-8 个）
        executor_type: 执行器类型，"thread" 或 "process"
        qa_max_workers: Session 内部 QA 的并发数（默认4，设为None则自动使用4）
        write_max_workers: Session 内部 Write (turns) 的并发数（默认4，设为None则自动使用4）
        load_only: 如果为 True，只加载已存在的 session（write.jsonl 存在），不存在则跳过并警告
        
    Returns:
        包含评估结果的字典
    """
    model_name = config.get("model_name", "unknown")
    split = config.get("split", "test")
    
    # 准备输出路径
    output_path = Path(output_dir) / benchmark_name / system.get_system_name() / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 sessions
    sessions = list(dataset_module.iter_sessions(split=split, limit=limit))
    print(f"Processing {len(sessions)} sessions with {max_workers} workers ({executor_type} executor)...")
    
    # 创建 system factory（用于为每个 session 创建独立的实例）
    # 注意：需要深拷贝配置，避免共享状态
    system_class = type(system)
    
    # 尝试获取原始配置
    # 优先使用显式传入的配置
    if system_config is not None:
        original_config = copy.deepcopy(system_config)
    elif hasattr(system, '_original_config'):
        original_config = copy.deepcopy(getattr(system, '_original_config'))
    elif hasattr(system, '_mem0_cfg'):
        # LinkedViewSystem 的情况：尝试从属性重建配置
        original_config = {
            'benchmark_name': benchmark_name,  # 传递 benchmark_name
            'mem0_config': copy.deepcopy(getattr(system, '_mem0_cfg', {})),
            'router_threshold': getattr(system, 'router', None).threshold if hasattr(system, 'router') and system.router else 0.5,
            'fast_model': config.get('model_name', 'gpt-4.1-mini'),
            'slow_model': config.get('model_name', 'gpt-4.1-mini'),
            'top_k': getattr(system, 'per_query_top_k', 5),
            'max_research_iters': getattr(system, 'max_research_iters', 3),
        }
    elif hasattr(system, '_config'):
        original_config = copy.deepcopy(getattr(system, '_config'))
    elif hasattr(system, 'config'):
        original_config = copy.deepcopy(getattr(system, 'config'))
    else:
        original_config = {}
    
    # 确保 benchmark_name 在配置中（用于生成 collection 名称）
    if 'benchmark_name' not in original_config:
        original_config['benchmark_name'] = benchmark_name
    
    def system_factory():
        """创建新的 system 实例"""
        # 深拷贝配置避免共享状态
        new_config = copy.deepcopy(original_config)
        
        # 根据不同的系统类型创建实例
        if system_class.__name__ == 'LinkedViewSystem':
            # LinkedViewSystem 直接传入配置字典
            return system_class(new_config)
        else:
            # 其他系统尝试使用 config 参数
            try:
                return system_class(config=new_config)
            except TypeError:
                # 如果失败，尝试直接传入配置
                try:
                    return system_class(new_config)
                except:
                    # 最后尝试无参数创建（不推荐，但作为后备）
                    return system_class()
    
    # 加载已处理的 sessions（断点续跑）
    # 注意：以 QA 完成为准（processed_sessions 包含已完成 Write + QA 的 session）
    # 现在使用每个 session 独立的日志文件，通过检查文件是否存在来判断是否已处理
    sessions_dir = output_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    from core.runner.logging_utils import load_processed_sessions
    # 从所有 session 的 QA 日志文件中加载已处理的 sessions
    processed_sessions = set()
    if sessions_dir.exists():
        for qa_file in sessions_dir.glob("*_qa.jsonl"):
            if qa_file.stat().st_size > 0:  # 文件存在且不为空
                # 从文件名提取 session_id
                session_id = qa_file.stem.replace("_qa", "")
                processed_sessions.add(session_id)
    
    # 选择执行器
    if executor_type == "process":
        executor_class = ProcessPoolExecutor
    else:
        executor_class = ThreadPoolExecutor
    
    # 并发执行 Session 原子化任务（Write + QA）
    print("\n" + "="*60)
    print("Phase: Session Atomic (Write + QA Concurrent)")
    print("="*60)
    print(f"每个 Session 作为一个原子任务：先执行 Write，然后立即执行 QA")
    print(f"Session 级别并发数: {max_workers}")
    write_workers = write_max_workers if write_max_workers is not None else 4
    qa_workers = qa_max_workers if qa_max_workers is not None else 4
    print(f"Session 内部 Turns 并发数: {write_workers}（带锁保护，避免 Qdrant 过载）")
    print(f"Session 内部 QA 并发数: {qa_workers}（带锁保护，避免 Qdrant 过载）")
    print(f"Qdrant 错误重试: 最多5次，指数退避，Qdrant错误延迟加倍")
    print("="*60)
    
    with executor_class(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                _process_full_session,
                session,
                system_factory,
                run_id,
                benchmark_name,
                model_name,
                output_path,
                None,  # write_logs_path 不再使用（每个 session 有自己的文件）
                None,  # qa_logs_path 不再使用（每个 session 有自己的文件）
                None,  # error_log_path 不再使用（每个 session 有自己的文件）
                processed_sessions,
                load_only,  # 传递 load_only 参数
                write_max_workers,  # 传递 write_max_workers 参数
                qa_max_workers,  # 传递 qa_max_workers 参数
            ): session["session_id"]
            for session in sessions
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(futures), desc="Sessions (Write+QA)", unit="session", ncols=100) as pbar:
            for future in as_completed(futures):
                session_id = futures[future]
                try:
                    result_session_id, success, error = future.result()
                    if success:
                        if error == "skipped":
                            pbar.set_postfix({"status": "skipped"})
                        else:
                            pbar.set_postfix({"status": "completed"})
                    else:
                        pbar.set_postfix({"status": "failed", "error": error[:30]})
                except Exception as e:
                    pbar.set_postfix({"status": "exception", "error": str(e)[:30]})
                finally:
                    pbar.update(1)
    
    # 3. 总结阶段（顺序执行）
    print("\n" + "="*60)
    print("Phase 3: Summary")
    print("="*60)
    summary = run_summary_phase(
        benchmark_name=benchmark_name,
        system_name=system.get_system_name(),
        model_name=model_name,
        run_id=run_id,
        config=config,
        output_path=output_path,
        num_sessions=len(sessions)
    )
    
    return summary


def main():
    """命令行入口（与原版兼容）"""
    parser = argparse.ArgumentParser(description="Run benchmark evaluation (concurrent version)")
    parser.add_argument("--system", type=str, required=True, help="System name")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of concurrent workers (session level, recommended: 5-8)")
    parser.add_argument("--qa-max-workers", type=int, default=4, help="Maximum number of concurrent QA pairs per session (default: min(4, num_qa_pairs))")
    parser.add_argument("--executor", type=str, default="thread", choices=["thread", "process"],
                       help="Executor type: thread or process")
    
    args = parser.parse_args()
    
    # 生成 run_id
    if args.run_id is None:
        args.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 导入数据集和系统（与原版相同）
    if args.benchmark == "locomo":
        from core.datasets import locomo as dataset_module
    elif args.benchmark == "hotpotqa":
        from core.datasets import hotpotqa as dataset_module
    elif args.benchmark == "memory_agent_bench":
        from core.datasets import memory_agent_bench as dataset_module
    elif args.benchmark == "longmemeval":
        from core.datasets import longmemeval as dataset_module
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # 导入系统（与原版相同）
    if args.system == "linked_view":
        from core.systems.linked_view_adapter import LinkedViewSystem
        lv_cfg = {
            "mem0_config": {
                "backend": "mem0",
                "llm": {
                    "provider": "openai",
                    "config": {"model": args.model_name},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": "localhost",
                        "port": 6333,
                        "collection_name": "mem0_linked",
                    },
                },
            },
            "router_threshold": 0.5,
            "fast_model": args.model_name,
            "slow_model": args.model_name,
            "top_k": 5,
            "max_research_iters": 3,
        }
        system = LinkedViewSystem(lv_cfg)
    else:
        # 其他系统...（与原版相同）
        raise ValueError(f"System {args.system} not yet supported in concurrent version")
    
    # 运行并发评估
    config = {
        "model_name": args.model_name,
        "split": args.split
    }
    
    summary = run_benchmark_multi(
        system=system,
        dataset_module=dataset_module,
        benchmark_name=args.benchmark,
        run_id=args.run_id,
        config=config,
        output_dir=args.output_dir,
        limit=args.limit,
        max_workers=args.max_workers,
        executor_type=args.executor,
        qa_max_workers=args.qa_max_workers,
    )
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

