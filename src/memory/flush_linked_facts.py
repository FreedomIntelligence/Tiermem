#!/usr/bin/env python3
"""
并发将 linked facts 记录写入数据库（支持严格质量筛选和智能回填）。

用法:
    python -m src.memory.flush_linked_facts LINKED_FACTS_DIR [选项]

示例:
    # 使用严格 LLM 质量筛选（默认开启，推荐）
    # 所有 facts 都会经过质量检查，只有高质量数据才会写入
    python -m src.memory.flush_linked_facts results/locomo/linked_view/test_run/linked_facts

    # 使用智能回填模式（召回比对 + LLM 决策）
    # 写入前召回已有 memory，LLM 决定 SKIP/UPDATE/DELETE_AND_ADD/ADD
    python -m src.memory.flush_linked_facts results/.../linked_facts --smart

    # 指定并发数
    python -m src.memory.flush_linked_facts results/locomo/linked_view/test_router_vllm_grpo_950_2/linked_facts --max-workers 10 --smart

    # 禁用 LLM 筛选（只做数量截断，不推荐）
    python -m src.memory.flush_linked_facts results/.../linked_facts --no-filter-llm

    # 只处理特定 session
    python -m src.memory.flush_linked_facts results/.../linked_facts --sessions conv-26,conv-30

    # 干跑模式（只统计，不写入）
    python -m src.memory.flush_linked_facts results/.../linked_facts --dry-run

    # 智能模式 dry-run（只统计 LLM 决策，不写入）
    python -m src.memory.flush_linked_facts results/.../linked_facts --smart --dry-run

质量筛选标准:
    1. 直接相关：fact 必须直接帮助回答特定问题
    2. 具体明确：包含具体细节（人名、日期、数字、地点）
    3. 事实根据：基于具体对话内容，非推断
    4. 非冗余：提供问题本身未包含的独特信息
    5. 自包含：不需要额外上下文即可理解

智能回填模式:
    写入前召回已有 memory，LLM 决定操作类型：
    - SKIP: 新 fact 与已有 memory 语义重复
    - UPDATE: 新 fact 应更新某个已有 memory
    - DELETE_AND_ADD: 已有 memory 过时/错误，需替换
    - ADD: 新 fact 是全新信息
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from tqdm import tqdm

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# 重试相关常量
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # 秒


def _is_retryable_error(error: Exception) -> bool:
    """判断是否是可重试的错误（Qdrant/网络相关）"""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    retryable_keywords = [
        "qdrant",
        "connection",
        "timeout",
        "refused",
        "reset",
        "network",
        "unavailable",
        "rate limit",
        "too many requests",
        "503",
        "502",
        "500",
        "429",
    ]

    return any(keyword in error_str for keyword in retryable_keywords) or "connection" in error_type


def _retry_with_backoff(func, *args, max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY, **kwargs):
    """
    带指数退避的重试包装器。

    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）

    Returns:
        函数执行结果

    Raises:
        最后一次重试失败时抛出的异常
    """
    last_exception = None

    for retry_count in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not _is_retryable_error(e):
                # 不可重试的错误，直接抛出
                raise

            if retry_count < max_retries - 1:
                wait_time = initial_delay * (2 ** retry_count)
                logger.warning(
                    f"操作失败 (attempt {retry_count + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"操作失败，已重试 {max_retries} 次: {e}")

    raise last_exception


# LLM 筛选相关
FILTER_PROMPT_TEMPLATE = """You are a strict quality judge for memory facts. Given a question and candidate facts, select ONLY high-quality facts that meet ALL criteria.

Question: {question}

Candidate Facts:
{facts_list}

Quality Criteria (ALL must be met):
1. **Directly Relevant**: The fact must directly help answer this specific question or very similar questions
2. **Specific & Concrete**: Contains specific details (names, dates, numbers, locations) - NOT vague or general statements
3. **Factually Grounded**: Based on concrete conversation content, not inferences or assumptions
4. **Non-redundant**: Provides unique information not already captured in the question itself
5. **Self-contained**: Makes sense on its own without needing additional context

Reject facts that are:
- Too general (e.g., "User likes reading" without specifics)
- Obvious from the question (e.g., question asks about X, fact says "X is important")
- Vague or ambiguous
- Tangentially related but not directly useful

IMPORTANT: Be STRICT. It's better to return [] than to include low-quality facts.
Select at most {max_facts} facts, but return fewer (or []) if quality criteria are not met.

Output format: Return ONLY a JSON array of selected fact indices (0-based), e.g., [0, 2], [1], or [] if none qualify.
Only output the JSON array, nothing else."""


# 单条 fact 质量评估 prompt
QUALITY_CHECK_PROMPT_TEMPLATE = """You are a strict quality judge for memory facts. Evaluate if this fact is worth storing in a memory database.

Question that triggered this fact: {question}

Fact to evaluate: {fact}

Quality Criteria (ALL must be met):
1. **Directly Relevant**: Helps answer this specific question or very similar questions
2. **Specific & Concrete**: Contains specific details (names, dates, numbers, locations)
3. **Factually Grounded**: Based on concrete information, not vague inferences
4. **Non-redundant**: Provides unique information beyond what's in the question
5. **Self-contained**: Makes sense on its own

Answer with ONLY "YES" or "NO".
- YES: Fact meets ALL criteria and is worth storing
- NO: Fact fails any criterion and should be discarded"""


def _check_single_fact_quality(
    question: str,
    fact: Dict[str, Any],
    model: str = "gpt-4.1-mini",
) -> bool:
    """
    检查单条 fact 的质量是否达标。

    Args:
        question: 原始问题
        fact: 要检查的 fact
        model: LLM 模型名称

    Returns:
        True 如果质量达标，False 否则
    """
    import os
    from openai import OpenAI

    fact_text = fact.get("fact", "")
    if not fact_text.strip():
        return False

    prompt = QUALITY_CHECK_PROMPT_TEMPLATE.format(
        question=question,
        fact=fact_text,
    )

    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )

        result_text = response.choices[0].message.content.strip().upper()
        return result_text.startswith("YES")

    except Exception as e:
        logger.warning(f"单条 fact 质量检查失败: {e}，默认拒绝")
        return False


def _filter_facts_with_llm(
    question: str,
    facts: List[Dict[str, Any]],
    model: str = "gpt-4.1-mini",
    max_facts: int = 2,
    always_filter: bool = True,
) -> List[Dict[str, Any]]:
    """
    使用 LLM 筛选高质量的 facts。

    Args:
        question: 原始问题
        facts: 候选 facts 列表
        model: LLM 模型名称
        max_facts: 最多选择的 facts 数量
        always_filter: 即使只有 1 条也要检查质量

    Returns:
        筛选后的 facts 列表（可能为空）
    """
    if not facts:
        return []

    import os
    from openai import OpenAI

    # 如果只有一条 fact，使用单条质量检查
    if len(facts) == 1:
        if always_filter:
            if _check_single_fact_quality(question, facts[0], model):
                logger.info(f"单条 fact 质量检查通过")
                return facts
            else:
                logger.info(f"单条 fact 质量检查未通过，已过滤")
                return []
        else:
            return facts

    # 多条 facts 使用批量筛选
    # 构建 facts 列表文本
    facts_list_text = ""
    for i, fact in enumerate(facts):
        fact_text = fact.get("fact", "")
        facts_list_text += f"[{i}] {fact_text}\n"

    prompt = FILTER_PROMPT_TEMPLATE.format(
        question=question,
        facts_list=facts_list_text.strip(),
        max_facts=max_facts,
    )

    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()

        # 解析 JSON 数组
        import re
        # 提取 JSON 数组
        match = re.search(r'\[[\d,\s]*\]', result_text)
        if match:
            selected_indices = json.loads(match.group())
        else:
            # 尝试直接解析
            selected_indices = json.loads(result_text)

        # 验证并返回选中的 facts
        filtered_facts = []
        for idx in selected_indices:
            if isinstance(idx, int) and 0 <= idx < len(facts):
                filtered_facts.append(facts[idx])

        # 允许返回空数组（LLM 认为没有高质量的 facts）
        logger.info(f"LLM 筛选: {len(facts)} -> {len(filtered_facts)} facts")
        return filtered_facts

    except Exception as e:
        logger.warning(f"LLM 筛选失败: {e}，默认全部过滤（保守策略）")
        # 保守策略：筛选失败时不写入，避免写入低质量数据
        return []


def load_linked_facts_file(file_path: Path) -> tuple[str, List[Dict[str, Any]]]:
    """
    加载单个 linked facts JSON 文件。

    Args:
        file_path: JSON 文件路径

    Returns:
        (session_id, records) 元组
    """
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 从文件名提取 session_id
    # 文件名格式: {session_id}_linked_facts.json
    session_id = file_path.stem.replace("_linked_facts", "")

    return session_id, records


def flush_session_to_database(
    session_id: str,
    records: List[Dict[str, Any]],
    system_config: Dict[str, Any],
    linked_facts_dir: str,
    dry_run: bool = False,
    max_retries: int = MAX_RETRIES,
    filter_with_llm: bool = False,
    max_facts_per_query: int = 2,
    filter_model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    将单个 session 的 linked facts 写入数据库（带重试机制和 LLM 筛选）。

    Args:
        session_id: 会话 ID
        records: linked facts 记录列表
        system_config: LinkedViewSystem 配置
        dry_run: 如果为 True，只统计不写入
        max_retries: 最大重试次数
        filter_with_llm: 是否使用 LLM 筛选 facts
        max_facts_per_query: 每个 query 最多写入的 facts 数量
        filter_model: 筛选用的 LLM 模型

    Returns:
        统计信息字典
    """
    import copy
    from src.memory.linked_view_system import LinkedViewSystem

    stats = {
        "session_id": session_id,
        "total_records": len(records),
        "total_facts": 0,
        "filtered_facts": 0,
        "rejected_facts": 0,  # 新增：质量不达标被完全拒绝的 facts
        "written_facts": 0,
        "skipped_facts": 0,
        "retry_count": 0,
        "errors": [],
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    # 统计 facts 数量
    for record in records:
        facts = record.get("linked_facts", [])
        stats["total_facts"] += len(facts)

    if dry_run:
        # 在 dry_run 模式下也进行筛选并保存 cleaned 文件
        cleaned_records = []
        for record in records:
            query_idx = record.get("query_idx", 0)
            question = record.get("question", "")
            facts = record.get("linked_facts", [])
            
            cleaned_record = record.copy()
            
            if filter_with_llm and facts:
                try:
                    filtered = _filter_facts_with_llm(
                        question=question,
                        facts=facts,
                        model=filter_model,
                        max_facts=max_facts_per_query,
                        always_filter=True,
                    )
                    cleaned_record["linked_facts"] = filtered
                except Exception as e:
                    logger.warning(f"[{session_id}] Dry run LLM 筛选失败: {e}")
                    cleaned_record["linked_facts"] = []
            elif len(facts) > max_facts_per_query:
                cleaned_record["linked_facts"] = facts[:max_facts_per_query]
            else:
                cleaned_record["linked_facts"] = facts
            
            cleaned_records.append(cleaned_record)
        
        # 按 query_idx 排序并保存
        cleaned_records.sort(key=lambda r: r.get("query_idx", 0))
        try:
            linked_facts_path = Path(linked_facts_dir)
            cleaned_file_path = linked_facts_path / f"{session_id}_linked_facts_cleaned.json"
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
            logger.info(f"[{session_id}] Dry run: 筛选后的记录已保存到: {cleaned_file_path}")
        except Exception as e:
            logger.warning(f"[{session_id}] Dry run: 保存 cleaned 文件失败: {e}")
        
        stats["status"] = "dry_run"
        stats["end_time"] = datetime.now(timezone.utc).isoformat()
        return stats

    system = None

    try:
        # 为每个 session 使用独立的 history_db_path，避免 SQLite 锁冲突
        session_config = copy.deepcopy(system_config)

        # 创建独立的 history.db 路径
        import tempfile
        history_db_dir = Path(tempfile.gettempdir()) / "mem0_history" / session_id
        history_db_dir.mkdir(parents=True, exist_ok=True)
        history_db_path = str(history_db_dir / "history.db")

        # 设置到 mem0 配置中
        if "mem0_config" in session_config:
            session_config["mem0_config"]["history_db_path"] = history_db_path

        # 创建 system 实例并加载 session（带重试）
        def _create_and_load_system():
            s = LinkedViewSystem(session_config)
            s.load(session_id)
            return s

        system = _retry_with_backoff(_create_and_load_system, max_retries=max_retries)

        # 写入每条记录的 linked facts（并发处理）
        import threading
        stats_lock = threading.Lock()

        def process_single_record(record):
            """处理单条记录"""
            local_stats = {
                "filtered_facts": 0,
                "written_facts": 0,
                "skipped_facts": 0,
                "rejected_facts": 0,  # 新增：质量不达标被拒绝的 facts
                "errors": [],
            }

            query_idx = record.get("query_idx", 0)
            question = record.get("question", "")
            facts = record.get("linked_facts", [])
            original_count = len(facts)

            # 创建筛选后的记录副本
            cleaned_record = record.copy()

            if not facts:
                cleaned_record["linked_facts"] = []
                return local_stats, cleaned_record

            # LLM 质量筛选（对所有 facts 都进行质量检查）
            if filter_with_llm:
                try:
                    filtered = _filter_facts_with_llm(
                        question=question,
                        facts=facts,
                        model=filter_model,
                        max_facts=max_facts_per_query,
                        always_filter=True,  # 即使只有 1 条也要检查质量
                    )
                    local_stats["filtered_facts"] += original_count - len(filtered)
                    if len(filtered) == 0:
                        local_stats["rejected_facts"] = original_count
                        logger.info(f"[{session_id}] Query {query_idx}: 所有 {original_count} 条 facts 都未通过质量检查")
                        cleaned_record["linked_facts"] = []
                        return local_stats, cleaned_record
                    facts = filtered
                except Exception as e:
                    # 保守策略：筛选失败时不写入
                    logger.warning(f"[{session_id}] LLM 筛选失败，跳过此记录: {e}")
                    local_stats["filtered_facts"] += original_count
                    local_stats["rejected_facts"] = original_count
                    cleaned_record["linked_facts"] = []
                    return local_stats, cleaned_record
            elif len(facts) > max_facts_per_query:
                # 不用 LLM 筛选时，直接截断
                local_stats["filtered_facts"] += len(facts) - max_facts_per_query
                facts = facts[:max_facts_per_query]

            # 更新筛选后的记录
            cleaned_record["linked_facts"] = facts

            # 如果没有剩余 facts，跳过写入
            if not facts:
                return local_stats, cleaned_record

            # 带重试的写入
            facts_to_write = facts

            def _write_facts():
                return system._do_write_facts_to_database(
                    linked_facts=facts_to_write,
                    session_id=session_id,
                    query_idx=query_idx,
                )

            try:
                written = _retry_with_backoff(_write_facts, max_retries=max_retries)
                local_stats["written_facts"] += written
                local_stats["skipped_facts"] += len(facts_to_write) - written
            except Exception as e:
                error_msg = f"Query {query_idx}: {str(e)}"
                local_stats["errors"].append(error_msg)
                logger.error(f"[{session_id}] Failed to write facts for query {query_idx} after retries: {e}")

            return local_stats, cleaned_record

        # 并发处理所有 records
        record_workers = min(len(records), 4)  # 每个 session 内最多 4 个并发
        cleaned_records = []  # 收集筛选后的记录
        
        with ThreadPoolExecutor(max_workers=record_workers) as record_executor:
            futures = {
                record_executor.submit(process_single_record, record): record
                for record in records
            }

            for future in as_completed(futures):
                try:
                    local_stats, cleaned_record = future.result()
                    with stats_lock:
                        stats["filtered_facts"] += local_stats["filtered_facts"]
                        stats["rejected_facts"] += local_stats.get("rejected_facts", 0)
                        stats["written_facts"] += local_stats["written_facts"]
                        stats["skipped_facts"] += local_stats["skipped_facts"]
                        stats["errors"].extend(local_stats["errors"])
                        # 收集筛选后的记录
                        cleaned_records.append(cleaned_record)
                except Exception as e:
                    with stats_lock:
                        stats["errors"].append(str(e))
                    logger.error(f"[{session_id}] Record processing exception: {e}")

        # 按 query_idx 排序并保存筛选后的记录
        cleaned_records.sort(key=lambda r: r.get("query_idx", 0))
        
        # 保存到 cleaned 文件
        try:
            linked_facts_path = Path(linked_facts_dir)
            cleaned_file_path = linked_facts_path / f"{session_id}_linked_facts_cleaned.json"
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
            logger.info(f"[{session_id}] 筛选后的记录已保存到: {cleaned_file_path}")
        except Exception as e:
            logger.warning(f"[{session_id}] 保存 cleaned 文件失败: {e}")

        stats["status"] = "success" if not stats["errors"] else "partial_success"

    except Exception as e:
        stats["status"] = "failed"
        stats["errors"].append(str(e))
        logger.error(f"[{session_id}] Session processing failed: {e}")

    stats["end_time"] = datetime.now(timezone.utc).isoformat()
    return stats


def smart_flush_session_to_database(
    session_id: str,
    records: List[Dict[str, Any]],
    system_config: Dict[str, Any],
    linked_facts_dir: str,
    dry_run: bool = False,
    max_retries: int = MAX_RETRIES,
    smart_top_k: int = 5,
    decision_model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    使用智能回填模式将单个 session 的 linked facts 写入数据库。

    智能回填流程：
    1. 对每条 fact，召回已有 memory (top_k)
    2. LLM 决策操作类型：SKIP/UPDATE/DELETE_AND_ADD/ADD
    3. 执行相应操作

    Args:
        session_id: 会话 ID
        records: linked facts 记录列表
        system_config: LinkedViewSystem 配置
        linked_facts_dir: linked facts 目录路径
        dry_run: 如果为 True，只统计不写入
        max_retries: 最大重试次数
        smart_top_k: 召回的 memory 数量
        decision_model: LLM 决策模型

    Returns:
        统计信息字典，包含各类操作的计数
    """
    import copy
    from src.memory.linked_view_system import LinkedViewSystem

    stats = {
        "session_id": session_id,
        "total_records": len(records),
        "total_facts": 0,
        # 智能回填统计
        "add_facts": 0,
        "skip_facts": 0,
        "update_facts": 0,
        "failed_facts": 0,
        "errors": [],
        "start_time": datetime.now(timezone.utc).isoformat(),
        "decisions": [],  # 详细决策记录（用于分析）
    }

    # 统计 facts 数量
    for record in records:
        facts = record.get("linked_facts", [])
        stats["total_facts"] += len(facts)

    if stats["total_facts"] == 0:
        stats["status"] = "empty"
        stats["end_time"] = datetime.now(timezone.utc).isoformat()
        return stats

    system = None

    try:
        # 为每个 session 使用独立的 history_db_path，避免 SQLite 锁冲突
        session_config = copy.deepcopy(system_config)

        # 创建独立的 history.db 路径
        import tempfile
        history_db_dir = Path(tempfile.gettempdir()) / "mem0_history" / session_id
        history_db_dir.mkdir(parents=True, exist_ok=True)
        history_db_path = str(history_db_dir / "history.db")

        # 设置到 mem0 配置中
        if "mem0_config" in session_config:
            session_config["mem0_config"]["history_db_path"] = history_db_path

        # 更新 memory_system_model 以使用指定的决策模型
        session_config["memory_system_model"] = decision_model

        # 创建 system 实例并加载 session（带重试）
        def _create_and_load_system():
            s = LinkedViewSystem(session_config)
            s.load(session_id)
            return s

        system = _retry_with_backoff(_create_and_load_system, max_retries=max_retries)

        # 处理每条记录的 linked facts（批量模式：每个 question 一次召回 + 一次 LLM）
        cleaned_records = []

        for record in records:
            query_idx = record.get("query_idx", 0)
            question = record.get("question", "")
            facts = record.get("linked_facts", [])

            if not facts:
                continue

            cleaned_facts = []

            if dry_run:
                # Dry run 模式：批量决策，不实际写入
                try:
                    # 用 question 召回一次
                    user_id = f"{session_id}:user"
                    with system._index_lock:
                        hits = system.index.search(user_id=user_id, query=question, top_k=smart_top_k)

                    existing_memories = [
                        {"id": h.memory_id, "memory": h.summary_text, "score": h.score}
                        for h in hits if h.memory_id
                    ]

                    # 批量 LLM 决策（一次调用）
                    decisions = system._batch_decide_memory_actions(facts, question, existing_memories)

                    # 处理每条决策
                    for i, fact in enumerate(facts):
                        fact_text = fact.get("fact", "").strip()
                        if not fact_text:
                            continue

                        decision = next((d for d in decisions if d.get("fact_index") == i),
                                        {"action": "ADD", "reason": "No decision"})
                        action = decision.get("action", "ADD")

                        stats["decisions"].append({
                            "query_idx": query_idx,
                            "fact_index": i,
                            "fact_text": fact_text[:100],
                            "action": action,
                            "reason": decision.get("reason", ""),
                            "target_memory_id": decision.get("target_memory_id"),
                            "existing_memories_count": len(existing_memories),
                        })

                        if action == "SKIP":
                            stats["skip_facts"] += 1
                        elif action == "UPDATE":
                            stats["update_facts"] += 1
                            updated_fact = fact.copy()
                            updated_fact["smart_action"] = "UPDATE"
                            # 保存完整的 UPDATE 信息
                            updated_fact["updated_text"] = decision.get("updated_text") or fact_text
                            updated_fact["target_memory_id"] = decision.get("target_memory_id")
                            updated_fact["reason"] = decision.get("reason", "")
                            cleaned_facts.append(updated_fact)
                        else:  # ADD
                            stats["add_facts"] += 1
                            cleaned_fact = fact.copy()
                            cleaned_fact["smart_action"] = "ADD"
                            cleaned_fact["reason"] = decision.get("reason", "")
                            cleaned_facts.append(cleaned_fact)

                except Exception as e:
                    stats["failed_facts"] += len(facts)
                    stats["errors"].append(f"Query {query_idx}: batch decision failed: {str(e)}")
                    logger.warning(f"[{session_id}] Batch decision failed for query {query_idx}: {e}")

            else:
                # 实际写入模式：使用批量方法
                def _batch_write():
                    return system._smart_write_facts_batch(
                        facts=facts,
                        question=question,
                        session_id=session_id,
                        query_idx=query_idx,
                        top_k=smart_top_k,
                    )

                try:
                    results = _retry_with_backoff(_batch_write, max_retries=max_retries)

                    for i, result in enumerate(results):
                        fact = facts[i] if i < len(facts) else {}
                        fact_text = result.get("fact_text", "")
                        action = result.get("action", "ADD")

                        stats["decisions"].append({
                            "query_idx": query_idx,
                            "fact_index": i,
                            "fact_text": fact_text[:100] if fact_text else "",
                            "action": action,
                            "reason": result.get("reason", ""),
                            "success": result.get("success", False),
                            "target_memory_id": result.get("target_memory_id"),
                        })

                        if result.get("success"):
                            if action == "SKIP":
                                stats["skip_facts"] += 1
                            elif action == "UPDATE":
                                stats["update_facts"] += 1
                                updated_fact = fact.copy()
                                updated_fact["smart_action"] = "UPDATE"
                                # 保存完整的 UPDATE 信息，便于重跑
                                updated_fact["updated_text"] = result.get("updated_text") or fact.get("fact", "")
                                updated_fact["target_memory_id"] = result.get("target_memory_id")
                                updated_fact["reason"] = result.get("reason", "")
                                cleaned_facts.append(updated_fact)
                            else:  # ADD
                                stats["add_facts"] += 1
                                cleaned_fact = fact.copy()
                                cleaned_fact["smart_action"] = "ADD"
                                cleaned_fact["reason"] = result.get("reason", "")
                                cleaned_facts.append(cleaned_fact)
                        else:
                            stats["failed_facts"] += 1
                            if result.get("error"):
                                stats["errors"].append(f"Query {query_idx}, fact {i}: {result['error']}")

                except Exception as e:
                    stats["failed_facts"] += len(facts)
                    stats["errors"].append(f"Query {query_idx}: batch write failed: {str(e)}")
                    logger.error(f"[{session_id}] Batch write failed for query {query_idx}: {e}")

            if cleaned_facts:
                cleaned_records.append({
                    "query_idx": query_idx,
                    "question": question,
                    "linked_facts": cleaned_facts,
                })

        # 按 query_idx 排序并保存筛选后的记录
        cleaned_records.sort(key=lambda r: r.get("query_idx", 0))

        # 保存到 cleaned 文件
        try:
            linked_facts_path = Path(linked_facts_dir)
            cleaned_file_path = linked_facts_path / f"{session_id}_linked_facts_cleaned.json"
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
            logger.info(f"[{session_id}] 智能筛选后的记录已保存到: {cleaned_file_path}")
        except Exception as e:
            logger.warning(f"[{session_id}] 保存 cleaned 文件失败: {e}")

        # 保存决策记录到文件
        try:
            linked_facts_path = Path(linked_facts_dir)
            decisions_file_path = linked_facts_path / f"{session_id}_smart_decisions.json"
            with open(decisions_file_path, "w", encoding="utf-8") as f:
                json.dump(stats["decisions"], f, ensure_ascii=False, indent=2)
            logger.info(f"[{session_id}] 智能决策记录已保存到: {decisions_file_path}")
        except Exception as e:
            logger.warning(f"[{session_id}] 保存决策记录失败: {e}")

        stats["status"] = "success" if not stats["errors"] else "partial_success"

    except Exception as e:
        stats["status"] = "failed"
        stats["errors"].append(str(e))
        logger.error(f"[{session_id}] Smart session processing failed: {e}")

    stats["end_time"] = datetime.now(timezone.utc).isoformat()
    return stats


def flush_linked_facts_smart(
    linked_facts_dir: str,
    system_config: Dict[str, Any],
    max_workers: int = 4,
    sessions: Optional[List[str]] = None,
    dry_run: bool = False,
    smart_top_k: int = 5,
    decision_model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    使用智能回填模式并发将 linked facts 写入数据库。

    Args:
        linked_facts_dir: linked facts 目录路径
        system_config: LinkedViewSystem 配置
        max_workers: 最大并发数
        sessions: 只处理指定的 session（None 表示全部）
        dry_run: 如果为 True，只统计不写入
        smart_top_k: 召回的 memory 数量
        decision_model: LLM 决策模型

    Returns:
        汇总统计信息
    """
    linked_facts_path = Path(linked_facts_dir)

    if not linked_facts_path.exists():
        raise ValueError(f"目录不存在: {linked_facts_dir}")

    # 查找所有 linked facts 文件
    json_files = list(linked_facts_path.glob("*_linked_facts.json"))

    if not json_files:
        logger.warning(f"在 {linked_facts_dir} 中没有找到 linked facts 文件")
        return {"total_sessions": 0, "sessions": []}

    # 过滤指定的 sessions
    if sessions:
        sessions_set = set(sessions)
        json_files = [
            f for f in json_files
            if f.stem.replace("_linked_facts", "") in sessions_set
        ]

    logger.info(f"找到 {len(json_files)} 个 linked facts 文件")

    # 加载所有文件
    session_data: List[tuple[str, List[Dict[str, Any]]]] = []
    for json_file in json_files:
        try:
            session_id, records = load_linked_facts_file(json_file)
            if records:  # 只处理非空文件
                session_data.append((session_id, records))
                logger.info(f"  {session_id}: {len(records)} 条记录")
        except Exception as e:
            logger.error(f"加载文件失败 {json_file}: {e}")

    if not session_data:
        logger.warning("没有有效的 linked facts 记录")
        return {"total_sessions": 0, "sessions": []}

    # 并发处理
    all_stats: List[Dict[str, Any]] = []
    mode_str = "[DRY RUN] " if dry_run else ""

    print(f"\n{mode_str}开始智能回填 {len(session_data)} 个 session 的 linked facts...")
    print(f"召回数量: top_k={smart_top_k}, 决策模型: {decision_model}")
    print(f"并发数: {max_workers}")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                smart_flush_session_to_database,
                session_id,
                records,
                system_config,
                linked_facts_dir,
                dry_run,
                MAX_RETRIES,
                smart_top_k,
                decision_model,
            ): session_id
            for session_id, records in session_data
        }

        with tqdm(total=len(futures), desc="Sessions", unit="session", ncols=100) as pbar:
            for future in as_completed(futures):
                session_id = futures[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)

                    status = stats.get("status", "unknown")
                    add_count = stats.get("add_facts", 0)
                    skip_count = stats.get("skip_facts", 0)
                    update_count = stats.get("update_facts", 0)

                    if status in ("success", "partial_success"):
                        pbar.set_postfix({
                            "last": f"{session_id}: ADD={add_count}, SKIP={skip_count}, UPD={update_count}"
                        })
                    else:
                        pbar.set_postfix({"last": f"{session_id}: {status}"})

                except Exception as e:
                    logger.error(f"[{session_id}] 处理异常: {e}")
                    all_stats.append({
                        "session_id": session_id,
                        "status": "exception",
                        "errors": [str(e)],
                    })

                pbar.update(1)

    # 汇总统计
    summary = {
        "mode": "smart",
        "total_sessions": len(all_stats),
        "success_sessions": sum(1 for s in all_stats if s.get("status") == "success"),
        "partial_success_sessions": sum(1 for s in all_stats if s.get("status") == "partial_success"),
        "failed_sessions": sum(1 for s in all_stats if s.get("status") in ("failed", "exception")),
        "total_facts": sum(s.get("total_facts", 0) for s in all_stats),
        # 智能回填统计
        "add_facts": sum(s.get("add_facts", 0) for s in all_stats),
        "skip_facts": sum(s.get("skip_facts", 0) for s in all_stats),
        "update_facts": sum(s.get("update_facts", 0) for s in all_stats),
        "failed_facts": sum(s.get("failed_facts", 0) for s in all_stats),
        "sessions": all_stats,
    }

    return summary


def flush_linked_facts_concurrent(
    linked_facts_dir: str,
    system_config: Dict[str, Any],
    max_workers: int = 4,
    sessions: Optional[List[str]] = None,
    dry_run: bool = False,
    filter_with_llm: bool = False,
    max_facts_per_query: int = 2,
    filter_model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    并发将 linked facts 写入数据库。

    Args:
        linked_facts_dir: linked facts 目录路径
        system_config: LinkedViewSystem 配置
        max_workers: 最大并发数
        sessions: 只处理指定的 session（None 表示全部）
        dry_run: 如果为 True，只统计不写入
        filter_with_llm: 是否使用 LLM 筛选 facts
        max_facts_per_query: 每个 query 最多写入的 facts 数量
        filter_model: 筛选用的 LLM 模型

    Returns:
        汇总统计信息
    """
    linked_facts_path = Path(linked_facts_dir)

    if not linked_facts_path.exists():
        raise ValueError(f"目录不存在: {linked_facts_dir}")

    # 查找所有 linked facts 文件
    json_files = list(linked_facts_path.glob("*_linked_facts.json"))

    if not json_files:
        logger.warning(f"在 {linked_facts_dir} 中没有找到 linked facts 文件")
        return {"total_sessions": 0, "sessions": []}

    # 过滤指定的 sessions
    if sessions:
        sessions_set = set(sessions)
        json_files = [
            f for f in json_files
            if f.stem.replace("_linked_facts", "") in sessions_set
        ]

    logger.info(f"找到 {len(json_files)} 个 linked facts 文件")

    # 加载所有文件
    session_data: List[tuple[str, List[Dict[str, Any]]]] = []
    for json_file in json_files:
        try:
            session_id, records = load_linked_facts_file(json_file)
            if records:  # 只处理非空文件
                session_data.append((session_id, records))
                logger.info(f"  {session_id}: {len(records)} 条记录")
        except Exception as e:
            logger.error(f"加载文件失败 {json_file}: {e}")

    if not session_data:
        logger.warning("没有有效的 linked facts 记录")
        return {"total_sessions": 0, "sessions": []}

    # 并发处理
    all_stats: List[Dict[str, Any]] = []
    mode_str = "[DRY RUN] " if dry_run else ""
    filter_str = f" (LLM筛选, max={max_facts_per_query})" if filter_with_llm else f" (截断, max={max_facts_per_query})"

    print(f"\n{mode_str}开始写入 {len(session_data)} 个 session 的 linked facts{filter_str}...")
    print(f"并发数: {max_workers}")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                flush_session_to_database,
                session_id,
                records,
                system_config,
                linked_facts_dir,
                dry_run,
                MAX_RETRIES,
                filter_with_llm,
                max_facts_per_query,
                filter_model,
            ): session_id
            for session_id, records in session_data
        }

        with tqdm(total=len(futures), desc="Sessions", unit="session", ncols=100) as pbar:
            for future in as_completed(futures):
                session_id = futures[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)

                    status = stats.get("status", "unknown")
                    written = stats.get("written_facts", 0)
                    filtered = stats.get("filtered_facts", 0)
                    total = stats.get("total_facts", 0)

                    if status == "success":
                        pbar.set_postfix({"last": f"{session_id}: {written}/{total} written, {filtered} filtered"})
                    elif status == "dry_run":
                        pbar.set_postfix({"last": f"{session_id}: {total} facts"})
                    else:
                        pbar.set_postfix({"last": f"{session_id}: {status}"})

                except Exception as e:
                    logger.error(f"[{session_id}] 处理异常: {e}")
                    all_stats.append({
                        "session_id": session_id,
                        "status": "exception",
                        "errors": [str(e)],
                    })

                pbar.update(1)

    # 汇总统计
    summary = {
        "total_sessions": len(all_stats),
        "success_sessions": sum(1 for s in all_stats if s.get("status") == "success"),
        "partial_success_sessions": sum(1 for s in all_stats if s.get("status") == "partial_success"),
        "failed_sessions": sum(1 for s in all_stats if s.get("status") in ("failed", "exception")),
        "dry_run_sessions": sum(1 for s in all_stats if s.get("status") == "dry_run"),
        "total_facts": sum(s.get("total_facts", 0) for s in all_stats),
        "filtered_facts": sum(s.get("filtered_facts", 0) for s in all_stats),
        "rejected_facts": sum(s.get("rejected_facts", 0) for s in all_stats),
        "written_facts": sum(s.get("written_facts", 0) for s in all_stats),
        "skipped_facts": sum(s.get("skipped_facts", 0) for s in all_stats),
        "sessions": all_stats,
    }

    return summary


def print_summary(summary: Dict[str, Any], dry_run: bool = False, filter_with_llm: bool = False, smart_mode: bool = False):
    """打印汇总统计信息。"""
    print("\n" + "=" * 60)
    if dry_run:
        print("统计结果 (DRY RUN - 未实际写入)")
    elif smart_mode:
        print("智能回填完成")
    else:
        print("写入完成")
    print("=" * 60)

    print(f"总 sessions: {summary['total_sessions']}")

    if dry_run and not smart_mode:
        print(f"  - 统计: {summary.get('dry_run_sessions', summary['total_sessions'])}")
    else:
        print(f"  - 成功: {summary['success_sessions']}")
        print(f"  - 部分成功: {summary['partial_success_sessions']}")
        print(f"  - 失败: {summary['failed_sessions']}")

    print(f"\n总 facts 数: {summary['total_facts']}")

    if smart_mode:
        # 智能回填统计
        add_facts = summary.get('add_facts', 0)
        skip_facts = summary.get('skip_facts', 0)
        update_facts = summary.get('update_facts', 0)
        failed_facts = summary.get('failed_facts', 0)

        print(f"\n智能回填结果:")
        print(f"  - ADD (新增): {add_facts}")
        print(f"  - SKIP (跳过重复): {skip_facts}")
        print(f"  - UPDATE (合并更新): {update_facts}")
        if failed_facts > 0:
            print(f"  - 失败: {failed_facts}")

        # 计算去重率
        total_processed = add_facts + skip_facts + update_facts
        if total_processed > 0:
            dedup_rate = skip_facts / total_processed * 100
            print(f"\n去重率: {dedup_rate:.1f}% (跳过 / 总处理)")
            actual_written = add_facts + update_facts
            print(f"实际写入: {actual_written} ({actual_written / summary['total_facts'] * 100:.1f}% of total)")

    elif not dry_run:
        filtered = summary.get('filtered_facts', 0)
        rejected = summary.get('rejected_facts', 0)
        written = summary.get('written_facts', 0)
        skipped = summary.get('skipped_facts', 0)

        if filter_with_llm:
            print(f"  - 质量筛选结果:")
            print(f"    - 通过质量检查并写入: {written}")
            print(f"    - 未通过质量检查 (已过滤): {filtered}")
            if rejected > 0:
                print(f"    - 完全被拒绝的记录: {rejected} 条 facts")
            print(f"  - 写入率: {written / summary['total_facts'] * 100:.1f}%" if summary['total_facts'] > 0 else "  - 写入率: N/A")
        else:
            if filtered > 0:
                print(f"  - 已截断: {filtered}")
            print(f"  - 已写入: {written}")

        if skipped > 0:
            print(f"  - 跳过 (写入失败): {skipped}")

    # 显示失败的 sessions
    failed_sessions = [
        s for s in summary["sessions"]
        if s.get("status") in ("failed", "exception", "partial_success")
    ]

    if failed_sessions:
        print(f"\n有问题的 sessions ({len(failed_sessions)}):")
        for s in failed_sessions[:10]:  # 最多显示 10 个
            errors = s.get("errors", [])
            error_preview = errors[0][:50] if errors else "unknown"
            print(f"  - {s['session_id']}: {s.get('status')} - {error_preview}")

        if len(failed_sessions) > 10:
            print(f"  ... 还有 {len(failed_sessions) - 10} 个")


def main():
    parser = argparse.ArgumentParser(
        description="并发将 linked facts 记录写入数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "linked_facts_dir",
        help="linked facts JSON 文件目录",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="最大并发数 (默认: 4)",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default=None,
        help="只处理指定的 sessions (逗号分隔，如: conv-26,conv-30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：只统计，不实际写入数据库",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="locomo",
        help="Benchmark 名称，用于生成 collection 名称 (默认: locomo)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="模型名称 (默认: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant 服务器地址 (默认: localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant 服务器端口 (默认: 6333)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出统计结果到 JSON 文件",
    )
    parser.add_argument(
        "--no-filter-llm",
        action="store_true",
        help="禁用 LLM 筛选（默认开启 LLM 筛选）",
    )
    parser.add_argument(
        "--max-facts",
        type=int,
        default=2,
        help="每个 query 最多写入的 facts 数量 (默认: 2)",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM 筛选使用的模型 (默认: gpt-4.1-mini)",
    )
    # 智能回填模式参数
    parser.add_argument(
        "--smart",
        action="store_true",
        help="启用智能回填模式：写入前召回已有 memory，LLM 决定 SKIP/UPDATE/DELETE_AND_ADD/ADD",
    )
    parser.add_argument(
        "--smart-top-k",
        type=int,
        default=5,
        help="智能模式下召回的 memory 数量 (默认: 5)",
    )
    parser.add_argument(
        "--decision-model",
        type=str,
        default="gpt-4.1-mini",
        help="智能模式下 LLM 决策使用的模型 (默认: gpt-4.1-mini)",
    )

    args = parser.parse_args()

    # 解析 sessions 参数
    sessions = None
    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",")]

    # 构建 system 配置
    system_config = {
        "benchmark_name": args.benchmark,
        "mem0_config": {
            "backend": "mem0",
            "llm": {
                "provider": "openai",
                "config": {"model": args.model},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": args.qdrant_host,
                    "port": args.qdrant_port,
                    "on_disk": True,
                },
            },
        },
        "fast_model": args.model,
        "slow_model": args.model,
        "write_facts_to_database": True,  # 启用写入
        "record_linked_facts": False,  # 不需要再记录
    }

    # 打印配置信息，方便排查问题
    print("=" * 60)
    if args.smart:
        print("Flush Linked Facts Configuration (SMART MODE)")
    else:
        print("Flush Linked Facts Configuration")
    print("=" * 60)
    print(f"Linked facts dir: {args.linked_facts_dir}")
    print(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    if args.smart:
        print(f"Mode: SMART (召回比对 + LLM 决策)")
        print(f"Smart top_k: {args.smart_top_k}")
        print(f"Decision model: {args.decision_model}")
    else:
        print(f"LLM Filter: {not args.no_filter_llm}")
        print(f"Max facts per query: {args.max_facts}")
    print(f"Dry run: {args.dry_run}")
    if sessions:
        print(f"Sessions: {sessions}")
    print("=" * 60)

    try:
        if args.smart:
            # 使用智能回填模式
            summary = flush_linked_facts_smart(
                linked_facts_dir=args.linked_facts_dir,
                system_config=system_config,
                max_workers=args.max_workers,
                sessions=sessions,
                dry_run=args.dry_run,
                smart_top_k=args.smart_top_k,
                decision_model=args.decision_model,
            )
            print_summary(summary, dry_run=args.dry_run, smart_mode=True)
        else:
            # 使用传统模式
            summary = flush_linked_facts_concurrent(
                linked_facts_dir=args.linked_facts_dir,
                system_config=system_config,
                max_workers=args.max_workers,
                sessions=sessions,
                dry_run=args.dry_run,
                filter_with_llm=not args.no_filter_llm,
                max_facts_per_query=args.max_facts,
                filter_model=args.filter_model,
            )
            print_summary(summary, dry_run=args.dry_run, filter_with_llm=not args.no_filter_llm)

        # 保存统计结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n统计结果已保存到: {args.output}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
