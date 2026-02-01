
import os
from uuid import uuid4
import time
import re
import json
from typing import Any, Dict, Optional, List, Tuple, Set
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# 模块级锁，用于保护 reranker 加载（在模块导入时创建，天然线程安全）
_RERANKER_INIT_LOCK = threading.Lock()

from .base import AnswerResult, MemorySystem, ObserveResult, Turn
import json_repair
# 注意：本模块位于 src/memory/ 下，必须通过顶层包 src 引用 linked_view
from src.linked_view import (
    SummaryIndex,
    EvidenceSet,
    fast_answer,
    ThinkingLLMRouter,
)
from src.linked_view.page_store import PageStore
from src.linked_view.pipelines_slow import EvidenceDoc
from src.linked_view.summary_index import SummaryHit
from src.linked_view.prompts import (
    MEM0_FACT_EXTRACTION_SYSTEM_PROMPT_JSON,
    RESEARCH_INTEGRATION_V2_PROMPT,
    RESEARCH_PLAN_V2_PROMPT,
)

try:
    from openai import OpenAI  # type: ignore

    OPENAI_AVAILABLE = True
except Exception as e:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.error(f"[LinkedViewSystem] OpenAI not available, using FakeLLM: {e}")


class _FakeLLM:
    """回退用 LLM：返回 prompt 末尾 200 字。"""

    def generate(self, prompt: str) -> str:
        return prompt[-200:]


class _OpenAILLM:
    """
    简单的 OpenAI LLM 封装：
    - 使用 ChatCompletion
    - 只负责 fast/slow 两条路径的 generate(prompt)
    """

    def __init__(self, model: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for LinkedViewSystem.")
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)  # type: ignore[call-arg]
        else:
            self.client = OpenAI(api_key=api_key)  # type: ignore[call-arg]
        self.model = model
        # 最近一次调用的 usage，用于上层做统计
        self.last_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def generate(self, prompt: str) -> str:
        """
        对 OpenAI Chat Completions 做一层简单的重试包装：
        - 默认重试 3 次，指数退避（1s, 2s, 4s）
        - 任何异常都会触发重试，最终失败时回退为 prompt 末尾 200 字
        """
        max_retries = 3
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.3,
                )
                # 记录 usage（兼容不同字段名）
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    prompt_tokens = (
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", None)
                        or 0
                    )
                    completion_tokens = (
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", None)
                        or 0
                    )
                    total_tokens = getattr(usage, "total_tokens", None) or (
                        prompt_tokens + completion_tokens
                    )
                    self.last_usage = {
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens),
                        "total_tokens": int(total_tokens),
                    }
                return resp.choices[0].message.content or ""
            except Exception as e:
                # 出错时清零 usage，并在重试前打印告警
                self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                print(
                    f"[LinkedViewSystem] OpenAILLM.generate failed on attempt "
                    f"{attempt}/{max_retries}, will retry: {e}"
                )
                if attempt >= max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2

        # 最终失败时，返回 prompt 末尾 200 字，避免完全崩溃
        return prompt[-200:]


class _UsageTrackingLLM:
    """
    对底层 LLM 做一层包装，记录每次 generate 的 token usage。
    模式参考 core.systems.gam_adapter.UsageTrackingGenerator。
    """

    def __init__(self, inner: Any, name: str, usage_events: List[Dict[str, int]]):
        self._inner = inner
        self._name = name
        self._usage_events = usage_events

    def generate(self, prompt: str) -> str:
        text = self._inner.generate(prompt)
        usage = getattr(self._inner, "last_usage", None) or {}
        event = {
            "generator": self._name,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        self._usage_events.append(event)
        return text

    def __getattr__(self, item):
        return getattr(self._inner, item)


# === LoCoMo 专用短答案 Prompt（对齐 GAM 实现风格） ===


def _make_locomo_summary_prompt(summary: str, question: str) -> str:

    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format "15 July 2023" and provide a specific date whenever possible. For example, if you need to answer "last year," give the specific year of last year rather than just saying "last year." Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def _make_locomo_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.
    
QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def _make_longmemeval_prompt(summary: str, question: str) -> str:
    return f"""\
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.
Answer the question in a short phrase.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""




def _answer_with_summary_locomo(
    category: Optional[int],
    summary: str,
    question: str,
    llm: Any,
) -> str:
    """
    根据 LoCoMo category 选择不同 prompt，并使用底层 LLM.generate 生成短答案。

    注意：这里的 llm 接口是 `generate(prompt: str) -> str`，
    与 GAM 里的 OpenAIGenerator.generate_single 接口略有不同。
    """
    if category == 3:
        prompt = _make_locomo_summary_prompt_category3(summary, question)
    else:
        prompt = _make_locomo_summary_prompt(summary, question)

    try:
        
        text = llm.generate(prompt)
        logger.info(f"[LinkedViewSystem] _answer_with_summary_locomo prompt: {prompt}, text: {text}")
        text = (text or "").strip()
        if not text:
            logger.warning(
                "[LinkedViewSystem] _answer_with_summary_locomo got empty answer. "
                f"summary_len={len(summary)}, question_snippet={question[:50]!r}"
            )
        return text
    except Exception as exc:
        logger.error(f"[LinkedViewSystem] _answer_with_summary_locomo failed: {exc}")
        return ""


def _answer_with_summary_longmemeval(
    summary: str,
    question: str,
    llm: Any,
) -> str:
    """
    使用 LongMemEval 专用 prompt 生成答案。

    注意：这里的 llm 接口是 `generate(prompt: str) -> str`。
    """
    prompt = _make_longmemeval_prompt(summary, question)

    try:
        text = llm.generate(prompt)
        logger.info(f"[LinkedViewSystem] _answer_with_summary_longmemeval prompt: {prompt}, text: {text}")
        text = (text or "").strip()
        if not text:
            logger.warning(
                "[LinkedViewSystem] _answer_with_summary_longmemeval got empty answer. "
                f"summary_len={len(summary)}, question_snippet={question[:50]!r}"
            )
        return text
    except Exception as exc:
        logger.error(f"[LinkedViewSystem] _answer_with_summary_longmemeval failed: {exc}")
        return ""


def build_evidence_from_page_store(page_store: PageStore, hits: List[SummaryHit]) -> List[EvidenceDoc]:
    """
    使用 PageStore + SummaryHit 构造 EvidenceDoc 列表。
    
    - 通过 page_id（存储在 raw_log_id 字段中）从 PageStore 获取页内容
    - 自动跳过缺失 page 的 hit
    """
    docs: List[EvidenceDoc] = []
    for h in hits:
        # 现在 raw_log_id 字段存储的是 page_id
        page_id = h.raw_log_id
        if not page_id:
            continue
        
        page = page_store.get_page_by_id(page_id)
        if page is None:
            continue
        
        # 使用页的原始内容作为 raw_text
        docs.append(
            EvidenceDoc(
                raw_log_id=page_id,  # 使用 page_id
                summary=h.summary_text,
                raw_text=page.content,  # 使用页的完整内容
                score=h.score,
                timestamp=h.timestamp,
            )
        )
    return docs


def _make_llm(kind: str, model_name: str, usage_events: Optional[List[Dict[str, int]]] = None) -> Any:
    """根据环境创建 LLM（优先 OpenAI，失败回退 FakeLLM），并在需要时包上一层 usage tracking。"""

    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            logger.info(f"[LinkedViewSystem] Using OpenAILLM ({kind}) with model={model_name}")
            base_llm = _OpenAILLM(model=model_name)
            if usage_events is not None:
                return _UsageTrackingLLM(base_llm, kind, usage_events)
            return base_llm
        except Exception as e:
            logger.error(f"[LinkedViewSystem] OpenAILLM init failed ({kind}), fallback FakeLLM: {e}")
    logger.error(f"[LinkedViewSystem] Using FakeLLM for {kind}")
    return _FakeLLM()


class LinkedViewSystem(MemorySystem):
    """
    完整 Linked-View 系统实现：
    - observe(): 调用 ingest_chunk，写 RawStore + SummaryIndex(mem0/backend)
    - answer(): 检索 → Router.decide → fast/slow 两条 pipeline
    """

    # 使用细粒度 turns，便于后续基于 turn-window 做 chunking
    #preferred_turns_key = "turns"
    preferred_turns_key = "session_chunks"
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = config or {}

        # === 存储与索引 ===
        # 注意：SummaryIndex 的 Mem0 backend 会持有 Qdrant 本地锁，
        # 因此只能在 reset() 时初始化一次，避免在 __init__ 和 reset() 各建一遍。
        mem0_cfg = cfg.get("mem0_config", {})
        self._mem0_cfg = mem0_cfg

        # 存储 benchmark_name（用于生成 collection 名称）
        self.benchmark_name = cfg.get("benchmark_name", "unknown")

        # PageStore 管理分页逻辑（替代 RawStore），支持持久化存储
        page_size = cfg.get("page_size", 2000)  # 每页最大字符数
        page_store_dir = cfg.get("page_store_dir", "./tmp/page_store")
        self.page_store = PageStore(page_size=page_size, storage_dir=page_store_dir)
        self.index = None  # type: ignore[assignment]
        
        # 索引操作的锁（保护 index.add() 和 index.search()，因为 mem0 可能不是线程安全的）
        self._index_lock = threading.Lock()

        # === LLM 配置 ===
        # llm_memory_system: 用于 mem0 操作、答案生成、研究循环等（原 llm_fast/llm_slow）
        memory_system_model = cfg.get("memory_system_model", cfg.get("fast_model", "gpt-4.1-mini"))
        self._usage_events: List[Dict[str, int]] = []
        self.llm_memory_system = _make_llm("memory_system", memory_system_model, self._usage_events)

        # 为了兼容性，保留 llm_fast 和 llm_slow 的引用
        self.llm_fast = self.llm_memory_system
        self.llm_slow = self.llm_memory_system

        self.per_query_top_k = int(cfg.get("per_query_top_k", 4))
        self.max_decomposed_queries = int(cfg.get("max_decomposed_queries", 3))
        self.use_multi_query_search = bool(cfg.get("use_multi_query_search", False))

        # === Router 配置（支持 OpenAI 和 vLLM 动态切换）===
        router_cfg = cfg.get("router_config", {})
        self.router = self._create_router(router_cfg)

        # 记录 LLM usage（与 GAM 适配器风格一致）
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_api_calls = 0

        # Research loop 参数（对应论文里的 PR(·|Q,Z,Hp(Z)) 的迭代）
        self.max_research_iters = int(cfg.get("max_research_iters", 3))

        # === Reranker 配置 ===
        self.use_reranker = bool(cfg.get("use_reranker", False))
        self.reranker_top_k = int(cfg.get("reranker_top_k", 5))
        self.reranker_model_path = cfg.get(
            "reranker_model_path",
            "Qwen/Qwen3-Reranker-0.6B"
        )
        self._reranker = None  # Lazy loading

        # R-path always uses optimized version (integration + plan loop)
        self.use_optimized_r_path = True

        # === 写回数据库配置 ===
        # 是否将研究过程中提取的 linked facts 写回 summary 数据库
        # 开启后，linked facts 会作为新的 memory 条目存储，供后续查询使用
        self.write_facts_to_database = bool(cfg.get("write_facts_to_database", False))
        logger.info(f"[LinkedViewSystem] write_facts_to_database: {self.write_facts_to_database}")

        # === Linked Facts 记录配置（用于分析和延迟写入）===
        # 是否记录 linked facts（不管是否立即写入，都会记录详细信息供分析）
        self.record_linked_facts = bool(cfg.get("record_linked_facts", True))
        # 待写入的 linked facts 列表（在 reset/load 时初始化）
        self._pending_linked_facts: List[Dict[str, Any]] = []

        # === 统计追踪配置 ===
        # 是否启用详细的统计追踪（用于分析 S/R 路径选择和 linked_fact 召回效果）
        self.enable_stats_tracking = bool(cfg.get("enable_stats_tracking", False))
        self.stats_dir = cfg.get("stats_dir", "./tmp/stats")
        # 统计数据结构（在 reset/load 时初始化）
        self._session_stats: Optional[Dict[str, Any]] = None

        # === 消融实验配置 ===
        # ablation_bm25_only: 只使用 BM25 检索，不使用 Mem0 语义检索（用于消融实验）
        self.ablation_bm25_only = bool(cfg.get("ablation_bm25_only", False))
        if self.ablation_bm25_only:
            logger.info("[LinkedViewSystem] Ablation mode: BM25-only retrieval enabled (no Mem0 semantic search)")

    def _create_router(self, router_cfg: Dict[str, Any]) -> Any:
        """
        根据配置创建 Router 实例。

        router_cfg 支持的参数：
        - type: "openai" | "vllm" (默认 "openai" 使用 ThinkingLLMRouter)
        - model: 模型名称
        - base_url: API base URL（vLLM 必需）
        - api_key: API key（可选，vLLM 可以是任意值）
        - is_thinking_model: 是否是 thinking 模型（vLLM 必需）
        - threshold: router 阈值（默认 0.5）

        Returns:
            Router 实例（ThinkingLLMRouter）
        """
        router_type = router_cfg.get("type", "openai").lower()
        threshold = float(router_cfg.get("threshold", self.config.get("router_threshold", 0.5)))

        if router_type == "vllm":
            # 使用 ThinkingLLMRouter（vLLM）
            try:
                from openai import OpenAI as OpenAIClient
            except ImportError:
                logger.error("[LinkedViewSystem] OpenAI client not available for vLLM router")
                raise RuntimeError("OpenAI client is required for vLLM router. Please install openai package.")

            base_url = router_cfg.get("base_url")
            api_key = router_cfg.get("api_key", "vllm-api-key")
            model = router_cfg.get("model", "Qwen3-4B-Thinking-2507")
            is_thinking_model = router_cfg.get("is_thinking_model", True)

            if not base_url:
                logger.error("[LinkedViewSystem] vLLM router requires 'base_url' in router_config")
                raise ValueError("vLLM router requires 'base_url' in router_config")

            client = OpenAIClient(api_key=api_key, base_url=base_url)
            router = ThinkingLLMRouter(
                client=client,
                model=model,
                is_thinking_model=is_thinking_model,
                threshold=threshold,
            )
            logger.info(
                f"[LinkedViewSystem] Created ThinkingLLMRouter (vLLM): "
                f"model={model}, base_url={base_url}, is_thinking={is_thinking_model}"
            )
            return router

        elif router_type == "openai":
            # 使用 ThinkingLLMRouter（OpenAI API）
            try:
                from openai import OpenAI as OpenAIClient
            except ImportError:
                logger.error("[LinkedViewSystem] OpenAI client not available")
                raise RuntimeError("OpenAI client is required for OpenAI router. Please install openai package.")

            api_key =  os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            model = router_cfg.get("model", "gpt-4.1-mini")

            if not api_key:
                logger.error("[LinkedViewSystem] OpenAI router requires API key")
                raise ValueError("OpenAI router requires API key. Set OPENAI_API_KEY environment variable or provide in router_config.")

            kwargs = {"api_key": api_key}
            kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
            client = OpenAIClient(**kwargs)
            router = ThinkingLLMRouter(
                client=client,
                model=model,
                is_thinking_model=False,  # OpenAI 模型不需要 thinking 模式
                threshold=threshold,
            )
            logger.info(f"[LinkedViewSystem] Created ThinkingLLMRouter (OpenAI): model={model}")
            return router

        else:
            # 未知类型，默认使用 OpenAI
            logger.warning(f"[LinkedViewSystem] Unknown router type '{router_type}', defaulting to 'openai'")
            router_cfg["type"] = "openai"
            return self._create_router(router_cfg)

    def get_system_name(self) -> str:
        return "linked_view"

    def _consume_last_mem0_usage(self) -> Dict[str, int]:
        """
        从 SummaryIndex 读取最近一次 mem0 调用的真实 usage（由 mem0 本地代码采集）。
        返回统一字段，方便写入 cost_metrics。
        """
        usage = {}
        if self.index is not None:
            usage = getattr(self.index, "last_mem0_usage", None) or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "api_calls": int(usage.get("api_calls", 0) or 0),
            "total_latency_ms": int(usage.get("total_latency_ms", 0) or 0),
        }

    def _extract_memories_from_results(self, add_results: Any) -> List[str]:
        """
        从 mem0.add 返回的结果中提取 memory 列表。
        
        Args:
            add_results: mem0.add 返回的结果
            
        Returns:
            memory 文本列表
        """
        memories: List[str] = []
        if add_results is None:
            return memories
        
        try:
            # 处理 {"results": [...]} 格式
            if isinstance(add_results, dict) and "results" in add_results:
                results = add_results["results"]
            elif isinstance(add_results, list):
                results = add_results
            else:
                results = [add_results]
            
            for item in results:
                if isinstance(item, dict):
                    memory = item.get("memory")
                    if memory:
                        memories.append(str(memory))
        except Exception as exc:
            logger.warning(f"[LinkedViewSystem] Failed to extract memories from results: {exc}")
        
        return memories

    def _retry_mem0_add(
        self,
        user_id: str,
        raw_chunk: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> Any:
        """
        带重试的 mem0.add 调用包装器。
        
        使用指数退避策略重试 mem0.add 操作，处理网络错误、服务器断开等临时故障。
        
        Args:
            user_id: 用户 ID
            raw_chunk: 原始数据块（messages 列表）
            metadata: 元数据字典
            max_retries: 最大重试次数（默认 3）
            initial_backoff: 初始退避时间（秒，默认 1.0）
            
        Returns:
            mem0.add 返回的结果
            
        Raises:
            Exception: 如果所有重试都失败，抛出最后一次的异常
        """
        backoff = initial_backoff
        
        for attempt in range(1, max_retries + 1):
            try:
                if self.index is None:
                    raise RuntimeError("SummaryIndex is not initialized")
                
                add_results = self.index.add(
                    user_id=user_id,
                    raw_chunk=raw_chunk,
                    metadata=metadata,
                    infer=True,
                )
                return add_results
            except Exception as e:
                error_msg = str(e).lower()
                # 判断是否是临时性错误（网络错误、服务器断开等）
                is_transient_error = any(
                    keyword in error_msg
                    for keyword in [
                        "server disconnected",
                        "connection",
                        "timeout",
                        "network",
                        "temporary",
                        "retry",
                        "unavailable",
                    ]
                )
                
                if attempt >= max_retries:
                    # 最后一次尝试失败，记录错误并抛出异常
                    logger.error(
                        f"[LinkedViewSystem] mem0.add failed after {max_retries} attempts: {e}"
                    )
                    raise
                
                if is_transient_error:
                    logger.warning(
                        f"[LinkedViewSystem] mem0.add failed on attempt {attempt}/{max_retries} "
                        f"(transient error), will retry after {backoff}s: {e}"
                    )
                    time.sleep(backoff)
                    backoff *= 2  # 指数退避
                else:
                    # 非临时性错误，不重试直接抛出
                    logger.error(
                        f"[LinkedViewSystem] mem0.add failed on attempt {attempt}/{max_retries} "
                        f"(non-transient error), aborting: {e}"
                    )
                    raise

    def auto_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        对没有存入 mem0 的页调用 mem0.add(infer=True)。
        
        - 如果指定了session_id，只处理该session的页
        - 如果没有指定session_id，处理所有session的页
        - 对每个没有存入mem0的页调用mem0.add(infer=True)
        - 将结果存储到page_store
        
        Args:
            session_id: 可选的session ID，如果指定则只处理该session的页
            
        Returns:
            包含统计信息的字典：
            {
                "total_pages": int,  # 找到的没有存入mem0的页总数
                "processed_pages": int,  # 成功处理的页数
                "total_tokens_in": int,
                "total_tokens_out": int,
                "total_api_calls": int,
            }
        """
        total_tokens_in = 0
        total_tokens_out = 0
        total_api_calls = 0
        processed_pages = 0
        
        # 获取需要处理的页（没有存入mem0的页）
        pages_not_stored = self.page_store.get_pages_not_stored_to_mem0(session_id)
        
        total_pages = len(pages_not_stored)
        logger.info(f"[LinkedViewSystem] Found {total_pages} pages not stored to mem0")
        
        if total_pages == 0:
            return {
                "total_pages": 0,
                "processed_pages": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_api_calls": 0,
            }
        
        # 处理每个页
        for page in pages_not_stored:
            try:
                user_id = page.user_id

                # 调用 mem0.add(infer=True)
                if self.index is None:
                    logger.warning(f"[LinkedViewSystem] index is None, skipping page {page.page_id}")
                    continue
                
                # 将整页内容拆成 messages（role 只能是 user/assistant/system）
                messages: List[Dict[str, Any]] = []
                for line in (page.content or "").splitlines():
                    line = (line or "").strip()
                    if not line:
                        continue
                    m = re.match(r"^\[(?P<ts>[^\]]*)\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$", line)
                    if m:
                        ts = (m.group("ts") or "").strip() or "unknown"
                        speaker_norm = (m.group("speaker") or "").strip()
                        text = (m.group("text") or "").strip()
                        role = "assistant" if speaker_norm.lower() in {"assistant", "ai"} else "user"
                        content = f"[{ts}] {speaker_norm}: {text}" if speaker_norm else f"[{ts}] {text}"
                        msg: Dict[str, Any] = {"role": role, "content": content}
                        if speaker_norm:
                            msg["name"] = speaker_norm
                        messages.append(msg)
                    else:
                        # 兜底：无法解析的行当作 user 内容
                        messages.append({"role": "user", "content": line})
                if not messages:
                    messages = [{"role": "user", "content": page.content or ""}]

                # 使用带重试的 mem0.add 调用
                add_results = self._retry_mem0_add(
                    user_id=user_id,
                    raw_chunk=messages,
                    metadata={
                        "raw_log_id": page.page_id,
                        "page_id": page.page_id,
                        "raw_log_ids": page.raw_log_ids,
                        "timestamp": page.created_at,
                        "session_id": page.session_id,
                        "chunk_type": "page_raw",
                    },
                )
                
                # 提取 memory 列表
                memories = self._extract_memories_from_results(add_results)
                
                # 存储 memory 列表到 page_store
                self.page_store.update_page_mem0_status(page.page_id, memories)
                
                mem0_u = self._consume_last_mem0_usage()
                total_tokens_in += mem0_u["prompt_tokens"]
                total_tokens_out += mem0_u["completion_tokens"]
                total_api_calls += mem0_u["api_calls"]
                
                processed_pages += 1
                
                logger.info(
                    f"[LinkedViewSystem] Auto-stored page {page.page_id} to mem0 "
                    f"(session={page.session_id}, memories_count={len(memories)}, "
                    f"tokens_in={mem0_u['prompt_tokens']}, tokens_out={mem0_u['completion_tokens']},api_calls={mem0_u['api_calls']})"
                )
            except Exception as e:
                logger.error(f"[LinkedViewSystem] Failed to auto-store page {page.page_id} to mem0 after retries: {e}")
                continue
        
        logger.info(
            f"[LinkedViewSystem] Auto-summary completed: {processed_pages}/{total_pages} pages processed, "
            f"total_tokens_in={total_tokens_in}, total_tokens_out={total_tokens_out}"
        )
        
        return {
            "total_pages": total_pages,
            "processed_pages": processed_pages,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_api_calls": total_api_calls,
        }

    # === 生命周期 ===
    def reset(self, session_id: str) -> None:
        super().reset(session_id)
        # 重置 page_store
        if self.page_store:
            self.page_store.reset_session(session_id)
        # 每个 session 独立 PageStore / SummaryIndex
        cfg = self.config or {}
        page_size = cfg.get("page_size", 2000)
        page_store_dir = cfg.get("page_store_dir", "./tmp/page_store")
        self.page_store = PageStore(page_size=page_size, storage_dir=page_store_dir)
        mem0_cfg = dict(cfg.get("mem0_config", self._mem0_cfg or {}))
        
        # 为每个 session 创建独立的 Qdrant collection（session-level isolation）
        # 使用 TierMem 命名格式：TierMem_{benchmark_name}_{session_id}
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")

        # 获取 benchmark_name（从配置或实例属性）
        benchmark_name = cfg.get("benchmark_name") or getattr(self, "benchmark_name", "unknown")
        # 生成 collection 名称：TierMem_{benchmark_name}_{session_id}
        collection_name = f"TierMem_{benchmark_name}_{safe_session_id}"
        
        # 更新 mem0_config 中的 collection_name
        if "vector_store" in mem0_cfg and "config" in mem0_cfg["vector_store"]:
            mem0_cfg["vector_store"]["config"]["collection_name"] = collection_name
            logger.info(f"[LinkedViewSystem] Reset: Using collection_name={collection_name} for session_id={session_id}")
        
        # 注入自定义 fact extraction prompt（本地 mem0 才能通过 MemoryConfig 生效）
        # 云端 MemoryClient 需要用 update_project(custom_instructions=...) 方式，这里不做。
        if mem0_cfg.get("backend") == "mem0" and mem0_cfg.get("mode", "local") != "api":
            mem0_cfg.setdefault("custom_fact_extraction_prompt", MEM0_FACT_EXTRACTION_SYSTEM_PROMPT_JSON)
        # 默认使用 in-memory 占位实现，避免在未正确配置 mem0 时报错；
        # 如需接入真实 Mem0，可显式设置 mem0_config.backend="mem0" 并传入合法配置对象。
        backend = mem0_cfg.get("backend", "inmemory")
        self.index = SummaryIndex(backend=backend, mem0_config=mem0_cfg)
        # 注意：由于每个 session 已有独立的 collection，不需要通过 user_id 来隔离
        # 但为了兼容性，仍然使用 user_id（格式：{session_id}:user）
        user_id = f"{session_id}:user"
        # 清空该 session 的 collection（因为 reset 意味着重新开始）
        # 由于每个 session 有独立的 collection，delete_all 会清空整个 collection
        if getattr(self.index, "_mem0", None) is not None:
            try:
                self.index._mem0.delete_all(user_id=user_id)
                logger.info(f"[LinkedViewSystem] Reset: Cleared collection {collection_name} for session_id={session_id}")
            except Exception as exc:
                logger.warning(f"[LinkedViewSystem] Reset: Failed to clear collection {collection_name}: {exc}")

        # 初始化统计追踪（如果启用）
        self._init_session_stats(session_id)

        # 初始化 linked facts 记录列表
        self._pending_linked_facts = []

    def load(self, session_id: str) -> None:
        """
        加载/恢复一个已有 session（不清空状态）。

        - PageStore：从持久化文件懒加载对应 session 的页数据
        - SummaryIndex：不做 delete_all / reset，仅确保后端可用（mem0 后端本身持久化）
        """
        super().reset(session_id)

        cfg = self.config or {}

        # 确保 PageStore 已初始化（不 reset，不删除文件）
        page_size = cfg.get("page_size", 2000)
        page_store_dir = cfg.get("page_store_dir", "./tmp/page_store")
        expected_dir = Path(page_store_dir)
        if (
            self.page_store is None
            or getattr(self.page_store, "page_size", None) != page_size
            or getattr(self.page_store, "storage_dir", None) != expected_dir
        ):
            self.page_store = PageStore(page_size=page_size, storage_dir=page_store_dir)

        # 触发 PageStore 对该 session 的文件加载（内部会做缓存，重复调用无副作用）
        try:
            _ = self.page_store.get_pages_by_session(session_id)
        except Exception as exc:
            logger.warning(f"[LinkedViewSystem] PageStore load session failed: session={session_id}, err={exc}")

        # 确保 SummaryIndex 可用（不清空 mem0）
        mem0_cfg = dict(cfg.get("mem0_config", self._mem0_cfg or {}))
        
        # 为每个 session 使用独立的 Qdrant collection（session-level isolation）
        # 使用与 reset() 相同的命名逻辑，确保可以恢复数据
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        
        # 获取 benchmark_name（从配置或实例属性）
        benchmark_name = cfg.get("benchmark_name") or getattr(self, "benchmark_name", "unknown")
        # 生成 collection 名称：TierMem_{benchmark_name}_{session_id}
        collection_name = f"TierMem_{benchmark_name}_{safe_session_id}"
        
        # 更新 mem0_config 中的 collection_name
        if "vector_store" in mem0_cfg and "config" in mem0_cfg["vector_store"]:
            mem0_cfg["vector_store"]["config"]["collection_name"] = collection_name
            logger.info(f"[LinkedViewSystem] Load: Using collection_name={collection_name} for session_id={session_id}")
        
        if mem0_cfg.get("backend") == "mem0" and mem0_cfg.get("mode", "local") != "api":
            mem0_cfg.setdefault("custom_fact_extraction_prompt", MEM0_FACT_EXTRACTION_SYSTEM_PROMPT_JSON)
        backend = mem0_cfg.get("backend", "inmemory")
        # 每次 load 时都重新创建 SummaryIndex，确保使用正确的 collection_name
        self.index = SummaryIndex(backend=backend, mem0_config=mem0_cfg)

        # 初始化统计追踪（如果启用）- load 时尝试加载已有统计
        self._init_session_stats(session_id, load_existing=True)

        # 初始化 linked facts 记录列表（load 时不清空已有记录）
        if not hasattr(self, '_pending_linked_facts') or self._pending_linked_facts is None:
            self._pending_linked_facts = []

    # === 写入 ===
    def observe(self, turn: Turn) -> ObserveResult:
        """
        写入逻辑（分页版本）：
        - turn 先累积到 PageStore（一页一页攒）
        - 当页满时调用 mem0.add(infer=True)，将整页内容存储到 mem0
        - 将 mem0.add 返回的 memory 列表存储到 page_store（只存 memory 文本）
        - 标记页为已存入 mem0（stored_to_mem0=True）
        - user_id 简单用 (session_id, speaker) 组合
        - timestamp 优先用 turn.timestamp，没有则用 "unknown"
        """

        start = time.time()
        tokens_in = 0
        tokens_out = 0
        api_calls = 0
        mem0_tokens_in = 0
        mem0_tokens_out = 0
        mem0_api_calls = 0
        mem0_latency_ms = 0

        session_id = self.current_session_id or "unknown_session"
        user_id = f"{session_id}:user"
        timestamp = turn.timestamp or "unknown"

        # 1. 添加到 PageStore
        # PageStore 里保留带 timestamp + speaker 的原始记录（便于回放/分页总结）
        if turn.dia_id:
            turn_content = f"[{timestamp}] [dia_id: {turn.dia_id}] {turn.speaker}: {turn.text}"
        else:
            turn_content = f"[{timestamp}]  {turn.speaker}: {turn.text}"
        turn_id = uuid4().hex
        
        page, is_full = self.page_store.add_content(
            session_id=session_id,
            user_id=user_id,
            raw_log_id=turn_id,
            content=turn_content,
            timestamp=timestamp,
        )

        # 2. 页满时写入 mem0（调用 mem0.add(infer=True)）
        # 注意：只锁住 index.add() 操作，因为 PageStore 已经是线程安全的
        if is_full and self.index is not None:
            # 将整页内容拆成 messages（role 只能是 user/assistant/system）
            # PageStore.content 的每行形如："[ts]  speaker: text"
            messages: List[Dict[str, Any]] = []
            for line in (page.content or "").splitlines():
                line = (line or "").strip()
                if not line:
                    continue
                m = re.match(r"^\[(?P<ts>[^\]]*)\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$", line)
                if m:
                    ts = (m.group("ts") or "").strip() or "unknown"
                    speaker_norm = (m.group("speaker") or "").strip()
                    text = (m.group("text") or "").strip()
                    role = "assistant" if speaker_norm.lower() in {"assistant", "ai"} else "user"
                    # 始终把 speaker 写进 content，保留主体上下文（LoCoMo 多人对话很关键）
                    content = f"[{ts}] {speaker_norm}: {text}" if speaker_norm else f"[{ts}] {text}"
                    msg: Dict[str, Any] = {"role": role, "content": content}
                    if speaker_norm:
                        msg["name"] = speaker_norm
                    messages.append(msg)
                else:
                    # 兜底：无法解析的行当作 user 内容
                    messages.append({"role": "user", "content": line})
            if not messages:
                messages = [{"role": "user", "content": page.content or ""}]

            # 尝试并发调用 mem0.add()（带重试）
            # 注意：虽然 mem0 可能有共享状态（LLM 客户端、embedding_model），
            # 但 Qdrant 客户端本身是线程安全的，而且每个 session 使用独立的 collection
            # 如果遇到线程安全问题，可以重新启用锁
            # with self._index_lock:
            try:
                add_results = self._retry_mem0_add(
                    user_id=user_id,
                    raw_chunk=messages,
                    metadata={
                        "raw_log_id": page.page_id,
                        "page_id": page.page_id,
                        "raw_log_ids": page.raw_log_ids,
                        "timestamp": page.created_at,
                        "session_id": session_id,
                        "chunk_type": "page_raw",
                        "source_type": "original",  # 标记为原始写入（区分 QA 阶段写回的 linked_fact）
                    },
                )
                
                # 提取 memory 列表（在锁外执行，因为只是读取 add_results）
                memories = self._extract_memories_from_results(add_results)
            except Exception as e:
                # 重试失败，记录错误但不中断 observe 流程
                logger.error(
                    f"[LinkedViewSystem] Failed to store page {page.page_id} to mem0 after retries: {e}"
                )
                memories = []
            
            # 存储 memory 列表到 page_store（PageStore 内部已经有锁保护）
            self.page_store.update_page_mem0_status(page.page_id, memories)
            
            mem0_u = self._consume_last_mem0_usage()
            mem0_tokens_in += mem0_u["prompt_tokens"]
            mem0_tokens_out += mem0_u["completion_tokens"]
            mem0_api_calls += mem0_u["api_calls"]
            mem0_latency_ms += mem0_u["total_latency_ms"]
            
            logger.info(
                f"[LinkedViewSystem] Page {page.page_id} stored to mem0 "
                f"(memories_count={len(memories)}, tokens_in={mem0_u['prompt_tokens']}, "
                f"tokens_out={mem0_u['completion_tokens']}, api_calls={mem0_u['api_calls']})"
            )

        # 把 mem0 内部真实 token/calls 计入写入阶段 cost（用于论文实验）
        tokens_in += mem0_tokens_in
        tokens_out += mem0_tokens_out
        api_calls += mem0_api_calls

        latency_ms = int((time.time() - start) * 1000)
        storage_stats = {
            "current_page_size": len(page.content) if page else 0,
            "page_id": page.page_id if page else None,
        }
        cost = {
            "total_latency_ms": latency_ms,
            "total_tokens_in": tokens_in,
            "total_tokens_out": tokens_out,
            "api_calls_count": api_calls,
            "mem0_prompt_tokens": mem0_tokens_in,
            "mem0_completion_tokens": mem0_tokens_out,
            "mem0_total_latency_ms": mem0_latency_ms,
            "mem0_api_calls": mem0_api_calls,
        }
        return ObserveResult(cost_metrics=cost, storage_stats=storage_stats)

        # === 读取 ===
    def answer(self, query: str, meta: Optional[Dict[str, Any]] = None,action: Optional[str] = None) -> AnswerResult:
        """
        完整 S/R 流程：
        1) SummaryIndex.search → hits
        2) Router.decide(EvidenceSet) → "S" / "R"
        3) S: fast_answer
           R: build_evidence + guided_research
        """

        meta = meta or {}
        start = time.time()

        session_id = self.current_session_id or "unknown_session"
        # 简化：只查 user speaker 的记忆（真实系统可用更细致的 user_id 策略）
        user_id = f"{session_id}:user"

        # LoCoMo: category（1/2/3）用于选择短答案 prompt
        category = meta.get("category")

        # 统计：mem0.search 的真实 token/calls（来自 mem0 本地 usage recorder）
        mem0_search_tokens_in = 0
        mem0_search_tokens_out = 0
        mem0_search_api_calls = 0
        mem0_search_latency_ms = 0

        retrieval_latency_ms = 0

        def _search(q: str, k: int) -> List[SummaryHit]:
            nonlocal retrieval_latency_ms, mem0_search_tokens_in, mem0_search_tokens_out, mem0_search_api_calls, mem0_search_latency_ms
            # === 正常模式：使用 Mem0 语义检索 ===
            t0 = time.time()
            hits = self.index.search(user_id=user_id, query=q, top_k=k)
            retrieval_latency_ms += int((time.time() - t0) * 1000)
            u = self._consume_last_mem0_usage()
            mem0_search_tokens_in += u["prompt_tokens"]
            mem0_search_tokens_out += u["completion_tokens"]
            mem0_search_api_calls += u["api_calls"]
            mem0_search_latency_ms += u["total_latency_ms"]
            return hits

        base_hits = _search(query, int(self.config.get("top_k", 5)))

        global_context_for_router = ""
        usage_before_idx = len(self._usage_events)
        
        # Router token 消耗统计（单独统计）
        router_tokens_in = 0
        router_tokens_out = 0
        router_api_calls = 0
        router_latency_ms = 0  # Router 耗时统计
        
        # === 迭代检索逻辑（支持 QUERY）===
        max_retrieval_rounds = self.max_research_iters
        retrieval_history: List[Dict[str, Any]] = []
        all_retrieved_hits = list(base_hits)
        # 使用 (raw_log_id, summary_text) 作为唯一标识，允许同一个 page 的不同 summary 都被保留
        seen_hit_keys: Set[tuple] = {
            (h.raw_log_id or "", h.summary_text or "") 
            for h in base_hits 
            if h.raw_log_id or h.summary_text
        }

        # 记录初始检索
        retrieval_history.append({
            "round": 0,
            "query": query,
            "hits_count": len(base_hits),
            "action": "initial_search"
        })
        #action = "R"
        # Router 迭代决策（最多 max_retrieval_rounds 轮）
        #final_action = "R"
        for round_idx in range(max_retrieval_rounds):
            # 构建当前 evidence（包含所有已检索的 hits）
            evidence = EvidenceSet(
                query=query,
                hits=all_retrieved_hits,
                extra_context=global_context_for_router
            )

            # 如果 action 被外部指定，使用外部值；否则调用 router
            if action is not None and round_idx == 0:
                # 第一轮使用外部指定的 action
                current_action = action
                logger.info(f"[LinkedViewSystem] Round {round_idx}: Action override: {current_action}")
            else:
                # 调用 router 决策（记录耗时）
                router_start = time.time()
                current_action = self.router.decide(
                    evidence,
                    retrieval_round=round_idx,
                    history=retrieval_history
                )
                router_latency_ms += int((time.time() - router_start) * 1000)
                logger.info(f"[LinkedViewSystem] Round {round_idx}: Router decided: {current_action}")
                
                # 收集 router 的 token usage
                # ThinkingLLMRouter 会单独记录 token usage
                if hasattr(self.router, "last_usage") and self.router.last_usage:
                    router_usage = self.router.last_usage
                    router_tokens_in += int(router_usage.get("prompt_tokens", 0))
                    router_tokens_out += int(router_usage.get("completion_tokens", 0))
                    router_api_calls += 1
                    logger.debug(f"[LinkedViewSystem] Round {round_idx}: Router usage: {router_usage}")

            # 判断 action 类型（支持 QUERY 和兼容 REFINE）
            is_query_action = current_action.startswith("QUERY:") or current_action.startswith("REFINE:")
            if is_query_action:
                # 提取新 query（兼容两种格式）
                if current_action.startswith("QUERY:"):
                    new_query = current_action[6:].strip()
                else:
                    new_query = current_action[7:].strip()

                if not new_query:
                    logger.warning(f"[LinkedViewSystem] Round {round_idx}: QUERY with empty query, falling back to R")
                    final_action = "R"
                    break

                # 用新 query 检索
                logger.info(f"[LinkedViewSystem] Round {round_idx}: Querying with: {new_query}")
                new_hits = _search(new_query, int(self.config.get("top_k", 5)))

                # 合并新 hits（去重：基于 (raw_log_id, summary_text) 组合）
                new_unique_hits = []
                for h in new_hits:
                    hit_key = (h.raw_log_id or "", h.summary_text or "")
                    if hit_key not in seen_hit_keys:
                        new_unique_hits.append(h)
                        seen_hit_keys.add(hit_key)
                        all_retrieved_hits.append(h)

                # 记录检索历史
                retrieval_history.append({
                    "round": round_idx + 1,
                    "query": new_query,
                    "hits_count": len(new_hits),
                    "new_unique_hits": len(new_unique_hits),
                    "action": "QUERY"
                })

                logger.info(
                    f"[LinkedViewSystem] Round {round_idx}: Query search found {len(new_hits)} hits "
                    f"({len(new_unique_hits)} new unique hits)"
                )

                # 如果是最后一轮，强制决策 S/R
                if round_idx == max_retrieval_rounds - 1:
                    logger.info(f"[LinkedViewSystem] Reached max retrieval rounds, forcing final decision")
                    # 再调用一次 router，但不允许 QUERY（记录耗时）
                    evidence = EvidenceSet(
                        query=query,
                        hits=all_retrieved_hits,
                        extra_context=global_context_for_router
                    )
                    router_start = time.time()
                    final_action = self.router.decide(
                        evidence,
                        retrieval_round=round_idx + 1,
                        history=retrieval_history
                    )
                    router_latency_ms += int((time.time() - router_start) * 1000)
                    # 收集 router 的 token usage（如果是 ThinkingLLMRouter）
                    if hasattr(self.router, "last_usage") and self.router.last_usage:
                        router_usage = self.router.last_usage
                        router_tokens_in += int(router_usage.get("prompt_tokens", 0))
                        router_tokens_out += int(router_usage.get("completion_tokens", 0))
                        router_api_calls += 1
                    
                    # 如果还是 QUERY，强制转为 R
                    if final_action.startswith("QUERY:") or final_action.startswith("REFINE:"):
                        logger.warning(f"[LinkedViewSystem] Router still returned QUERY at max round, forcing R")
                        final_action = "R"
                    break

            elif current_action in ("S", "R"):
                # 最终决策
                final_action = current_action
                retrieval_history.append({
                    "round": round_idx + 1,
                    "query": query,
                    "hits_count": len(all_retrieved_hits),
                    "action": final_action
                })
                logger.info(f"[LinkedViewSystem] Round {round_idx}: Final action decided: {final_action}")
                break
            else:
                # 未知 action，保守走 R
                logger.warning(f"[LinkedViewSystem] Round {round_idx}: Unknown action '{current_action}', falling back to R")
                final_action = "R"
                break

        # 如果循环结束还没有 final_action（理论上不应该发生），回退到 R
        if final_action is None:
            logger.warning(f"[LinkedViewSystem] No final action after {max_retrieval_rounds} rounds, falling back to R")
            final_action = "R"

        action = final_action
        generation_start = time.time()
        logger.info(f"[LinkedViewSystem] Final action: {action}, total_hits: {len(all_retrieved_hits)}, retrieval_rounds: {round_idx}")

        # === 统计追踪：记录查询事件 ===
        query_idx = meta.get("query_idx", meta.get("qa_idx", 0))  # 从 meta 获取 query 索引
        linked_fact_count, original_count, linked_fact_details = self._count_linked_fact_hits(all_retrieved_hits)
        router_confidence = getattr(self.router, "last_confidence", None)  # 获取 router 置信度（如果有）
        self._record_query_event(
            session_id=session_id,
            query_idx=query_idx,
            query=query,
            action=action,
            total_hits=len(all_retrieved_hits),
            linked_fact_hits=linked_fact_count,
            original_hits=original_count,
            linked_fact_hit_details=linked_fact_details,
            router_confidence=router_confidence,
        )

        # 初始化变量（用于两种路径）
        all_hits = []
        research_iterations = []
        research_summary = ""
        if action == "S":
            # === S：Summary-only ===
            # 使用 all_retrieved_hits 而不是 base_hits，因为 router 可能改写了 query 并重新检索
            hits_for_answer = all_retrieved_hits
            raw_docs_used = 0
            summary_block = "\n".join(f"- {h.summary_text}" for h in hits_for_answer)
            
            # 根据 benchmark_name 选择 prompt
            benchmark_name = getattr(self, "benchmark_name", "").lower()
            is_longmemeval = "longmemeval" in benchmark_name
            is_locomo = "locomo" in benchmark_name
            
            if is_longmemeval:
                # LongMemEval：使用专用 prompt
                answer = _answer_with_summary_longmemeval(
                    summary=summary_block,
                    question=query,
                    llm=self.llm_fast,
                )
            elif is_locomo and category is not None:
                # LoCoMo：如果带 category，走短答案 prompt，对齐 GAM 风格
                # category == 3 时使用 _make_locomo_summary_prompt_category3
                answer = _answer_with_summary_locomo(
                    category=category,
                    summary=summary_block,
                    question=query,
                    llm=self.llm_fast,
                )
            else:
                answer = fast_answer(self.llm_fast, query, hits_for_answer)

        else:
            # === R：Research loop over Mem0 + PageStore ===
            # 支持两种模式：
            # 1. 优化版 R 路径：使用合并的 _research_step（1次LLM调用/迭代）
            # 2. 原版 R 路径：plan + reflection + integration（向后兼容）

            # 初始化：从初始检索结果开始
            all_hits = list(base_hits)
            seen_ids = {h.raw_log_id for h in base_hits if h.raw_log_id}
            page_summaries: Dict[str, List[str]] = {}
            for h in base_hits:
                if h.raw_log_id and h.summary_text:
                    page_summaries.setdefault(h.raw_log_id, []).append(h.summary_text)

            # 记录已搜索的查询（用于去重）
            searched_mem0_queries: Set[str] = set()
            searched_keyword_sets: Set[Tuple[str, ...]] = set()

            # 第一步：用原问题做初始 BM25 检索
            logger.info(f"[LinkedViewSystem] Initial BM25 search with original query: {query}")
            session_id = self.current_session_id

            bm25_results = self.page_store.search_pages_by_keywords(
                keywords=[query],
                session_id=session_id,
                limit=int(self.config.get("top_k", 8))
            )

            # 将 BM25 结果转换为 SummaryHit 并合并
            for page, score in bm25_results:
                summary_text = page.summary if page.summary else (page.content[:500] + "..." if len(page.content) > 500 else page.content)
                hit = SummaryHit(
                    summary_text=summary_text,
                    score=float(score),
                    raw_log_id=page.page_id,
                    timestamp=page.created_at,
                    session_id=page.session_id,
                    extra={"source": "bm25_initial"}
                )
                page_id = hit.raw_log_id
                if page_id and page_id not in seen_ids:
                    all_hits.append(hit)
                    seen_ids.add(page_id)
                if page_id and summary_text:
                    page_summaries.setdefault(page_id, []).append(summary_text)

            logger.info(f"[LinkedViewSystem] After initial BM25: {len(all_hits)} total hits")

            # === Rerank 初始结果 ===
            # current_hits: 当前轮次要整合的 hits（每轮更新为新检索的结果）
            # processed_page_ids: 已处理过的 page_id（用于去重，避免重复整合）
            # pending_hits: 超出 top_k 但还没被整合的 hits（保留到后续轮次）
            current_hits: List[SummaryHit] = []
            pending_hits: List[SummaryHit] = []  # 新增：候选池
            processed_page_ids: Set[str] = set()

            if self.use_reranker and all_hits:
                # 使用 return_all=True 获取所有排序后的 hits（按 page 级别排序）
                reranked_all = self._rerank_hits(query, all_hits, self.reranker_top_k, return_all=True)
                # 按 page 数量切分：前 top_k 个 pages 给当前轮整合，剩余放入候选池
                current_hits, pending_hits = self._split_hits_by_page_count(reranked_all, self.reranker_top_k)
                processed_page_ids = {h.raw_log_id for h in current_hits if h.raw_log_id}
                unique_pages_count = len(processed_page_ids)
                logger.info(f"[LinkedViewSystem] After initial rerank: {unique_pages_count} pages ({len(current_hits)} hits) for integration, {len(pending_hits)} pending hits")
            else:
                # 没有 reranker，按 page 数量切分
                current_hits, pending_hits = self._split_hits_by_page_count(all_hits, self.reranker_top_k)
                processed_page_ids = {h.raw_log_id for h in current_hits if h.raw_log_id}

            # 研究循环
            research_iterations = []
            research_summary = ""

            # === 优化版 R 路径 V2：Integration + Plan 分离 ===
            # 流程：检索 → Rerank → 整合(提取linked facts) → 计划(决定是否继续) → 循环
            # 重要：每轮只整合新检索到的内容，旧的已经被整合成 linked_facts 了
            # 新增：超出 top_k 的 hits 保留在 pending_hits，下轮和新 hits 一起 rerank

            all_linked_facts = []  # 收集所有迭代的 linked facts

            for it in range(1, self.max_research_iters + 1):
                logger.info(f"[LinkedViewSystem] Optimized V2 research iteration {it}/{self.max_research_iters}")

                # Step 1: Integration - 整合当前轮次的 hits，提取 linked facts
                # current_hits: 第一轮是初始 rerank 结果，后续轮是新检索+rerank 的结果
                integration_result = self._research_integration_v2(
                    query=query,
                    hits=current_hits,
                    _page_summaries=page_summaries,
                )

                linked_facts = integration_result.get("linked_facts", [])
                coverage_assessment = integration_result.get("coverage_assessment", "")

                # 收集 linked facts（用于后续写回数据库）
                all_linked_facts.extend(linked_facts)

                # 只记录当前轮的 linked facts（用于分析），不立即写入数据库
                if self.record_linked_facts and linked_facts:
                    self._record_linked_facts(
                        question=query,
                        linked_facts=linked_facts,
                        raw_hits=current_hits,
                        session_id=session_id,
                        query_idx=query_idx,
                    )

                logger.info(f"[LinkedViewSystem] Integration: {len(linked_facts)} linked facts extracted")

                # Step 2: Plan - 决定是否需要更多搜索
                # 构建当前 facts 文本
                current_facts_text = "\n".join([
                    f"- {lf.get('fact', '')}" for lf in all_linked_facts
                ]) if all_linked_facts else "No facts extracted yet."

                plan_result = self._research_plan_v2(
                    query=query,
                    current_facts=current_facts_text,
                    coverage_assessment=coverage_assessment,
                    research_history=research_iterations,
                    searched_queries=searched_mem0_queries,
                    searched_keywords=searched_keyword_sets,
                )

                decision = plan_result.get("decision", "DONE").upper()
                reasoning = plan_result.get("reasoning", "")
                search_commands = plan_result.get("search_commands", [])

                logger.info(f"[LinkedViewSystem] Plan: decision={decision}, reasoning={reasoning[:100]}...")

                # 如果决定结束，退出循环
                if decision == "DONE":
                    research_iterations.append({
                        "iteration": it,
                        "decision": decision,
                        "linked_facts": linked_facts,
                        "coverage_assessment": coverage_assessment,
                        "search_commands": [],
                        "new_hits_count": 0,
                    })
                    logger.info(f"[LinkedViewSystem] Research ended at iteration {it}, decision=DONE")
                    break

                # Step 3: 执行搜索命令
                new_hits = []
                executed_commands = []

                for cmd in search_commands:
                    cmd_type = cmd.get("type", "").upper()

                    if cmd_type == "MEM0_SEARCH":
                        cmd_query = cmd.get("query", "")
                        if cmd_query and cmd_query.lower() not in searched_mem0_queries:
                            searched_mem0_queries.add(cmd_query.lower())
                            # Mem0 搜索（消融模式下 _search 会自动用 BM25）
                            mem0_hits = _search(cmd_query, int(self.config.get("top_k", 8)))
                            new_hits.extend(mem0_hits)
                            executed_commands.append({"type": "MEM0_SEARCH" if not self.ablation_bm25_only else "BM25_SEARCH", "query": cmd_query, "hits_count": len(mem0_hits)})
                            logger.info(f"[LinkedViewSystem] {'BM25_SEARCH (ablation)' if self.ablation_bm25_only else 'MEM0_SEARCH'}: {cmd_query}, found {len(mem0_hits)} hits")

                            # 同时用 query 做 BM25 搜索（仅在非消融模式下，因为消融模式 _search 已经是 BM25）
                            if not self.ablation_bm25_only:
                                bm25_results = self.page_store.search_pages_by_keywords(
                                    keywords=[cmd_query],
                                    session_id=session_id,
                                    limit=int(self.config.get("top_k", 8))
                                )
                                for page, score in bm25_results:
                                    summary_text = page.summary if page.summary else (page.content[:500] + "..." if len(page.content) > 500 else page.content)
                                    hit = SummaryHit(
                                        summary_text=summary_text,
                                        score=float(score),
                                        raw_log_id=page.page_id,
                                        timestamp=page.created_at,
                                        session_id=page.session_id,
                                        extra={"source": "bm25_query"}
                                    )
                                    new_hits.append(hit)
                                executed_commands.append({"type": "BM25_SEARCH", "query": cmd_query, "hits_count": len(bm25_results)})
                                logger.info(f"[LinkedViewSystem] BM25_SEARCH: {cmd_query}, found {len(bm25_results)} hits")

                    elif cmd_type == "KEYWORD_SEARCH":
                        keywords = cmd.get("keywords", [])
                        if keywords:
                            kw_tuple = tuple(sorted([k.lower() for k in keywords]))
                            if kw_tuple not in searched_keyword_sets:
                                searched_keyword_sets.add(kw_tuple)
                                keyword_results = self.page_store.search_pages_by_keywords(
                                    keywords=keywords,
                                    session_id=session_id,
                                    limit=int(self.config.get("top_k", 8))
                                )
                                for page, score in keyword_results:
                                    summary_text = page.summary if page.summary else (page.content[:500] + "..." if len(page.content) > 500 else page.content)
                                    hit = SummaryHit(
                                        summary_text=summary_text,
                                        score=float(score),
                                        raw_log_id=page.page_id,
                                        timestamp=page.created_at,
                                        session_id=page.session_id,
                                        extra={"source": "bm25_keywords"}
                                    )
                                    new_hits.append(hit)
                                executed_commands.append({"type": "KEYWORD_SEARCH", "query": ", ".join(keywords), "hits_count": len(keyword_results)})
                                logger.info(f"[LinkedViewSystem] KEYWORD_SEARCH: {keywords}, found {len(keyword_results)} hits")

                # 去重：排除已处理过的 page_id
                new_hits_filtered = []
                for h in new_hits:
                    page_id = h.raw_log_id
                    # 排除已处理过的，只保留真正新的
                    if page_id and page_id not in seen_ids and page_id not in processed_page_ids:
                        new_hits_filtered.append(h)
                        all_hits.append(h)
                        seen_ids.add(page_id)
                    if page_id and h.summary_text:
                        page_summaries.setdefault(page_id, []).append(h.summary_text)

                # Rerank 策略：新 hits + pending_hits 一起 rerank
                # 这样超出 top_k 的 hits 有机会在后续轮次中"上位"
                rerank_pool = new_hits_filtered + pending_hits

                if self.use_reranker and rerank_pool:
                    # 使用 return_all=True 获取所有排序后的 hits（按 page 级别排序）
                    reranked_all = self._rerank_hits(query, rerank_pool, self.reranker_top_k, return_all=True)
                    # 按 page 数量切分：前 top_k 个 pages 给当前轮整合，剩余放入候选池
                    current_hits, pending_hits = self._split_hits_by_page_count(reranked_all, self.reranker_top_k)
                    # 将新处理的 page_id 加入已处理集合
                    new_processed_page_ids = {h.raw_log_id for h in current_hits if h.raw_log_id}
                    processed_page_ids.update(new_processed_page_ids)
                    logger.info(
                        f"[LinkedViewSystem] Rerank: {len(new_hits_filtered)} new + {len(pending_hits) + len(current_hits) - len(new_hits_filtered)} pending "
                        f"-> {len(new_processed_page_ids)} pages ({len(current_hits)} hits) for integration, {len(pending_hits)} pending hits"
                    )
                elif rerank_pool:
                    # 没有 reranker，按 page 数量切分
                    current_hits, pending_hits = self._split_hits_by_page_count(rerank_pool, self.reranker_top_k)
                    new_processed_page_ids = {h.raw_log_id for h in current_hits if h.raw_log_id}
                    processed_page_ids.update(new_processed_page_ids)
                else:
                    # 没有新 hits 也没有 pending，current_hits 清空
                    current_hits = []

                research_iterations.append({
                    "iteration": it,
                    "decision": "SEARCH",
                    "linked_facts": linked_facts,
                    "coverage_assessment": coverage_assessment,
                    "search_commands": executed_commands,
                    "new_hits_count": len(new_hits_filtered),
                })

                logger.info(f"[LinkedViewSystem] Iteration {it}: {len(new_hits_filtered)} new hits, total {len(all_hits)} hits")

                # 如果没有新结果，提前退出
                if not new_hits_filtered:
                    logger.info(f"[LinkedViewSystem] No new hits at iteration {it}, ending research")
                    break

            # 构建 research_summary（从所有 linked_facts + evidence_snippets）
            if all_linked_facts:
                # Part 1: Linked facts
                facts_text = "\n".join([f"- {lf.get('fact', '')}" for lf in all_linked_facts])

                # Part 2: 收集 evidence_snippets（直接从 linked_facts 中获取）
                # evidence_snippets 是 LLM 从原文中提取的完整引用，比整个 page 更精准
                evidence_parts = []
                for lf in all_linked_facts:
                    snippets = lf.get("evidence_snippets", [])
                    source_pages = lf.get("source_pages", [])

                    if snippets:
                        # 将每个 snippet 与其来源关联
                        for i, snippet in enumerate(snippets):
                            if snippet:
                                page_ref = source_pages[i] if i < len(source_pages) else "unknown"
                                # 简化 page_id 显示
                                page_ref_short = page_ref[:8] + "..." if len(page_ref) > 8 else page_ref
                                evidence_parts.append(f"[{page_ref_short}] {snippet}")

                # 组合：Facts + Evidence Snippets
                if evidence_parts:
                    # 去重 evidence_parts
                    unique_evidence = list(dict.fromkeys(evidence_parts))
                    evidence_text = "\n".join(unique_evidence)
                    research_summary = f"## Extracted Facts:\n{facts_text}\n\n## Evidence Snippets:\n{evidence_text}"
                else:
                    research_summary = facts_text
            else:
                # 如果没有提取到任何 linked facts，则退回到简单的 summary 拼接，
                # 确保 R 路径下始终有可用的 research_summary。
                research_summary = "\n".join(
                    f"- {h.summary_text}" for h in all_hits if h.summary_text
                )

            logger.info(f"[LinkedViewSystem] Research summary generated, length={len(research_summary)}")

            # 研究循环结束后，使用智能写入机制写入所有 linked facts（召回比对 + LLM 决策）
            if self.write_facts_to_database and all_linked_facts:
                smart_results = self._smart_write_facts_batch(
                    facts=all_linked_facts,
                    question=query,
                    session_id=session_id,
                    query_idx=query_idx,
                    top_k=5,
                )
                # 统计结果
                add_count = sum(1 for r in smart_results if r.get("action") == "ADD" and r.get("success"))
                update_count = sum(1 for r in smart_results if r.get("action") == "UPDATE" and r.get("success"))
                skip_count = sum(1 for r in smart_results if r.get("action") == "SKIP")
                logger.info(f"[LinkedViewSystem] Smart write: ADD={add_count}, UPDATE={update_count}, SKIP={skip_count} (total {len(all_linked_facts)} facts)")

            # 生成答案：根据 benchmark_name 选择 prompt
            benchmark_name = getattr(self, "benchmark_name", "").lower()
            is_longmemeval = "longmemeval" in benchmark_name
            is_locomo = "locomo" in benchmark_name
            
            if is_longmemeval:
                # LongMemEval：使用专用 prompt
                answer = _answer_with_summary_longmemeval(
                    summary=research_summary,
                    question=query,
                    llm=self.llm_slow,
                )
            elif is_locomo and category is not None:
                # LoCoMo：如果带 category，走短答案 prompt，对齐 GAM 风格
                # category == 3 时使用 _make_locomo_summary_prompt_category3
                answer = _answer_with_summary_locomo(
                    category=category,
                    summary=research_summary,
                    question=query,
                    llm=self.llm_slow,
                )
            else:
                # 默认使用 locomo prompt（向后兼容）
                answer = _answer_with_summary_locomo(
                    category=category,
                    summary=research_summary,
                    question=query,
                    llm=self.llm_slow,
                )
            # 只统计在优化版 R 路径中实际参与处理的 page 数量
            raw_docs_used = len(processed_page_ids)
            logger.info(
                "[LinkedViewSystem] R-path answer generated (GAM-style). "
                f"hits_count={raw_docs_used}, iterations={len(research_iterations)}, query={query}"
            )
        generation_latency_ms = int((time.time() - generation_start) * 1000)

        total_latency_ms = int((time.time() - start) * 1000)
        generation_latency_ms = int((time.time() - generation_start) * 1000)

        # 统计本次 answer 过程中产生的 LLM usage（不包括 router，router 单独统计）
        usage_after_idx = len(self._usage_events)
        new_events = self._usage_events[usage_before_idx:usage_after_idx]
        llm_tokens_in = sum(int(e.get("prompt_tokens", 0)) for e in new_events)
        llm_tokens_out = sum(int(e.get("completion_tokens", 0)) for e in new_events)
        llm_api_calls = len(new_events)

        # online cost：LLM（router+research+answer） + mem0.search（真实 usage）
        # 注意：llm_tokens_in/out 不包括 router（如果使用 ThinkingLLMRouter），router 单独统计
        tokens_in = llm_tokens_in + router_tokens_in + mem0_search_tokens_in
        tokens_out = llm_tokens_out + router_tokens_out + mem0_search_tokens_out
        api_calls = llm_api_calls + router_api_calls + mem0_search_api_calls

        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_api_calls += api_calls

        cost = {
            "online_tokens_in": tokens_in,
            "online_tokens_out": tokens_out,
            "online_retrieval_latency_ms": retrieval_latency_ms,
            "online_generation_latency_ms": generation_latency_ms,
            "online_total_latency_ms": total_latency_ms,
            "online_api_calls": api_calls,
            # LLM tokens（research + answer，不包括 router）
            "llm_prompt_tokens": llm_tokens_in,
            "llm_completion_tokens": llm_tokens_out,
            "llm_api_calls": llm_api_calls,
            # Router tokens（单独统计）
            "router_prompt_tokens": router_tokens_in,
            "router_completion_tokens": router_tokens_out,
            "router_api_calls": router_api_calls,
            "router_latency_ms": router_latency_ms,
            # Mem0 search tokens
            "mem0_search_prompt_tokens": mem0_search_tokens_in,
            "mem0_search_completion_tokens": mem0_search_tokens_out,
            "mem0_search_api_calls": mem0_search_api_calls,
            "mem0_search_latency_ms": mem0_search_latency_ms,
        }

        # 根据路径选择正确的hits变量
        # 优化版 R 路径：使用 all_hits（包含所有处理过的 hits）
        final_hits = hits_for_answer if action == "S" else all_hits

        # 构建 hits_summary：记录每个 hit 的详细信息（用于分析 linked_fact 召回情况）
        hits_summary = []
        linked_fact_hits_count = 0
        original_hits_count = 0
        smart_backfill_hits_count = 0  # 智能回填阶段写入的
        for h in final_hits:
            extra = h.extra if hasattr(h, 'extra') else {}
            source_type = extra.get("source_type", "unknown")
            flush_phase = extra.get("flush_phase", "")  # smart_backfill 或空
            smart_action = extra.get("smart_action", "")  # ADD, UPDATE 或空

            # 统计不同来源的 hits
            if flush_phase == "smart_backfill":
                smart_backfill_hits_count += 1
            elif source_type == "linked_fact":
                linked_fact_hits_count += 1
            elif source_type == "original" or source_type == "unknown":
                original_hits_count += 1

            hits_summary.append({
                "score": h.score,
                "source_type": source_type,
                "flush_phase": flush_phase,  # 区分是否是智能回填写入
                "smart_action": smart_action,  # 智能回填时的操作类型
                "page_id": extra.get("page_id", ""),
                "memory_id": h.memory_id if hasattr(h, 'memory_id') else "",
                "memory_preview": (h.summary_text[:100] + "...") if hasattr(h, 'summary_text') and h.summary_text and len(h.summary_text) > 100 else (h.summary_text if hasattr(h, 'summary_text') else ""),
            })

        mechanism_trace: Dict[str, Any] = {
            "route": action,
            "final_action": action,  # 最终决策（与 route 相同，但更明确）
            "hit_count": len(final_hits),
            "scores": [h.score for h in final_hits],
            "raw_docs_used": raw_docs_used,

            # Linked Fact 召回统计（用于论文分析）
            "linked_fact_hits": linked_fact_hits_count,  # 原始 benchmark 运行时写入的
            "original_hits": original_hits_count,        # observe 阶段写入的
            "smart_backfill_hits": smart_backfill_hits_count,  # 智能回填阶段写入的
            "hits_summary": hits_summary,  # 每个 hit 的详细信息（含 source_type, flush_phase, smart_action）

            # Router 决策全过程
            "router_history": retrieval_history,  # 包含所有轮次的 query、action、hits_count 等
            "total_retrieval_rounds": len(retrieval_history),  # 总检索轮次

            # Router 思维链（如果有）
            "router_thinking": getattr(self.router, "last_raw_response", "") if hasattr(self, "router") else "",
        }
        
        # R路径下添加研究迭代信息（GAM风格）
        if action == "R":
            mechanism_trace["research_iterations"] = research_iterations
            mechanism_trace["summary_block"] = research_summary if research_summary else ""
        if action == "S":
            mechanism_trace["summary_block"] = summary_block if summary_block else ""
        return AnswerResult(answer=answer, cost_metrics=cost, mechanism_trace=mechanism_trace)

    # === Reranker 相关方法 ===
    # 类级别共享的 reranker 实例（用于多线程场景）
    _shared_reranker = None

    def _get_reranker(self):
        """
        懒加载 Reranker 实例（线程安全）。

        使用类级别的共享实例，避免多个 LinkedViewSystem 实例各自加载 reranker。
        使用模块级 _RERANKER_INIT_LOCK 保护加载过程。
        """
        # 快速检查：如果已经禁用 reranker，直接返回
        if not self.use_reranker:
            return None

        # 如果实例已经有 reranker，直接返回
        if self._reranker is not None:
            return self._reranker

        # 检查共享实例（无锁快速路径）
        if LinkedViewSystem._shared_reranker is not None:
            self._reranker = LinkedViewSystem._shared_reranker
            return self._reranker

        # 使用模块级锁保护加载过程
        with _RERANKER_INIT_LOCK:
            # 双重检查：另一个线程可能已经加载了
            if LinkedViewSystem._shared_reranker is not None:
                self._reranker = LinkedViewSystem._shared_reranker
                return self._reranker

            try:
                logger.info(f"[LinkedViewSystem] Loading reranker (thread-safe) from {self.reranker_model_path}")
                from src.linked_view.reranker import Qwen3Reranker
                reranker = Qwen3Reranker(
                    model_name_or_path=self.reranker_model_path,
                    max_length=2048,
                )
                # 存储到共享实例和当前实例
                LinkedViewSystem._shared_reranker = reranker
                self._reranker = reranker
                logger.info(f"[LinkedViewSystem] Reranker loaded successfully")
            except Exception as e:
                import traceback
                logger.error(f"[LinkedViewSystem] Failed to load reranker: {e}")
                logger.error(f"[LinkedViewSystem] Reranker traceback:\n{traceback.format_exc()}")
                logger.warning("[LinkedViewSystem] Disabling reranker - will use original retrieval order")
                self.use_reranker = False
                self._reranker = None

        return self._reranker

    def _rerank_hits(
        self,
        query: str,
        hits: List[SummaryHit],
        top_k: Optional[int] = None,
        protect_threshold: float = 0.6,
        return_all: bool = False,
    ) -> List[SummaryHit]:
        """
        对 hits 进行 rerank，返回最相关的结果。

        **重要**: top_k 按 **unique page** 计数，不是按 hit 计数。
        同一个 page_id 的多个 summaries 会被合并，算作一个 page。

        分层策略：
        1. 按 page_id 分组，同一 page 的多个 hits 合并
        2. page 级别判断保护：如果 page 的任意 hit 是 mem0 且分数 >= threshold，保护该 page
        3. 对候选 pages 做 rerank（使用 page content）
        4. 选择 top_k 个 pages，返回这些 pages 的所有 hits

        Args:
            query: 用户问题
            hits: SummaryHit 列表
            top_k: rerank 后保留的最大 **page 数量**（默认使用 self.reranker_top_k）
            protect_threshold: 保护阈值，mem0 分数 >= 此值的 page 不参与淘汰
            return_all: 如果为 True，返回所有 pages 的 hits（只排序，不截断）

        Returns:
            Reranked 后的 hits 列表（包含选中 pages 的所有 hits）
        """
        if not self.use_reranker or not hits:
            return hits

        top_k = top_k or self.reranker_top_k

        # 1. 按 page_id 分组
        # page_groups: {page_id: {"hits": [hit, ...], "best_score": float, "is_protected": bool, "content": str}}
        page_groups: Dict[str, Dict[str, Any]] = {}
        no_page_id_hits: List[SummaryHit] = []  # 没有 page_id 的 hits 单独处理

        for hit in hits:
            page_id = hit.raw_log_id
            if not page_id:
                no_page_id_hits.append(hit)
                continue

            source = hit.extra.get("source", "mem0") if hit.extra else "mem0"
            is_mem0 = not source.startswith("bm25")

            if page_id not in page_groups:
                # 获取 page content
                raw_content = ""
                if self.page_store:
                    try:
                        page = self.page_store.get_page_by_id(page_id)
                        if page:
                            raw_content = page.content or ""
                    except Exception:
                        pass

                page_groups[page_id] = {
                    "hits": [],
                    "best_score": 0.0,
                    "is_protected": False,
                    "content": raw_content,
                }

            page_groups[page_id]["hits"].append(hit)
            # 更新 best_score
            if hit.score > page_groups[page_id]["best_score"]:
                page_groups[page_id]["best_score"] = hit.score
            # 如果任意 hit 满足保护条件，整个 page 被保护
            if is_mem0 and hit.score >= protect_threshold:
                page_groups[page_id]["is_protected"] = True

        # 2. 分离保护的 pages 和候选 pages
        protected_page_ids: List[str] = []
        candidate_page_ids: List[str] = []

        for page_id, group in page_groups.items():
            if group["is_protected"]:
                protected_page_ids.append(page_id)
            else:
                candidate_page_ids.append(page_id)

        logger.info(
            f"[LinkedViewSystem] Rerank (page-level): {len(protected_page_ids)} protected pages, "
            f"{len(candidate_page_ids)} candidate pages (from {len(hits)} hits)"
        )

        # 3. 如果候选为空，直接返回保护的 pages 的所有 hits
        if not candidate_page_ids:
            result_hits = []
            for page_id in protected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            result_hits.extend(no_page_id_hits)
            logger.info(f"[LinkedViewSystem] No candidate pages, returning {len(result_hits)} hits from {len(protected_page_ids)} protected pages")
            return result_hits

        # 计算候选池需要保留多少 pages
        remaining_page_slots = max(0, top_k - len(protected_page_ids))

        if len(candidate_page_ids) <= remaining_page_slots:
            # 候选 pages 数量不足，全部返回
            result_hits = []
            for page_id in protected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            for page_id in candidate_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            result_hits.extend(no_page_id_hits)
            logger.info(f"[LinkedViewSystem] Only {len(candidate_page_ids)} candidate pages, returning all {len(result_hits)} hits")
            return result_hits

        reranker = self._get_reranker()
        if reranker is None:
            # 没有 reranker，按 best_score 排序
            candidate_page_ids.sort(key=lambda pid: page_groups[pid]["best_score"], reverse=True)
            selected_page_ids = candidate_page_ids if return_all else candidate_page_ids[:remaining_page_slots]
            result_hits = []
            for page_id in protected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            for page_id in selected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            result_hits.extend(no_page_id_hits)
            return result_hits

        try:
            # 4. 使用 page content 做 rerank（每个 page 只参与一次 rerank）
            # 创建临时的 page 代表 hit（用于 reranker）
            page_representatives = []
            for page_id in candidate_page_ids:
                group = page_groups[page_id]
                # 使用第一个 hit 作为代表，但用 page content 做 rerank
                representative = group["hits"][0]
                content = group["content"]
                if not content:
                    # fallback: 合并所有 summaries
                    content = "\n".join(h.summary_text or "" for h in group["hits"])
                representative._rerank_text = content
                representative._page_id = page_id  # 标记 page_id
                page_representatives.append(representative)

            # Rerank pages
            rerank_k = len(candidate_page_ids) if return_all else remaining_page_slots
            reranked_representatives = reranker.rerank(
                query=query,
                hits=page_representatives,
                top_k=rerank_k,
                text_field="_rerank_text",
                force_score=return_all,
            )

            # 提取 reranked page_ids
            reranked_page_ids = [getattr(rep, "_page_id", rep.raw_log_id) for rep in reranked_representatives]

            # 清理临时属性
            for rep in page_representatives:
                if hasattr(rep, '_rerank_text'):
                    delattr(rep, '_rerank_text')
                if hasattr(rep, '_page_id'):
                    delattr(rep, '_page_id')

            # 5. 收集结果：protected pages + reranked pages 的所有 hits
            result_hits = []
            selected_page_ids = []

            # 先加入 protected pages
            for page_id in protected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
                selected_page_ids.append(page_id)

            # 再加入 reranked pages
            for page_id in reranked_page_ids:
                if page_id and page_id in page_groups:
                    result_hits.extend(page_groups[page_id]["hits"])
                    selected_page_ids.append(page_id)

            # 加入没有 page_id 的 hits
            result_hits.extend(no_page_id_hits)

            logger.info(
                f"[LinkedViewSystem] Rerank result (page-level): {len(protected_page_ids)} protected + "
                f"{len(reranked_page_ids)} reranked = {len(selected_page_ids)} pages, {len(result_hits)} total hits"
            )
            return result_hits

        except Exception as e:
            logger.error(f"[LinkedViewSystem] Rerank failed: {e}")
            # fallback: 按 best_score 排序
            candidate_page_ids.sort(key=lambda pid: page_groups[pid]["best_score"], reverse=True)
            selected_page_ids = candidate_page_ids if return_all else candidate_page_ids[:remaining_page_slots]
            result_hits = []
            for page_id in protected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            for page_id in selected_page_ids:
                result_hits.extend(page_groups[page_id]["hits"])
            result_hits.extend(no_page_id_hits)
            return result_hits

    def _split_hits_by_page_count(
        self,
        hits: List[SummaryHit],
        top_k_pages: int,
    ) -> Tuple[List[SummaryHit], List[SummaryHit]]:
        """
        按 unique page 数量切分 hits。

        将 hits 分成两部分：
        1. 前 top_k_pages 个 unique pages 的所有 hits
        2. 剩余 pages 的所有 hits

        Args:
            hits: SummaryHit 列表（应该已经按相关性排序）
            top_k_pages: 要保留的 unique page 数量

        Returns:
            (selected_hits, remaining_hits) - 选中的 hits 和剩余的 hits
        """
        if not hits:
            return [], []

        selected_hits: List[SummaryHit] = []
        remaining_hits: List[SummaryHit] = []
        seen_page_ids: Set[str] = set()
        selected_page_ids: Set[str] = set()

        # 第一遍：确定哪些 page_ids 被选中（按 hits 出现顺序，保持 rerank 的排序）
        for hit in hits:
            page_id = hit.raw_log_id
            if not page_id:
                # 没有 page_id 的 hit 归入 selected
                continue

            if page_id not in seen_page_ids:
                seen_page_ids.add(page_id)
                if len(selected_page_ids) < top_k_pages:
                    selected_page_ids.add(page_id)

        # 第二遍：根据 page_id 分配 hits
        for hit in hits:
            page_id = hit.raw_log_id
            if not page_id:
                # 没有 page_id 的 hit 归入 selected
                selected_hits.append(hit)
            elif page_id in selected_page_ids:
                selected_hits.append(hit)
            else:
                remaining_hits.append(hit)

        logger.debug(
            f"[LinkedViewSystem] _split_hits_by_page_count: "
            f"{len(selected_page_ids)} pages selected, "
            f"{len(selected_hits)} selected hits, {len(remaining_hits)} remaining hits"
        )
        return selected_hits, remaining_hits

    # === JSON 修复工具方法 ===
    def _repair_and_parse_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        尝试修复并解析可能格式错误的 JSON 字符串。

        处理的常见问题：
        1. 字符串内的未转义换行符
        2. 字符串内的未转义引号
        3. 末尾多余的逗号
        4. 控制字符

        Args:
            json_str: 可能格式错误的 JSON 字符串

        Returns:
            解析后的字典，如果所有修复尝试都失败则返回 None
        """
        # 策略 1: 移除末尾多余的逗号
        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 策略 2: 修复字符串内的换行符和控制字符
        # 匹配 JSON 字符串（非贪婪，处理转义引号）
        # 这个正则比较简单，可能会有边界情况
        try:
            # 先尝试处理字符串内的换行
            lines = fixed.split('\n')
            in_string = False
            repaired_lines = []

            for line in lines:
                # 简单的启发式：如果行以未闭合的引号结束，则在换行处添加 \n
                quote_count = line.count('"') - line.count('\\"')
                if in_string:
                    # 在字符串内，这个换行需要转义
                    repaired_lines[-1] += '\\n' + line
                else:
                    repaired_lines.append(line)

                # 更新 in_string 状态
                if quote_count % 2 == 1:
                    in_string = not in_string

            fixed = '\n'.join(repaired_lines)
            result = json.loads(fixed)
            logger.info("[LinkedViewSystem] JSON repaired successfully using newline escape strategy")
            return result
        except (json.JSONDecodeError, Exception):
            pass

        # 策略 3: 尝试提取 linked_facts 数组中的各个对象
        try:
            facts = []
            # 匹配 linked_facts 数组内容
            facts_match = re.search(r'"linked_facts"\s*:\s*\[([\s\S]*?)\]\s*[,}]', json_str)
            if facts_match:
                facts_content = facts_match.group(1)
                # 尝试匹配每个 fact 对象
                fact_pattern = r'\{\s*"fact"\s*:\s*"([^"]*(?:\\"[^"]*)*)"\s*,\s*"source_pages"\s*:\s*\[([^\]]*)\]\s*,\s*"evidence_snippets"\s*:\s*\[([^\]]*)\]\s*\}'
                for fact_match in re.finditer(fact_pattern, facts_content):
                    try:
                        fact_text = fact_match.group(1).replace('\\"', '"')
                        source_pages_str = fact_match.group(2)
                        # 提取 source_pages
                        source_pages = [s.strip().strip('"') for s in source_pages_str.split(',') if s.strip()]
                        # evidence_snippets 可能很长，简化处理
                        facts.append({
                            "fact": fact_text,
                            "source_pages": source_pages,
                            "evidence_snippets": []  # 简化，不提取复杂的 snippets
                        })
                    except Exception:
                        continue

                if facts:
                    logger.info(f"[LinkedViewSystem] JSON repair: extracted {len(facts)} facts using regex")
                    return {"linked_facts": facts, "coverage_assessment": "Partial extraction due to JSON error"}
        except Exception:
            pass

        # 策略 4: 最后尝试 - 使用更宽松的解析
        try:
            # 移除所有控制字符
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
            # 再次尝试移除末尾逗号
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            result = json_repair.loads(cleaned)
            logger.info("[LinkedViewSystem] JSON repaired successfully using control char removal")
            return result
        except json.JSONDecodeError:
            pass

        # 所有策略都失败
        logger.warning("[LinkedViewSystem] All JSON repair strategies failed, returning empty linked_facts")
        return {"linked_facts": [], "coverage_assessment": "JSON parse error, all repair strategies failed"}

    # === 优化版 R 路径 V2：Integration 方法 ===
    def _research_integration_v2(
        self,
        query: str,
        hits: List[SummaryHit],
        _page_summaries: Dict[str, List[str]],  # 未使用，保留兼容性
    ) -> Dict[str, Any]:
        """
        整合证据并提取 linked facts。

        V2 改进：模型只负责提取 fact + quote，source_pages 通过后处理匹配。
        这样更可靠，避免模型在生成 source_page 时产生幻觉。

        Args:
            query: 用户问题
            hits: 检索到的 hits
            _page_summaries: page_id -> summaries 的映射（未使用，保留兼容性）

        Returns:
            {
                "linked_facts": [
                    {
                        "fact": str,
                        "source_pages": [page_id, ...],
                        "evidence_snippets": [str, ...]
                    },
                    ...
                ],
                "coverage_assessment": str
            }
        """
        # 确保 session 数据已加载
        session_id = self.current_session_id
        logger.info(f"[LinkedViewSystem] _research_integration_v2: session_id={session_id}, page_store={self.page_store is not None}")
        if self.page_store and session_id:
            self.page_store._load_session(session_id)
            pages_count = len(self.page_store.get_pages_by_session(session_id))
            logger.info(f"[LinkedViewSystem] Loaded {pages_count} pages for session {session_id}")

        # 按 page_id 分组，合并同一个 page 的多个 summaries
        # 结构: {page_id: {"summaries": [str, ...], "source": str, "raw_content": str}}
        page_info: Dict[str, Dict[str, Any]] = {}

        for hit in hits:
            page_id = hit.raw_log_id or "unknown"
            source = hit.extra.get("source", "mem0") if hit.extra else "mem0"
            summary = hit.summary_text or ""

            # 判断是否是真正的 mem0 summary（不是 BM25 的 fallback）
            # BM25 召回时 source 会是 "bm25_query", "bm25_keywords", "bm25_initial" 等
            is_real_mem0_summary = not source.startswith("bm25")

            if page_id not in page_info:
                # 获取原始内容（只获取一次）
                raw_content = ""
                if self.page_store and page_id != "unknown":
                    try:
                        page = self.page_store.get_page_by_id(page_id)
                        if page:
                            raw_content = page.content
                            logger.debug(f"[LinkedViewSystem] Got raw content for page {page_id}: {len(raw_content)} chars")
                        else:
                            logger.warning(f"[LinkedViewSystem] Page not found in page_store: {page_id}")
                    except Exception as e:
                        logger.warning(f"[LinkedViewSystem] Failed to get raw content for page {page_id}: {e}")

                page_info[page_id] = {
                    "summaries": [],  # 只存真正的 mem0 summary
                    "source": source,
                    "raw_content": raw_content,
                }

            # 只添加真正的 mem0 summary（不是 BM25 的 fallback content）
            if summary and is_real_mem0_summary and summary not in page_info[page_id]["summaries"]:
                page_info[page_id]["summaries"].append(summary)

            # 更新 source（优先保留 mem0，因为它有真正的 summary）
            if source == "mem0" and page_info[page_id]["source"].startswith("bm25"):
                page_info[page_id]["source"] = "mem0"

        # 构建证据文本（每个 page 只出现一次）
        evidence_parts = []

        for page_id, info in page_info.items():
            source = info["source"]
            raw_content = info["raw_content"]
            summaries = info["summaries"]

            if source.startswith("bm25") and not summaries:
                # BM25 召回且没有 summary：只有原始内容
                evidence_parts.append(f"[Page ID: {page_id}] (BM25)\nRaw Content:\n{raw_content}\n")
            else:
                # mem0 召回或有 summary：合并所有 summaries + 原始内容
                if summaries:
                    summaries_text = "\n".join(f"  - {s}" for s in summaries)
                    evidence_parts.append(
                        f"[Page ID: {page_id}] (mem0)\n"
                        f"Mem0 Summaries:\n{summaries_text}\n"
                        f"Raw Content:\n{raw_content}\n"
                    )
                else:
                    evidence_parts.append(f"[Page ID: {page_id}] (mem0)\nRaw Content:\n{raw_content}\n")

        evidence_text = "\n---\n".join(evidence_parts) if evidence_parts else "No evidence retrieved."

        prompt = RESEARCH_INTEGRATION_V2_PROMPT.format(
            question=query,
            evidence=evidence_text,
        )
        logger.info(f"[LinkedViewSystem] Research integration v2 prompt: {prompt}")
        try:
            response = self.llm_slow.generate(prompt)
            response = (response or "").strip()
            logger.info(f"[LinkedViewSystem] Research integration v2 raw response: {response}")

            # 解析 JSON（增强错误处理）
            result = None
            json_str = None

            # 尝试提取 JSON
            if response.startswith("{"):
                json_str = response
            else:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group()

            if json_str:
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"[LinkedViewSystem] JSON parse failed: {e}, trying to fix...")
                    result = self._repair_and_parse_json(json_str)

            if result is None:
                logger.warning(f"[LinkedViewSystem] Failed to parse integration_v2 response: {response[:200]}")
                result = {"extracted_facts": [], "coverage_assessment": "Failed to parse response"}

            # 适配新的 prompt 格式（linked_facts 保持不变，但移除了 source_pages）
            extracted_facts = result.get("linked_facts", result.get("extracted_facts", []))

            # 后处理：通过 evidence_quote 匹配链接到 pages
            linked_facts = self._link_facts_to_pages(extracted_facts, page_info)

            result = {
                "linked_facts": linked_facts,
                "coverage_assessment": result.get("coverage_assessment", ""),
            }
            logger.info(f"[LinkedViewSystem] Research integration v2 result: {result}")
            return result

        except Exception as e:
            logger.error(f"[LinkedViewSystem] _research_integration_v2 failed: {e}")
            return {"linked_facts": [], "coverage_assessment": f"Error: {e}"}

    def _link_facts_to_pages(
        self,
        extracted_facts: List[Dict[str, Any]],
        page_info: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        通过 evidence_quote 匹配将 facts 链接到 source pages（后处理）。

        这比让 LLM 直接生成 source_pages 更可靠，因为：
        1. 字符串匹配是确定性的，不会产生幻觉
        2. 可以处理模型提取的 quote 略有不同的情况（子串匹配）
        3. 即使模型没有提供 quote，也可以通过 fact 内容模糊匹配

        Args:
            extracted_facts: LLM 返回的 facts，格式为 [{"fact": str, "evidence_quote": str}, ...]
            page_info: page_id -> {"summaries": [...], "raw_content": str} 的映射

        Returns:
            linked_facts: [{"fact": str, "source_pages": [...], "evidence_snippets": [...]}, ...]
        """
        linked_facts = []

        for fact_item in extracted_facts:
            fact_text = fact_item.get("fact", "")
            # 兼容多种字段名
            quote = fact_item.get("evidence_quote", fact_item.get("quote", fact_item.get("evidence_snippet", "")))

            if not fact_text:
                continue

            source_pages = []
            evidence_snippets = [quote] if quote else []

            # 策略 1：通过 quote 精确匹配（优先）
            if quote:
                # 标准化 quote（移除多余空白）
                quote_normalized = " ".join(quote.split())

                for page_id, info in page_info.items():
                    if page_id == "unknown":
                        continue

                    raw_content = info.get("raw_content", "")
                    content_normalized = " ".join(raw_content.split())

                    # 尝试精确子串匹配
                    if quote_normalized in content_normalized:
                        source_pages.append(page_id)
                        logger.debug(f"[LinkedViewSystem] Quote matched in page {page_id}: {quote[:50]}...")
                        continue

                    # 尝试 summary 匹配
                    for summary in info.get("summaries", []):
                        summary_normalized = " ".join(summary.split())
                        if quote_normalized in summary_normalized:
                            source_pages.append(page_id)
                            logger.debug(f"[LinkedViewSystem] Quote matched in summary of page {page_id}")
                            break

            # 策略 2：如果 quote 匹配失败，尝试通过 fact 中的关键词模糊匹配
            if not source_pages:
                # 从 fact 中提取关键词（去除常见词）
                fact_words = set(fact_text.lower().split())
                stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "that", "this"}
                keywords = [w for w in fact_words if len(w) > 3 and w not in stop_words]

                if keywords:
                    best_match_page = None
                    best_match_count = 0

                    for page_id, info in page_info.items():
                        if page_id == "unknown":
                            continue

                        raw_content = info.get("raw_content", "").lower()
                        match_count = sum(1 for kw in keywords if kw in raw_content)

                        if match_count > best_match_count:
                            best_match_count = match_count
                            best_match_page = page_id

                    # 如果超过一半的关键词匹配，认为是有效匹配
                    if best_match_page and best_match_count >= len(keywords) / 2:
                        source_pages.append(best_match_page)
                        logger.debug(f"[LinkedViewSystem] Keyword fuzzy match: page {best_match_page}, matched {best_match_count}/{len(keywords)} keywords")

            linked_facts.append({
                "fact": fact_text,
                "source_pages": source_pages,
                "evidence_snippets": evidence_snippets,
            })

        # 统计匹配情况
        matched_count = sum(1 for f in linked_facts if f["source_pages"])
        total_count = len(linked_facts)
        logger.info(f"[LinkedViewSystem] _link_facts_to_pages: {matched_count}/{total_count} facts linked to pages")

        return linked_facts

    # === 优化版 R 路径 V2：Plan 方法 ===
    def _research_plan_v2(
        self,
        query: str,
        current_facts: str,
        coverage_assessment: str,
        research_history: List[Dict[str, Any]],
        searched_queries: Set[str],
        searched_keywords: Set[Tuple[str, ...]],
    ) -> Dict[str, Any]:
        """
        基于当前 facts 决定是否需要更多搜索。

        Args:
            query: 用户问题
            current_facts: 当前整合的 facts 文本
            coverage_assessment: 覆盖度评估
            research_history: 研究历史
            searched_queries: 已搜索的 MEM0 查询
            searched_keywords: 已搜索的关键词集合

        Returns:
            {
                "decision": "DONE" | "SEARCH",
                "reasoning": str,
                "search_commands": [{"type": str, "query"/"keywords": ...}, ...]
            }
        """
        # 构建研究历史文本
        history_parts = []
        for hist in research_history:
            it = hist.get("iteration", 0)
            searches = hist.get("search_commands", [])
            new_hits = hist.get("new_hits_count", 0)
            search_str = "; ".join([f"{s.get('type')}: {s.get('query', s.get('keywords', ''))}" for s in searches])
            history_parts.append(f"Iteration {it}: {search_str} -> {new_hits} new hits")
        research_history_text = "\n".join(history_parts) if history_parts else "First iteration"

        # 构建已搜索查询列表
        searched_list = []
        if searched_queries:
            searched_list.append("MEM0_SEARCH: " + ", ".join(sorted(searched_queries)[:10]))
        if searched_keywords:
            kw_strs = [", ".join(kw) for kw in list(searched_keywords)[:5]]
            searched_list.append("KEYWORD_SEARCH: " + "; ".join(kw_strs))
        searched_queries_text = "\n".join(searched_list) if searched_list else "None"

        prompt = RESEARCH_PLAN_V2_PROMPT.format(
            question=query,
            current_facts=current_facts,
            coverage_assessment=coverage_assessment,
            research_history=research_history_text,
            searched_queries=searched_queries_text,
        )
        logger.info(f"[LinkedViewSystem] Research plan v2 prompt: {prompt}")
        try:
            response = self.llm_slow.generate(prompt)
            response = (response or "").strip()
            logger.info(f"[LinkedViewSystem] Research plan v2 result: {response}")
            # 解析 JSON（使用与 integration_v2 相同的鲁棒解析）
            result = None
            json_str = None

            if response.startswith("{"):
                json_str = response
            else:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group()

            if json_str:
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"[LinkedViewSystem] Plan v2 JSON parse failed: {e}, trying to fix...")
                    result = self._repair_and_parse_json(json_str)

            if result is None:
                logger.warning(f"[LinkedViewSystem] Failed to parse plan_v2 response: {response[:200]}")
                result = {"decision": "DONE", "reasoning": "Failed to parse response", "search_commands": []}

            result.setdefault("decision", "DONE")
            result.setdefault("reasoning", "")
            result.setdefault("search_commands", [])

            return result

        except Exception as e:
            logger.error(f"[LinkedViewSystem] _research_plan_v2 failed: {e}")
            return {"decision": "DONE", "reasoning": f"Error: {e}", "search_commands": []}

    # === 统计追踪方法 ===
    def _init_session_stats(self, session_id: str, load_existing: bool = False) -> None:
        """
        初始化 session 级别的统计追踪数据结构。

        Args:
            session_id: 会话 ID
            load_existing: 是否尝试加载已有的统计文件（用于 load() 恢复场景）
        """
        if not self.enable_stats_tracking:
            self._session_stats = None
            return

        stats_file = Path(self.stats_dir) / f"{session_id}_stats.json"

        # 尝试加载已有统计
        if load_existing and stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    self._session_stats = json.load(f)
                logger.info(f"[LinkedViewSystem] Loaded existing stats for session {session_id}")
                return
            except Exception as e:
                logger.warning(f"[LinkedViewSystem] Failed to load stats file: {e}")

        # 创建新的统计结构
        self._session_stats = {
            "session_id": session_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_queries": 0,
                "s_path_count": 0,
                "r_path_count": 0,
                "total_linked_facts_written": 0,
                "total_linked_facts_retrieved": 0,
                "s_with_linked_facts": 0,  # 走 S 路径且召回了 linked_fact 的次数
                "r_with_linked_facts": 0,  # 走 R 路径且召回了 linked_fact 的次数
            },
            "write_events": [],  # 写回事件记录
            "query_events": [],  # 查询事件记录
        }

        # 确保目录存在
        Path(self.stats_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"[LinkedViewSystem] Initialized stats tracking for session {session_id}")

    def _record_write_event(
        self,
        session_id: str,
        query_idx: int,
        facts_written: int,
        linked_facts: List[Dict[str, Any]],
    ) -> None:
        """
        记录 linked_fact 写回事件。

        Args:
            session_id: 会话 ID
            query_idx: 当前是第几个 query（从 0 开始）
            facts_written: 成功写入的 fact 数量
            linked_facts: 写入的 linked_facts 列表
        """
        if not self.enable_stats_tracking or self._session_stats is None:
            return

        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_idx": query_idx,
            "facts_written": facts_written,
            "facts": [
                {
                    "fact": lf.get("fact", "")[:200],  # 截断，避免太长
                    "source_pages": lf.get("source_pages", []),
                }
                for lf in linked_facts[:10]  # 最多记录 10 条
            ]
        }
        self._session_stats["write_events"].append(event)
        self._session_stats["summary"]["total_linked_facts_written"] += facts_written

        # 保存统计
        self._save_session_stats()
        logger.debug(f"[LinkedViewSystem] Recorded write event: query_idx={query_idx}, facts_written={facts_written}")

    def _record_query_event(
        self,
        session_id: str,
        query_idx: int,
        query: str,
        action: str,
        total_hits: int,
        linked_fact_hits: int,
        original_hits: int,
        linked_fact_hit_details: List[Dict[str, Any]],
        router_confidence: Optional[float] = None,
    ) -> None:
        """
        记录查询事件（包括路由决策和召回统计）。

        Args:
            session_id: 会话 ID
            query_idx: 当前是第几个 query（从 0 开始）
            query: 查询文本
            action: 路由决策 ("S" / "R")
            total_hits: 总召回数
            linked_fact_hits: 来自 linked_fact 的召回数（QA 阶段写回的）
            original_hits: 来自原始记忆的召回数（observe 阶段写入的）
            linked_fact_hit_details: linked_fact 召回的详细信息
            router_confidence: router 置信度（如果可用）
        """
        if not self.enable_stats_tracking or self._session_stats is None:
            return

        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_idx": query_idx,
            "query": query[:200],  # 截断
            "action": action,
            "total_hits": total_hits,
            "original_hits": original_hits,  # 原始记忆召回数
            "linked_fact_hits": linked_fact_hits,  # QA 阶段写回的 fact 召回数
            "linked_fact_ratio": linked_fact_hits / total_hits if total_hits > 0 else 0.0,
            "router_confidence": router_confidence,
            "linked_fact_details": linked_fact_hit_details[:5],  # 最多记录 5 条详情
        }
        self._session_stats["query_events"].append(event)

        # 更新 summary
        summary = self._session_stats["summary"]
        summary["total_queries"] += 1
        summary["total_linked_facts_retrieved"] += linked_fact_hits

        if action == "S":
            summary["s_path_count"] += 1
            if linked_fact_hits > 0:
                summary["s_with_linked_facts"] += 1
        elif action == "R":
            summary["r_path_count"] += 1
            if linked_fact_hits > 0:
                summary["r_with_linked_facts"] += 1

        # 保存统计
        self._save_session_stats()

        # 记录到日志
        logger.info(
            f"[LinkedViewSystem][STATS] query_idx={query_idx}, action={action}, "
            f"total_hits={total_hits}, linked_fact_hits={linked_fact_hits}, "
            f"s_count={summary['s_path_count']}, r_count={summary['r_path_count']}"
        )

    def _count_linked_fact_hits(
        self,
        hits: List[SummaryHit],
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """
        统计召回结果中有多少来自 linked_fact vs 原始记忆。

        Args:
            hits: SummaryHit 列表

        Returns:
            (linked_fact_count, original_count, linked_fact_details)
        """
        linked_fact_count = 0
        original_count = 0
        unknown_count = 0
        linked_fact_details = []

        for hit in hits:
            # 检查 extra 中的 source_type
            source_type = hit.extra.get("source_type", "")
            if source_type == "linked_fact":
                linked_fact_count += 1
                linked_fact_details.append({
                    "summary_text": hit.summary_text[:100] if hit.summary_text else "",
                    "score": hit.score,
                    "source_pages": hit.extra.get("source_pages", []),
                    "raw_log_id": hit.raw_log_id,
                    "written_at": hit.extra.get("written_at", ""),
                    "written_at_query_idx": hit.extra.get("written_at_query_idx", -1),
                })
            elif source_type == "original":
                original_count += 1
            else:
                # 旧数据可能没有 source_type，视为 original
                original_count += 1
                unknown_count += 1

        # 记录详细统计到日志
        if unknown_count > 0:
            logger.debug(
                f"[LinkedViewSystem] Hit source breakdown: "
                f"linked_fact={linked_fact_count}, original={original_count} (unknown={unknown_count})"
            )

        return linked_fact_count, original_count, linked_fact_details

    def _save_session_stats(self) -> None:
        """保存当前 session 的统计到文件。"""
        if not self.enable_stats_tracking or self._session_stats is None:
            return

        session_id = self._session_stats.get("session_id", "unknown")
        stats_file = Path(self.stats_dir) / f"{session_id}_stats.json"

        try:
            # 更新最后修改时间
            self._session_stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self._session_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[LinkedViewSystem] Failed to save stats: {e}")

    def get_session_stats(self) -> Optional[Dict[str, Any]]:
        """获取当前 session 的统计数据。"""
        return self._session_stats

    # === Linked Facts 记录与写入 ===

    def _record_linked_facts(
        self,
        question: str,
        linked_facts: List[Dict[str, Any]],
        raw_hits: List[Any],
        session_id: str,
        query_idx: int = 0,
    ) -> None:
        """
        记录 linked facts 及其上下文信息（用于后续分析和延迟写入）。

        Args:
            question: 当前查询问题
            linked_facts: 提取的 linked facts 列表
            raw_hits: 召回的原始数据 hits
            session_id: 会话 ID
            query_idx: 当前 query 索引
        """
        if not linked_facts:
            return

        # 构建 raw_hits 摘要（用于分析，避免记录过多数据）
        raw_hits_summary = []
        for hit in raw_hits[:10]:  # 最多记录前 10 个
            if hasattr(hit, 'summary_text'):
                raw_hits_summary.append({
                    "text": hit.summary_text[:500] if hit.summary_text else "",
                    "score": getattr(hit, 'score', 0.0),
                    "raw_log_id": getattr(hit, 'raw_log_id', ""),
                })
            elif isinstance(hit, dict):
                raw_hits_summary.append({
                    "text": str(hit.get("text", hit.get("content", "")))[:500],
                    "score": hit.get("score", 0.0),
                    "raw_log_id": hit.get("raw_log_id", hit.get("page_id", "")),
                })

        record = {
            "query_idx": query_idx,
            "question": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "linked_facts": linked_facts,  # 完整的 linked facts
            "raw_hits_count": len(raw_hits),
            "raw_hits_summary": raw_hits_summary,  # 召回数据摘要
        }

        self._pending_linked_facts.append(record)
        logger.info(
            f"[LinkedViewSystem] Recorded {len(linked_facts)} linked facts for Q{query_idx}: {question[:50]}..."
        )

    def get_pending_linked_facts(self) -> List[Dict[str, Any]]:
        """
        获取所有待写入的 linked facts 记录。

        Returns:
            记录列表，每条记录包含：
            - query_idx: 问题索引
            - question: 问题文本
            - timestamp: 记录时间
            - session_id: 会话 ID
            - linked_facts: 提取的 facts 列表
            - raw_hits_count: 召回的原始数据数量
            - raw_hits_summary: 召回数据摘要
        """
        return self._pending_linked_facts.copy()

    def save_pending_linked_facts(self, output_path: str) -> str:
        """
        将待写入的 linked facts 保存到 JSON 文件。

        Args:
            output_path: 输出目录路径或文件路径
                - 如果是目录：生成 {session_id}_linked_facts.json
                - 如果是文件路径：直接使用

        Returns:
            实际保存的文件路径，如果没有数据则返回空字符串
        """
        import json
        from pathlib import Path

        if not self._pending_linked_facts:
            logger.info("[LinkedViewSystem] No pending linked facts to save")
            return ""

        output_path_obj = Path(output_path)

        # 判断是目录还是文件路径
        if output_path_obj.suffix == "" or output_path_obj.is_dir():
            # 如果是目录，自动生成文件名
            output_path_obj.mkdir(parents=True, exist_ok=True)
            # 从记录中获取 session_id，或使用 current_session_id
            session_id = self.current_session_id
            if self._pending_linked_facts and "session_id" in self._pending_linked_facts[0]:
                session_id = self._pending_linked_facts[0]["session_id"]
            output_file = output_path_obj / f"{session_id}_linked_facts.json"
        else:
            # 如果是文件路径，直接使用
            output_file = output_path_obj
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self._pending_linked_facts, f, ensure_ascii=False, indent=2)

        logger.info(f"[LinkedViewSystem] Saved {len(self._pending_linked_facts)} linked fact records to {output_file}")
        return str(output_file)

    def flush_linked_facts_to_database(self, session_id: str = None) -> int:
        """
        将所有待写入的 linked facts 写入数据库。

        Args:
            session_id: 会话 ID（如果不指定，使用记录中的 session_id）

        Returns:
            成功写入的 fact 数量
        """
        if not self._pending_linked_facts:
            logger.info("[LinkedViewSystem] No pending linked facts to flush")
            return 0

        total_written = 0
        for record in self._pending_linked_facts:
            sid = session_id or record.get("session_id", self.current_session_id)
            query_idx = record.get("query_idx", 0)
            linked_facts = record.get("linked_facts", [])

            written = self._do_write_facts_to_database(
                linked_facts=linked_facts,
                session_id=sid,
                query_idx=query_idx,
            )
            total_written += written

        logger.info(f"[LinkedViewSystem] Flushed {total_written} linked facts to database")

        # 清空待写入列表
        self._pending_linked_facts = []

        return total_written

    def _write_facts_to_database(
        self,
        linked_facts: List[Dict[str, Any]],
        session_id: str,
        query_idx: int = 0,
        question: str = "",
        raw_hits: List[Any] = None,
    ) -> int:
        """
        记录并可选地写入 linked facts 到数据库。

        功能：
        1. 记录 linked facts 及其上下文（question, raw_hits）用于分析
        2. 如果 write_facts_to_database=True，立即写入数据库
        3. 否则只记录，等待手动调用 flush_linked_facts_to_database()

        Args:
            linked_facts: [{"fact": str, "source_pages": [page_id], "evidence_snippets": [str]}, ...]
            session_id: 会话 ID
            query_idx: 当前 query 索引
            question: 当前查询问题（用于记录）
            raw_hits: 召回的原始数据（用于记录）

        Returns:
            成功写入的 fact 数量（如果 write_facts_to_database=False，返回 0）
        """
        if not linked_facts:
            return 0

        # 始终记录（如果开启了记录功能）
        if self.record_linked_facts:
            self._record_linked_facts(
                question=question,
                linked_facts=linked_facts,
                raw_hits=raw_hits or [],
                session_id=session_id,
                query_idx=query_idx,
            )

        # 如果开启了立即写入，则写入数据库
        if self.write_facts_to_database:
            return self._do_write_facts_to_database(
                linked_facts=linked_facts,
                session_id=session_id,
                query_idx=query_idx,
            )

        return 0

    def _do_write_facts_to_database(
        self,
        linked_facts: List[Dict[str, Any]],
        session_id: str,
        query_idx: int = 0,
    ) -> int:
        """
        实际执行写入 linked facts 到数据库的操作。

        Args:
            linked_facts: [{"fact": str, "source_pages": [page_id], "evidence_snippets": [str]}, ...]
            session_id: 会话 ID
            query_idx: 当前 query 索引

        Returns:
            成功写入的 fact 数量
        """
        if not linked_facts:
            return 0

        # 检查 index 是否可用
        if self.index is None:
            logger.warning(f"[LinkedViewSystem] _do_write_facts_to_database: index not initialized, skipping")
            return 0

        written_count = 0
        for lf in linked_facts:
            fact_text = lf.get("fact", "").strip()
            if not fact_text:
                logger.info(f"[LinkedViewSystem] _do_write_facts_to_database: fact_text is empty, skipping")
                continue

            source_pages = lf.get("source_pages", [])
            evidence_snippets = lf.get("evidence_snippets", [])

            # 使用第一个 source_page 作为 page_id（用于检索时的溯源）
            primary_page_id = source_pages[0] if source_pages else f"linked_fact_{session_id}_{written_count}"

            # 构建 metadata
            metadata = {
                "page_id": primary_page_id,
                "session_id": session_id,
                "source_type": "linked_fact",
                "source_pages": source_pages,
                "evidence_snippets": evidence_snippets,
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "written_at_query_idx": query_idx,
            }

            try:
                with self._index_lock:
                    self.index.add(
                        user_id=f"{session_id}:user",  # 与原始数据格式一致
                        raw_chunk=fact_text,
                        metadata=metadata,
                        infer=False,
                    )
                written_count += 1
                logger.info(f"[LinkedViewSystem] Wrote linked fact to database: {fact_text[:100]}...")
            except Exception as e:
                logger.error(f"[LinkedViewSystem] Failed to write linked fact: {e}")
                continue

        logger.info(f"[LinkedViewSystem] _do_write_facts_to_database: {written_count}/{len(linked_facts)} facts written")

        # 记录写回事件
        if written_count > 0:
            self._record_write_event(
                session_id=session_id,
                query_idx=query_idx,
                facts_written=written_count,
                linked_facts=linked_facts,
            )

        return written_count

    # === 智能 Memory 回填相关方法 ===

    MEMORY_DECISION_PROMPT = """You are a memory management assistant. Decide how to handle a new fact given existing memories.

Question: {question}

NEW FACT:
{new_fact}

EXISTING MEMORIES (top {top_k} by relevance):
{existing_memories}

Output a JSON decision (ONLY output valid JSON, no other text):
{{
  "action": "SKIP" or "UPDATE" or "DELETE_AND_ADD" or "ADD",
  "target_memory_index": <index number from the list above (0, 1, 2...) or null if action is ADD>,
  "reason": "<brief explanation>",
  "updated_text": "<merged text if UPDATE, otherwise null>"
}}

Criteria:
- SKIP: New fact is semantically redundant (same meaning, different words)
- UPDATE: New fact improves/corrects an existing memory (merge them)
- DELETE_AND_ADD: Existing memory is wrong/outdated, replace with new fact
- ADD: New fact is genuinely new information

Be conservative: prefer SKIP when uncertain.
IMPORTANT: For target_memory_index, use the index number [0], [1], [2] etc from the list above, NOT the ID string."""

    def _decide_memory_action(
        self,
        fact_text: str,
        question: str,
        existing_memories: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        使用 LLM 决策如何处理新 fact。

        Args:
            fact_text: 新 fact 的文本
            question: 触发此 fact 的问题
            existing_memories: 已有的相关 memories（包含 id, memory, score）
            top_k: 召回的 memory 数量

        Returns:
            决策结果，包含 action, target_memory_id, reason, updated_text
        """
        if not existing_memories:
            return {"action": "ADD", "target_memory_id": None, "reason": "No existing memories"}

        # 格式化已有 memories（使用索引而非 ID，便于 LLM 理解）
        memories_text = ""
        for i, mem in enumerate(existing_memories):
            memories_text += f"[{i}] Text: {mem['memory']}\n    Score: {mem['score']:.3f}\n\n"

        prompt = self.MEMORY_DECISION_PROMPT.format(
            question=question,
            new_fact=fact_text,
            existing_memories=memories_text,
            top_k=top_k,
        )

        try:
            response = self.llm_memory_system.generate(prompt)
            # 尝试解析 JSON
            decision = json_repair.loads(response)

            # 确保 decision 是字典
            if isinstance(decision, str):
                # 如果返回的是字符串，尝试再次解析
                decision = json_repair.loads(decision)

            if not isinstance(decision, dict):
                logger.warning(f"[LinkedViewSystem] LLM decision is not a dict: {type(decision)}, defaulting to ADD")
                return {"action": "ADD", "target_memory_id": None, "reason": "Invalid LLM response format"}

            # 将 target_memory_index 转换为实际的 memory_id
            target_idx = decision.get("target_memory_index")
            target_memory_id = None
            if target_idx is not None:
                try:
                    idx = int(target_idx)
                    if 0 <= idx < len(existing_memories):
                        target_memory_id = existing_memories[idx].get("id")
                    else:
                        logger.warning(f"[LinkedViewSystem] Invalid target_memory_index {idx}, out of range [0, {len(existing_memories)})")
                except (ValueError, TypeError):
                    logger.warning(f"[LinkedViewSystem] Invalid target_memory_index: {target_idx}")

            # 构建标准化的决策结果
            result = {
                "action": decision.get("action", "ADD"),
                "target_memory_id": target_memory_id,
                "reason": decision.get("reason", ""),
                "updated_text": decision.get("updated_text"),
            }

            logger.info(f"[LinkedViewSystem] Memory decision: action={result['action']}, target_id={target_memory_id}, reason={result['reason'][:50] if result['reason'] else 'N/A'}")
            return result

        except Exception as e:
            logger.warning(f"[LinkedViewSystem] LLM decision failed: {e}, defaulting to ADD")
            return {"action": "ADD", "target_memory_id": None, "reason": f"LLM error: {e}"}

    def _smart_write_fact_to_database(
        self,
        fact: Dict[str, Any],
        question: str,
        session_id: str,
        query_idx: int,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        智能写入单条 fact，包含召回比对和 LLM 决策。

        流程：
        1. 用 fact_text 作为 query 搜索已有 memory (top_k=5)
        2. 让 LLM 判断操作类型：SKIP/UPDATE/DELETE_AND_ADD/ADD
        3. 执行相应操作

        Args:
            fact: 要写入的 fact，格式为 {"fact": str, "source_pages": [...], ...}
            question: 触发此 fact 的问题
            session_id: 会话 ID
            query_idx: 当前 query 索引
            top_k: 召回的 memory 数量

        Returns:
            {
                "action": "SKIP" | "UPDATE" | "DELETE_AND_ADD" | "ADD",
                "fact_text": str,
                "target_memory_id": str | None,
                "reason": str,
                "success": bool,
                "error": str | None,
            }
        """
        fact_text = fact.get("fact", "").strip()
        if not fact_text:
            return {"action": "SKIP", "fact_text": "", "reason": "Empty fact", "success": True}

        user_id = f"{session_id}:user"

        # 1. 召回已有 memory
        with self._index_lock:
            hits = self.index.search(user_id=user_id, query=fact_text, top_k=top_k)

        existing_memories = [
            {
                "id": h.memory_id,
                "memory": h.summary_text,
                "score": h.score,
            }
            for h in hits if h.memory_id
        ]

        # 2. LLM 决策
        decision = self._decide_memory_action(fact_text, question, existing_memories, top_k)
        action = decision.get("action", "ADD")
        target_id = decision.get("target_memory_id")

        # 验证 target_id 是否有效（应该是 UUID 字符串）
        def _is_valid_memory_id(mid):
            if mid is None:
                return False
            if not isinstance(mid, str):
                return False
            # 简单检查：UUID 通常是 36 字符（含连字符）或 32 字符（不含连字符）
            return len(mid) >= 32

        if action in ("UPDATE", "DELETE_AND_ADD") and not _is_valid_memory_id(target_id):
            logger.warning(f"[LinkedViewSystem] Invalid target_memory_id '{target_id}' for {action}, falling back to ADD")
            action = "ADD"
            target_id = None

        # 3. 执行操作
        result = {
            "action": action,
            "fact_text": fact_text,
            "target_memory_id": target_id,
            "reason": decision.get("reason", ""),
            "success": False,
            "error": None,
        }

        try:
            if action == "SKIP":
                result["success"] = True
                logger.info(f"[LinkedViewSystem] Smart write SKIP: {fact_text[:50]}... (reason: {result['reason'][:30] if result['reason'] else 'N/A'})")

            elif action == "UPDATE" and target_id:
                updated_text = decision.get("updated_text") or fact_text
                with self._index_lock:
                    update_success = self.index.update(target_id, updated_text, user_id)
                if update_success:
                    result["success"] = True
                    logger.info(f"[LinkedViewSystem] Smart write UPDATE: memory_id={target_id}, new_text={updated_text[:50]}...")
                else:
                    # UPDATE 失败，回退到 ADD
                    logger.warning(f"[LinkedViewSystem] UPDATE failed for {target_id}, falling back to ADD")
                    source_pages = fact.get("source_pages", [])
                    metadata = {
                        "page_id": source_pages[0] if source_pages else f"smart_{session_id}_{query_idx}",
                        "session_id": session_id,
                        "source_type": "smart_linked_fact",
                        "source_pages": source_pages,
                        "evidence_snippets": fact.get("evidence_snippets", []),
                        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "written_at_query_idx": query_idx,
                    }
                    with self._index_lock:
                        self.index.add(user_id=user_id, raw_chunk=fact_text, metadata=metadata, infer=False)
                    result["action"] = "ADD"
                    result["success"] = True

            elif action == "DELETE_AND_ADD" and target_id:
                with self._index_lock:
                    delete_success = self.index.delete(target_id)
                    if not delete_success:
                        logger.warning(f"[LinkedViewSystem] DELETE failed for {target_id}, proceeding with ADD anyway")
                    # 添加新 fact
                    source_pages = fact.get("source_pages", [])
                    metadata = {
                        "page_id": source_pages[0] if source_pages else f"smart_{session_id}_{query_idx}",
                        "session_id": session_id,
                        "source_type": "smart_linked_fact",
                        "source_pages": source_pages,
                        "evidence_snippets": fact.get("evidence_snippets", []),
                        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "written_at_query_idx": query_idx,
                        "replaced_memory_id": target_id if delete_success else None,
                    }
                    self.index.add(user_id=user_id, raw_chunk=fact_text, metadata=metadata, infer=False)
                result["success"] = True
                logger.info(f"[LinkedViewSystem] Smart write DELETE_AND_ADD: deleted={target_id}, added={fact_text[:50]}...")

            elif action == "ADD":
                source_pages = fact.get("source_pages", [])
                metadata = {
                    "page_id": source_pages[0] if source_pages else f"smart_{session_id}_{query_idx}",
                    "session_id": session_id,
                    "source_type": "smart_linked_fact",
                    "source_pages": source_pages,
                    "evidence_snippets": fact.get("evidence_snippets", []),
                    "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "written_at_query_idx": query_idx,
                }
                with self._index_lock:
                    self.index.add(user_id=user_id, raw_chunk=fact_text, metadata=metadata, infer=False)
                result["success"] = True
                logger.info(f"[LinkedViewSystem] Smart write ADD: {fact_text[:50]}...")

            else:
                # 未知 action 或缺少 target_id，回退到 ADD
                logger.warning(f"[LinkedViewSystem] Smart write: unknown action '{action}' or missing target_id, falling back to ADD")
                source_pages = fact.get("source_pages", [])
                metadata = {
                    "page_id": source_pages[0] if source_pages else f"smart_{session_id}_{query_idx}",
                    "session_id": session_id,
                    "source_type": "smart_linked_fact",
                    "source_pages": source_pages,
                    "evidence_snippets": fact.get("evidence_snippets", []),
                    "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "written_at_query_idx": query_idx,
                }
                with self._index_lock:
                    self.index.add(user_id=user_id, raw_chunk=fact_text, metadata=metadata, infer=False)
                result["action"] = "ADD"
                result["success"] = True

        except Exception as e:
            logger.error(f"[LinkedViewSystem] Smart write failed: {e}")
            result["error"] = str(e)

        return result

    # === 批量智能回填方法（优化版）===

    MEMORY_BATCH_DECISION_PROMPT = """Given a question and some facts, decide which facts to save to a database.

Question: {question}

New facts:
{new_facts}

Existing database entries:
{existing_memories}

For each fact, decide action:
- ADD: new information, save as-is
- SKIP: duplicates existing entry or another fact
- UPDATE: improves/corrects an existing entry, provide merged text

Return JSON array:
[
  {{"idx": 0, "act": "ADD"}},
  {{"idx": 1, "act": "SKIP"}},
  {{"idx": 2, "act": "UPDATE", "target": 0, "text": "merged content here"}}
]

For UPDATE: target is the existing entry index, text is the new merged content.
Output only valid JSON array."""

    def _batch_decide_memory_actions(
        self,
        facts: List[Dict[str, Any]],
        question: str,
        existing_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        批量使用 LLM 决策如何处理多条 facts（一次 LLM 调用）。

        Args:
            facts: 待处理的 facts 列表
            question: 触发这些 facts 的问题
            existing_memories: 用 question 召回的已有 memories

        Returns:
            决策结果列表
        """
        if not facts:
            return []

        # 格式化 facts
        facts_text = ""
        for i, fact in enumerate(facts):
            fact_text = fact.get("fact", "").strip()
            facts_text += f"[{i}] {fact_text}\n"

        # 格式化已有 memories
        if existing_memories:
            memories_text = ""
            for i, mem in enumerate(existing_memories):
                memories_text += f"[{i}] {mem['memory']}\n"
        else:
            memories_text = "(No existing memories)"

        prompt = self.MEMORY_BATCH_DECISION_PROMPT.format(
            question=question,
            new_facts=facts_text,
            existing_memories=memories_text,
        )

        response = None
        try:
            #print(f"Batch decision prompt: {prompt}")
            response = self.llm_memory_system.generate(prompt)
            #print(f"Batch decision response: {response}")
            # 记录原始响应到日志文件（用于调试）
            try:
                import os
                log_dir = os.path.join("logs", "smart_flush_debug")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"batch_decisions_{time.strftime('%Y%m%d_%H%M%S')}.log")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"=== Question: {question[:100]} ===\n")
                    f.write(f"Facts count: {len(facts)}\n")
                    f.write(f"Existing memories count: {len(existing_memories)}\n")
                    f.write(f"Raw response:\n{response}\n")
                    f.write("=" * 50 + "\n\n")
            except Exception as log_e:
                logger.debug(f"[LinkedViewSystem] Failed to write debug log: {log_e}")

            decisions = json_repair.loads(response)

            if isinstance(decisions, str):
                decisions = json_repair.loads(decisions)

            if not isinstance(decisions, list):
                logger.warning(f"[LinkedViewSystem] Batch decision not a list (got {type(decisions).__name__}), defaulting all to ADD")
                return [{"fact_index": i, "action": "ADD", "target_memory_id": None, "reason": "Invalid response type"} for i in range(len(facts))]

            # 验证并修复每个 decision 的格式
            validated_decisions = []
            for i, d in enumerate(decisions):
                if isinstance(d, dict):
                    # 支持新格式 {"idx": 0, "act": "ADD"} 和旧格式 {"fact_index": 0, "action": "ADD"}
                    fact_idx = d.get("idx") if "idx" in d else d.get("fact_index", i)
                    action = d.get("act") if "act" in d else d.get("action", "ADD")
                    # 标准化 action
                    action = action.upper() if isinstance(action, str) else "ADD"
                    if action not in ("ADD", "SKIP", "UPDATE"):
                        action = "ADD"

                    # 处理 UPDATE 的 target 和 text
                    target_idx = d.get("target") if "target" in d else d.get("target_memory_index")
                    updated_text = d.get("text") if "text" in d else d.get("updated_text")

                    # 转换 target 索引到 memory_id
                    target_memory_id = None
                    if target_idx is not None and existing_memories:
                        try:
                            idx = int(target_idx)
                            if 0 <= idx < len(existing_memories):
                                target_memory_id = existing_memories[idx].get("id")
                        except (ValueError, TypeError):
                            pass

                    validated_decisions.append({
                        "fact_index": fact_idx,
                        "action": action,
                        "target_memory_id": target_memory_id,
                        "updated_text": updated_text,
                        "reason": d.get("reason", str(d.get("dup", ""))),
                    })
                elif isinstance(d, int):
                    logger.warning(f"[LinkedViewSystem] Decision {i} is int, converting to ADD")
                    validated_decisions.append({"fact_index": i, "action": "ADD", "target_memory_id": None, "reason": "Auto-converted"})
                else:
                    logger.warning(f"[LinkedViewSystem] Decision {i} has unexpected type {type(d).__name__}, defaulting to ADD")
                    validated_decisions.append({"fact_index": i, "action": "ADD", "target_memory_id": None, "reason": "Invalid format"})

            decisions = validated_decisions

            actions_summary = ", ".join(f"{d.get('action', 'ADD')}" for d in decisions[:5])
            if len(decisions) > 5:
                actions_summary += f"... ({len(decisions)} total)"
            logger.info(f"[LinkedViewSystem] Batch decision: {actions_summary}")
            return decisions

        except Exception as e:
            # 记录失败的响应
            try:
                import os
                log_dir = os.path.join("logs", "smart_flush_debug")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"batch_decisions_errors_{time.strftime('%Y%m%d')}.log")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"=== ERROR at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"Question: {question[:100]}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"Raw response: {response}\n")
                    f.write("=" * 50 + "\n\n")
            except Exception:
                pass

            logger.warning(f"[LinkedViewSystem] Batch LLM decision failed: {e}, defaulting all to ADD")
            return [{"fact_index": i, "action": "ADD", "target_memory_id": None, "reason": f"LLM error"} for i in range(len(facts))]

    def _smart_write_facts_batch(
        self,
        facts: List[Dict[str, Any]],
        question: str,
        session_id: str,
        query_idx: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        批量智能写入多条 facts（一次召回 + 一次 LLM 决策）。

        Args:
            facts: 要写入的 facts 列表
            question: 触发这些 facts 的问题
            session_id: 会话 ID
            query_idx: 当前 query 索引
            top_k: 召回的 memory 数量

        Returns:
            每条 fact 的处理结果列表
        """
        if not facts:
            return []

        user_id = f"{session_id}:user"

        # 过滤空 facts，记录原始索引
        valid_facts = []
        valid_indices = []
        for i, fact in enumerate(facts):
            if fact.get("fact", "").strip():
                valid_facts.append(fact)
                valid_indices.append(i)

        if not valid_facts:
            return [{"action": "SKIP", "fact_text": "", "reason": "Empty fact", "success": True} for _ in facts]

        # 1. 用 question 召回已有 memory（只召回一次）
        with self._index_lock:
            hits = self.index.search(user_id=user_id, query=question, top_k=top_k)

        existing_memories = [
            {"id": h.memory_id, "memory": h.summary_text, "score": h.score}
            for h in hits if h.memory_id
        ]

        # 2. 批量 LLM 决策（一次调用）
        decisions = self._batch_decide_memory_actions(valid_facts, question, existing_memories)

        # 创建决策映射
        decision_map = {d.get("fact_index", -1): d for d in decisions}

        # 3. 执行各条 fact 的操作
        results = []
        for orig_idx, fact in enumerate(facts):
            fact_text = fact.get("fact", "").strip()
            if not fact_text:
                results.append({"action": "SKIP", "fact_text": "", "reason": "Empty fact", "success": True})
                continue

            # 找到在 valid_facts 中的索引
            try:
                valid_idx = valid_indices.index(orig_idx)
            except ValueError:
                results.append({"action": "SKIP", "fact_text": fact_text, "reason": "Index error", "success": False})
                continue

            decision = decision_map.get(valid_idx, {"action": "ADD", "target_memory_id": None, "reason": "No decision"})
            action = decision.get("action", "ADD")
            target_id = decision.get("target_memory_id")

            # 验证 target_id
            if action in ("UPDATE", "DELETE_AND_ADD"):
                if not (target_id and isinstance(target_id, str) and len(target_id) >= 32):
                    action = "ADD"
                    target_id = None

            result = {
                "action": action,
                "fact_text": fact_text,
                "target_memory_id": target_id,
                "reason": decision.get("reason", ""),
                "updated_text": decision.get("updated_text"),
                "success": False,
                "error": None,
            }

            try:
                if action == "SKIP":
                    result["success"] = True

                elif action == "UPDATE" and target_id:
                    # UPDATE = 删除旧的 + 添加新的合并文本
                    updated_text = decision.get("updated_text") or fact_text
                    with self._index_lock:
                        # 先删除旧的
                        self.index.delete(target_id)
                    # 添加新的合并文本
                    updated_fact = fact.copy()
                    updated_fact["fact"] = updated_text
                    self._do_add_fact_internal(updated_fact, session_id, query_idx, user_id, replaced_id=target_id, smart_action="UPDATE")
                    result["success"] = True
                    result["updated_text"] = updated_text

                else:  # ADD
                    self._do_add_fact_internal(fact, session_id, query_idx, user_id, smart_action="ADD")
                    result["action"] = "ADD"
                    result["success"] = True

            except Exception as e:
                logger.error(f"[LinkedViewSystem] Batch write failed for fact {orig_idx}: {e}")
                result["error"] = str(e)

            results.append(result)

        return results

    def _do_add_fact_internal(
        self,
        fact: Dict[str, Any],
        session_id: str,
        query_idx: int,
        user_id: str,
        replaced_id: Optional[str] = None,
        smart_action: str = "ADD",
    ):
        """内部辅助方法：执行添加 fact 操作（智能回填专用）"""
        fact_text = fact.get("fact", "").strip()
        source_pages = fact.get("source_pages", [])
        evidence_snippets = fact.get("evidence_snippets", [])

        # 使用 source_pages[0] 作为 page_id（它就是 page_store 里已有的 page_id）
        page_id = source_pages[0] if source_pages else f"smart_{session_id}_{query_idx}"

        if not source_pages:
            logger.warning(f"[LinkedViewSystem] Fact has no source_pages, using fake page_id: {page_id}, fact: {fact_text[:50]}...")

        metadata = {
            "page_id": page_id,
            "session_id": session_id,
            # 区分来源
            "source_type": "smart_linked_fact",  # 区别于 "linked_fact"
            "flush_phase": "smart_backfill",     # 明确标记是回填阶段写入
            "smart_action": smart_action,        # ADD 或 UPDATE
            # 原始信息
            "source_pages": source_pages,
            "evidence_snippets": evidence_snippets,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "written_at_query_idx": query_idx,
        }
        if replaced_id:
            metadata["replaced_memory_id"] = replaced_id
        with self._index_lock:
            self.index.add(user_id=user_id, raw_chunk=fact_text, metadata=metadata, infer=False)


__all__ = ["LinkedViewSystem"]


