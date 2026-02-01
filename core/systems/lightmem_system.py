"""
LightMem 系统的 MemorySystem 接口包装器

将 LightMem 包装为统一的 MemorySystem 接口，用于 benchmark 评估。
支持使用 OpenAI embedding 和共享的 Qdrant 服务器。
"""
import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from openai import OpenAI

# 添加 LightMem 源码路径
LIGHTMEM_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "relatedwork", "LightMem", "src"
)
if LIGHTMEM_SRC not in sys.path:
    sys.path.insert(0, LIGHTMEM_SRC)

from core.systems.base import MemorySystem, Turn, ObserveResult, AnswerResult

logger = logging.getLogger(__name__)


class LightMemSystem(MemorySystem):
    """
    LightMem 的 MemorySystem 包装器

    配置示例:
    {
        "benchmark_name": "longmemeval",
        "model": "gpt-4.1-mini",           # LLM model for memory extraction and QA
        "embedding_model": "text-embedding-3-small",  # OpenAI embedding model
        "embedding_dims": 1536,
        "qdrant_host": "localhost",        # Qdrant 服务器
        "qdrant_port": 6333,
        "top_k": 20,                        # 检索数量
        "pre_compress": False,              # 是否预压缩
        "topic_segment": False,             # 是否做主题分割
    }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}

        # 保存原始配置用于创建新实例
        self._original_config = config.copy() if config else {}

        # 基本配置
        self.benchmark_name = self.config.get("benchmark_name", "longmemeval")
        self.model = self.config.get("model", "gpt-4.1-mini")
        self.embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        self.embedding_dims = self.config.get("embedding_dims", 1536)
        self.qdrant_host = self.config.get("qdrant_host", "localhost")
        self.qdrant_port = self.config.get("qdrant_port", 6333)
        self.top_k = self.config.get("top_k", 20)
        self.pre_compress = self.config.get("pre_compress", False)
        self.topic_segment = self.config.get("topic_segment", False)

        # LLM client for QA
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = self.config.get("base_url") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        self.llm_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # LightMem 实例（在 reset/load 时创建）
        self.lightmem = None
        self.current_session_id = None

        # 统计信息
        self._total_observe_time_ms = 0
        self._total_observe_calls = 0
        self._total_memories = 0

    def _get_collection_name(self, session_id: str) -> str:
        """生成唯一的 collection 名称"""
        # 清理 session_id 中的特殊字符
        clean_session_id = session_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        return f"lightmem_{self.benchmark_name}_{clean_session_id}"

    def _create_lightmem_config(self, collection_name: str) -> Dict[str, Any]:
        """创建 LightMem 配置"""
        config = {
            # 禁用预压缩（如需启用需要安装 llmlingua-2）
            "pre_compress": self.pre_compress,

            # 禁用主题分割（简化版）
            "topic_segment": self.topic_segment,

            # 使用 user_only 模式
            "messages_use": "user_only",

            # 启用元数据生成和文本摘要
            "metadata_generate": True,
            "text_summary": True,

            # Memory Manager 配置（使用 OpenAI）
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": self.model,
                    "api_key": self.api_key,
                    "max_tokens": 16000,
                    "openai_base_url": self.base_url or ""
                }
            },

            # 提取阈值
            "extract_threshold": 0.1,

            # 索引策略：使用 embedding
            "index_strategy": "embedding",

            # Text Embedder 配置（使用 OpenAI embedding）
            "text_embedder": {
                "model_name": "huggingface",  # 使用 huggingface factory，但配置为调用 OpenAI API
                "configs": {
                    "model": self.embedding_model,
                    "embedding_dims": self.embedding_dims,
                    "huggingface_base_url": self.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "",
                    "api_key": self.api_key,
                },
            },

            # 检索策略：使用 embedding
            "retrieve_strategy": "embedding",

            # Embedding Retriever 配置（使用共享 Qdrant 服务器）
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": collection_name,
                    "embedding_model_dims": self.embedding_dims,
                    "host": self.qdrant_host,
                    "port": self.qdrant_port,
                    # 不使用本地路径，使用远程服务器
                    # "path": None,
                    "on_disk": False,
                }
            },

            # 更新模式：离线
            "update": "offline",
        }

        return config

    def _create_lightmem(self, session_id: str):
        """创建 LightMem 实例"""
        from lightmem.memory.lightmem import LightMemory

        collection_name = self._get_collection_name(session_id)
        config = self._create_lightmem_config(collection_name)

        logger.info(f"Creating LightMem instance for session {session_id}, collection: {collection_name}")

        self.lightmem = LightMemory.from_config(config)

        # 重置 collection（清空旧数据）
        try:
            self.lightmem.embedding_retriever.reset()
            logger.info(f"Reset collection {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to reset collection: {e}")

    def reset(self, session_id: str) -> None:
        """开始新的会话，清空状态"""
        self.current_session_id = session_id

        # 重置统计信息
        self._total_observe_time_ms = 0
        self._total_observe_calls = 0
        self._total_memories = 0

        # 创建新的 LightMem 实例
        self._create_lightmem(session_id)

        logger.info(f"[{session_id}] LightMemSystem reset complete")

    def load(self, session_id: str) -> None:
        """加载已存在的会话（不清空数据）"""
        from lightmem.memory.lightmem import LightMemory

        self.current_session_id = session_id

        collection_name = self._get_collection_name(session_id)
        config = self._create_lightmem_config(collection_name)

        logger.info(f"Loading LightMem instance for session {session_id}, collection: {collection_name}")

        # 创建实例但不重置 collection
        self.lightmem = LightMemory.from_config(config)

        logger.info(f"[{session_id}] LightMemSystem load complete")

    def observe(self, turn: Turn) -> ObserveResult:
        """处理一条对话轮次，写入记忆"""
        if self.lightmem is None:
            raise RuntimeError("LightMem not initialized. Call reset() or load() first.")

        start_time = time.time()

        # 构造消息格式
        # LightMem 需要 time_stamp 字段
        timestamp = turn.timestamp or datetime.now().strftime("%Y/%m/%d (%a) %H:%M")
        msg = {
            "role": turn.speaker or "user",
            "content": turn.text,
            "time_stamp": timestamp,
        }

        # 判断是否是最后一条消息（强制提取）
        # 注意：在并发模式下，每条消息都单独处理，所以这里设置为 True
        force_segment = True
        force_extract = True

        try:
            result = self.lightmem.add_memory(
                messages=[msg],
                force_segment=force_segment,
                force_extract=force_extract,
            )

            # 统计 API 调用次数
            api_calls = result.get("api_call_nums", 0) if isinstance(result, dict) else 0

        except Exception as e:
            logger.error(f"[{self.current_session_id}] LightMem add_memory failed: {e}")
            raise

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        self._total_observe_time_ms += latency_ms
        self._total_observe_calls += 1

        return ObserveResult(
            cost_metrics={
                "total_latency_ms": latency_ms,
                "total_tokens_in": 0,  # LightMem 不直接提供 token 统计
                "total_tokens_out": 0,
                "api_calls_count": api_calls,
            },
            storage_stats={
                "total_observe_calls": self._total_observe_calls,
            },
            mechanism_trace={
                "add_memory_result": result if isinstance(result, dict) else {},
            }
        )

    def answer(self, query: str, meta: Optional[Dict[str, Any]] = None) -> AnswerResult:
        """回答问题"""
        if self.lightmem is None:
            raise RuntimeError("LightMem not initialized. Call reset() or load() first.")

        start_time = time.time()

        # 检索相关记忆
        try:
            related_memories = self.lightmem.retrieve(query, limit=self.top_k)
        except Exception as e:
            logger.error(f"[{self.current_session_id}] LightMem retrieve failed: {e}")
            related_memories = ""

        # 构造 prompt 生成答案
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided memories."},
            {"role": "user", "content": f"Question: {query}\n\nRelevant memories:\n{related_memories}\n\nPlease answer the question based on the memories above. If the memories don't contain relevant information, say so."}
        ]

        # 调用 LLM 生成答案
        max_retries = 3
        generated_answer = ""
        tokens_in = 0
        tokens_out = 0

        for attempt in range(max_retries):
            try:
                completion = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.0,
                )
                generated_answer = completion.choices[0].message.content or ""

                # 获取 token 使用量
                if hasattr(completion, 'usage') and completion.usage:
                    tokens_in = completion.usage.prompt_tokens
                    tokens_out = completion.usage.completion_tokens

                break
            except Exception as e:
                logger.warning(f"[{self.current_session_id}] LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    generated_answer = f"Error generating answer: {e}"
                else:
                    time.sleep(2 ** attempt)  # 指数退避

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return AnswerResult(
            answer=generated_answer,
            cost_metrics={
                "online_tokens_in": tokens_in,
                "online_tokens_out": tokens_out,
                "online_total_latency_ms": latency_ms,
                "online_api_calls": 1,
            },
            mechanism_trace={
                "retrieved_memories": related_memories,
                "retrieval_count": self.top_k,
            }
        )

    def auto_summary(self, session_id: str) -> None:
        """自动生成会话摘要（LightMem 不需要）"""
        pass

    def get_system_name(self) -> str:
        """返回系统名称"""
        return "lightmem"
