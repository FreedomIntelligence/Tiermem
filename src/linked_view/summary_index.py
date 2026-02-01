"""
SummaryIndex：Mem0 / VectorDB 的统一封装。

职责：
- 接收 raw_chunk + metadata（含 raw_log_id）
- 对外提供统一的 SummaryHit 列表

实现模式：
- backend="inmemory"（默认）：内存列表 + 朴素 token overlap 打分
- backend="mem0"：调用 mem0.Memory / MemoryClient 进行写入与检索
"""

from typing import Any, Dict, List, Optional, Union
import logging
import sys
from pathlib import Path
import traceback
logger = logging.getLogger(__name__)

# 可选 mem0 依赖
try:
    # 优先使用仓库内置的 mem0（src/mem0）
    from src.mem0 import Memory, MemoryClient  # type: ignore
    from src.mem0.configs.base import MemoryConfig
    MEM0_AVAILABLE = True
    print(f"[SummaryIndex] mem0 available src.mem0")
    logger.info(f"[SummaryIndex] mem0 available src.mem0")
except Exception:
    traceback.print_exc()


class SummaryHit:
    def __init__(
        self,
        summary_text: str,
        score: float,
        raw_log_id: str,
        timestamp: Optional[str],
        session_id: Optional[str],
        extra: Dict[str, Any],
    ):
        self.summary_text = summary_text
        self.score = score
        self.raw_log_id = raw_log_id
        self.timestamp = timestamp
        self.session_id = session_id
        self.extra = extra


class EvidenceSet:
    def __init__(self, query: str, hits: List[SummaryHit], extra_context: Optional[str] = None):
        """
        Router 输入证据：
        - query: 用户当前问题 Q
        - hits: 召回到的 summary hits（Z）
        - extra_context: 额外上下文（例如 GlobalMetaMemory 的 G_t），用于 Latent-Aware Routing
        """
        self.query = query
        self.hits = hits
        self.extra_context = extra_context or ""


class SummaryIndex:
    """
    可切换后端的 SummaryIndex。

    backend:
        - "inmemory"（默认）: 轻量占位实现
        - "mem0": 使用 mem0 本地/云端
    mem0_config:
        - 仅在 backend="mem0" 时生效，原样传入 Memory / MemoryClient
        - 需要包含 mode: "local" | "api"，以及相关 key/base_url 等
    """

    def __init__(self, backend: str = "inmemory", mem0_config: Optional[Dict[str, Any]] = None) -> None:
        self.backend = backend
        self.mem0_config = mem0_config or {}
        # 最近一次 mem0 调用的真实 usage（由 mem0 本地实现采集）
        # 结构：{"prompt_tokens","completion_tokens","total_tokens","api_calls","total_latency_ms","events",...}
        self.last_mem0_usage: Optional[Dict[str, Any]] = None

        if backend == "mem0" and MEM0_AVAILABLE:
            mode = self.mem0_config.get("mode", "local")
            if mode == "api":
                self._mem0 = MemoryClient(
                    api_key=self.mem0_config.get("mem0_api_key"),
                    org_id=self.mem0_config.get("mem0_org_id"),
                    project_id=self.mem0_config.get("mem0_project_id"),
                )
            else:
                # 本地模式使用 Memory.from_config(config_dict)
                # 允许传入一个简化 dict（例如只包含 embedder/vector_store 等），其余用默认。
                cfg_raw = dict(self.mem0_config)      # 拷贝一份，避免改到原 dict
                cfg_raw.pop("backend", None)          # 去掉不是 MemoryConfig 字段的 backend

                cfg = MemoryConfig(**cfg_raw)  
                try:
                    self._mem0 = Memory(config=cfg)  # type: ignore[attr-defined]
                    logger.info(f"[SummaryIndex] Memory.from_config success: {cfg}")
                except Exception as exc:
                    logger.error(f"[SummaryIndex] Memory.from_config failed, fallback to in-memory: {exc}")
                    self._mem0 = None
        else:
            self._mem0 = None

        # 内存占位数据结构
        self._items: List[Dict[str, Any]] = []

    # === 写入 ===
    def add(
        self,
        user_id: str,
        raw_chunk: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        metadata: Dict[str, Any],
        infer: bool = True,
    ) -> None:
        """
        写入一条 summary 记录。

        约定：
        - metadata 中必须包含 "raw_log_id" 或 "page_id"
        - 可以包含 "timestamp" / "session_id" / 其它额外字段
        
        Args:
            user_id: 用户ID
            raw_chunk: 要存储的内容（总结文本）
            metadata: 元数据
            infer: 是否让mem0自动推断memory（默认True）。设为False时直接存储raw_chunk作为memory
        """

        raw_log_id = metadata.get("raw_log_id")
        page_id = metadata.get("page_id")
        if not raw_log_id and not page_id:
            raise ValueError("metadata must contain either 'raw_log_id' or 'page_id' for SummaryIndex.add")

        def _coerce_text(chunk: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
            """将 messages 结构降级为可存储/可检索的纯文本。"""
            if isinstance(chunk, str):
                return chunk
            if isinstance(chunk, dict):
                role = str(chunk.get("role") or "").strip()
                content = str(chunk.get("content") or "").strip()
                if role and content:
                    return f"{role}: {content}"
                return content or role
            if isinstance(chunk, list):
                parts: List[str] = []
                for msg in chunk:
                    if isinstance(msg, dict):
                        role = str(msg.get("role") or "").strip()
                        content = str(msg.get("content") or "").strip()
                        if role and content:
                            parts.append(f"{role}: {content}")
                        elif content:
                            parts.append(content)
                    else:
                        parts.append(str(msg))
                return "\n".join([p for p in parts if p])
            return str(chunk)

        if self._mem0 is not None:
            # mem0 写入：遵循 mem0.Memory.add(messages, *, user_id, metadata, ...)
            try:
                # 论文/实验：reset mem0 内部 recorder（如果是本地 Memory）
                if hasattr(self._mem0, "_tiermem_usage") and hasattr(self._mem0, "_tiermem_usage_scope"):
                    try:
                        self._mem0._tiermem_usage_scope = "add"  # type: ignore[attr-defined]
                        self._mem0._tiermem_usage.reset(scope="mem0.add")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if infer:
                    # 默认行为：让mem0自动推断memory
                    results = self._mem0.add(raw_chunk, user_id=user_id, metadata=metadata)  # type: ignore[call-arg]
                else:
                    # infer=False: 直接将raw_chunk作为memory存储，不进行推断
                    # - raw_chunk 为 str：包装成 messages 写入
                    # - raw_chunk 为 dict/list：直接作为 messages 写入（保留 role）
                    if isinstance(raw_chunk, str):
                        results = self._mem0.add(
                            messages=[{"role": "user", "content": raw_chunk}],
                            user_id=user_id,
                            metadata=metadata,
                            infer=infer,  # type: ignore[call-arg]
                        )
                    else:
                        results = self._mem0.add(
                            raw_chunk,
                            user_id=user_id,
                            metadata=metadata,
                            infer=infer,  # type: ignore[call-arg]
                        )
                # 读取本次 mem0.add 的真实 usage
                if hasattr(self._mem0, "_tiermem_usage"):
                    try:
                        self.last_mem0_usage = self._mem0._tiermem_usage.finalize()  # type: ignore[attr-defined]
                    except Exception:
                        self.last_mem0_usage = None
                logger.info(f"[SummaryIndex] mem0.add success (infer={infer}): {results}")
            except Exception as exc:
                # 如果infer参数不支持，尝试直接存储
                try:
                    # 某些版本的mem0可能不支持infer参数，尝试直接存储
                    results = self._mem0.add(raw_chunk, user_id=user_id, metadata=metadata)  # type: ignore[call-arg]
                    if hasattr(self._mem0, "_tiermem_usage"):
                        try:
                            self.last_mem0_usage = self._mem0._tiermem_usage.finalize()  # type: ignore[attr-defined]
                        except Exception:
                            self.last_mem0_usage = None
                    logger.info(f"[SummaryIndex] mem0.add success (fallback, infer={infer}): {results}")
                except Exception as exc2:
                    # 回退存一份本地，避免数据丢失
                    logger.error(f"[SummaryIndex] mem0.add failed, fallback to in-memory: {exc2}")
                    self._items.append({"user_id": user_id, "summary_text": _coerce_text(raw_chunk), "metadata": dict(metadata)})
            return results

        # 默认内存写入
        self._items.append({"user_id": user_id, "summary_text": _coerce_text(raw_chunk), "metadata": dict(metadata)})
        return None
    # === 检索 ===
    def search(self, user_id: str, query: str, top_k: int) -> List[SummaryHit]:
        """
        统一返回 SummaryHit 列表：
        - mem0 模式：调用 mem0.search
        - inmemory 模式：token overlap 打分
        """

        if self._mem0 is not None:
            try:
                if hasattr(self._mem0, "_tiermem_usage") and hasattr(self._mem0, "_tiermem_usage_scope"):
                    try:
                        self._mem0._tiermem_usage_scope = "search"  # type: ignore[attr-defined]
                        self._mem0._tiermem_usage.reset(scope="mem0.search")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                logger.info(f"[SummaryIndex] Calling mem0.search: query={query[:50]}..., user_id={user_id}, limit={top_k}")
                res = self._mem0.search(query=query, user_id=user_id, limit=top_k)
                logger.info(f"[SummaryIndex] mem0.search succeeded: query={query[:50]}..., result_count={len(res.get('results', [])) if isinstance(res, dict) else len(res) if isinstance(res, list) else 0}")
                if hasattr(self._mem0, "_tiermem_usage"):
                    try:
                        self.last_mem0_usage = self._mem0._tiermem_usage.finalize()  # type: ignore[attr-defined]
                    except Exception:
                        self.last_mem0_usage = None
            except Exception as exc:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(
                    f"[SummaryIndex] mem0.search failed, fallback to in-memory: "
                    f"query={query[:50]}..., user_id={user_id}, error={exc}, "
                    f"error_type={type(exc).__name__}"
                )
                logger.debug(f"[SummaryIndex] mem0.search error traceback: {error_detail}")
                # 检查是否是 502 错误（可能是 OpenAI API 或 Qdrant 的问题）
                error_str = str(exc)
                if "502" in error_str or "Bad Gateway" in error_str:
                    logger.warning(
                        f"[SummaryIndex] 502 Bad Gateway detected - "
                        f"This usually indicates: "
                        f"1) OpenAI API proxy/gateway issue (embedding generation), "
                        f"2) Qdrant service issue, or "
                        f"3) Network connectivity problem. "
                        f"Check OpenAI API status and Qdrant service health."
                    )
            else:
                return self._parse_mem0_results(res, top_k)

        # 内存检索
        tokens = set(query.lower().split())
        scored: List[SummaryHit] = []

        for item in self._items:
            if item["user_id"] != user_id:
                continue

            text: str = item["summary_text"]
            text_tokens = set(text.lower().split())
            overlap = len(tokens & text_tokens)
            if overlap == 0 and tokens:
                score = 0.0
            elif not tokens:
                score = 0.0
            else:
                score = float(overlap) / float(len(tokens))

            meta = item["metadata"]
            scored.append(
                SummaryHit(
                    summary_text=text,
                    score=score,
                    raw_log_id=str(meta.get("raw_log_id") or ""),
                    timestamp=meta.get("timestamp"),
                    session_id=meta.get("session_id"),
                    extra={k: v for k, v in meta.items() if k not in {"raw_log_id", "timestamp", "session_id"}},
                )
            )

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    # === 工具：解析 mem0 返回 ===
    def _parse_mem0_results(self, res: Any, top_k: int) -> List[SummaryHit]:
        """
        mem0.search 可能返回：
        - {"results": [...]}  或 直接 list
        每个元素含 memory / score / metadata / created_at 等。
        """

        items = res.get("results") if isinstance(res, dict) else res
        items = items or []

        hits: List[SummaryHit] = []
        for it in items:
            memory_text = it.get("memory") or it.get("text") or ""
            meta = it.get("metadata") or {}
            raw_log_id = str(meta.get("raw_log_id") or "")
            page_id = str(meta.get("page_id") or "")
            # 支持page_id或raw_log_id（page_id优先，因为现在使用分页机制）
            if not raw_log_id and not page_id:
                # 缺 raw_log_id 和 page_id 的命中对 R 没意义，跳过
                continue
            # 如果只有page_id，使用page_id作为raw_log_id（兼容现有接口）
            if not raw_log_id and page_id:
                raw_log_id = page_id
            logger.info(f"[SummaryIndex] _parse_mem0_results:  memory_text: {memory_text}, score: {it.get('score', 0.0)}, page_id: {page_id}, raw_log_id: {raw_log_id}, source_type: {meta.get('source_type', 'N/A')}")
            # 保留 page_id 和 source_type 在 extra 中，用于统计分析
            # 只排除已经单独处理的字段：raw_log_id, timestamp, session_id
            hits.append(
                SummaryHit(
                    summary_text=memory_text,
                    score=float(it.get("score", 0.0)),
                    raw_log_id=raw_log_id,
                    timestamp=it.get("created_at") or meta.get("timestamp"),
                    session_id=meta.get("session_id"),
                    extra={k: v for k, v in meta.items() if k not in {"raw_log_id", "timestamp", "session_id"}},
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


__all__ = ["SummaryHit", "EvidenceSet", "SummaryIndex"]


