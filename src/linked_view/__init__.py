"""
Linked-View Architecture 实现（binary routing: S vs R）。

本包提供：
- `raw_store`：append-only RawStore
- `summary_index`：SummaryIndex 封装（面向 Mem0 / 其它向量库）
- `router`：只看 (Q, Z) 的二分类 Router
- `pipelines_fast`：基于 summaries 的快速回答（S）
- `pipelines_slow`：Guided GAM Researcher（R）
- `api`：统一的 `answer()` 入口
"""

from .raw_store import RawLogRecord, RawStore
from .summary_index import SummaryHit, SummaryIndex, EvidenceSet
from .router import RouteExample, BinaryRouter, LLMRouter, QueryRewriter, ThinkingLLMRouter
from .pipelines_fast import fast_answer
from .pipelines_slow import EvidenceDoc, build_evidence, guided_research, GuidedResearcher
from .api import answer
from .page_store import Page, PageStore

__all__ = [
    "RawLogRecord",
    "RawStore",
    "SummaryHit",
    "SummaryIndex",
    "EvidenceSet",
    "RouteExample",
    "BinaryRouter",
    "LLMRouter",
    "QueryRewriter",
    "ThinkingLLMRouter",
    "fast_answer",
    "EvidenceDoc",
    "build_evidence",
    "guided_research",
    "GuidedResearcher",
    "answer",
    "Page",
    "PageStore",
]



