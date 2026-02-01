"""
主入口 API：二路 (S / R) Linked-View answer()。

严格对应规范中的签名和逻辑：

def answer(user_id: str, query: str,
           index: SummaryIndex, raw_store: RawStore,
           router, llm_fast, llm_slow,
           top_k: int = 8) -> dict:
    ...
"""

from typing import Any, Dict

from .raw_store import RawStore
from .summary_index import SummaryIndex, EvidenceSet
from .router import BinaryRouter
from .pipelines_fast import fast_answer
from .pipelines_slow import build_evidence, guided_research


def answer(
    user_id: str,
    query: str,
    index: SummaryIndex,
    raw_store: RawStore,
    router: BinaryRouter,
    llm_fast,
    llm_slow,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    核心查询入口：
    - 统一 Retrieval -> Router -> {S, R} Pipeline
    - 返回 answer / route / hits / cost 便于上层实验记录
    """

    hits = index.search(user_id, query, top_k=top_k)
    # API 层暂时不维护 GlobalMetaMemory，上层如需可自行构造 extra_context
    evidence = EvidenceSet(query=query, hits=hits)

    action = router.decide(evidence)  # "S" or "R"

    if action == "S":
        ans = fast_answer(llm_fast, query, hits)
        cost = {"route": "S"}
    else:
        evidence_docs = build_evidence(raw_store, hits)
        mem_ctx = "\n".join([h.summary_text for h in hits])
        ans = guided_research(llm_slow, query, evidence_docs)
        cost = {"route": "R", "raw_docs_used": len(evidence_docs)}

    return {"answer": ans, "route": action, "hits": hits, "cost": cost}


__all__ = ["answer"]


