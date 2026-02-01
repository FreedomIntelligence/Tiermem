"""
Slow Pipeline（R）：Guided GAM Researcher（只读指定资料）。

实现内容：
- EvidenceDoc 数据结构
- build_evidence：根据 SummaryHit + RawStore 构造 evidence_docs
- guided_research：结构化 prompt + 引用 Record ID
- GuidedResearcher 包装器：在不重写 GAM 的前提下，封装 run() 接口
"""

from typing import List, Optional, Any

from .raw_store import RawStore
from .summary_index import SummaryHit
from .prompts import GUIDED_RESEARCH_PROMPT


class EvidenceDoc:
    def __init__(self, raw_log_id: str, summary: str, raw_text: str, score: Optional[float], timestamp: Optional[str]):
        self.raw_log_id = raw_log_id
        self.summary = summary
        self.raw_text = raw_text
        self.score = score
        self.timestamp = timestamp


def build_evidence(raw_store: RawStore, hits: List[SummaryHit]) -> List[EvidenceDoc]:
    """
    使用 RawStore + SummaryHit 构造 EvidenceDoc 列表。

    - 按 raw_log_id 批量拉取 raw docs
    - 自动跳过缺失 raw 的 hit
    """

    raw_ids = [h.raw_log_id for h in hits if h.raw_log_id]
    raws = raw_store.batch_get(raw_ids)
    raw_by_id = {r.raw_log_id: r for r in raws}

    docs: List[EvidenceDoc] = []
    for h in hits:
        if not h.raw_log_id:
            continue
        record = raw_by_id.get(h.raw_log_id)
        if record is None:
            continue
        docs.append(
            EvidenceDoc(
                raw_log_id=h.raw_log_id,
                summary=h.summary_text,
                raw_text=record.text,
                score=h.score,
                timestamp=h.timestamp,
            )
        )
    return docs


def guided_research(llm: Any, query: str, evidence_docs: List[EvidenceDoc]) -> str:
    """
    Guided Researcher Prompt：
    - 只允许使用给定 records
    - 要求引用 Record ID，便于误差分析
    """

    if not evidence_docs:
        # 没有 evidence 时直接返回空串；由上游决定如何 fallback
        return ""

    parts = []
    for i, d in enumerate(evidence_docs):
        parts.append(
            f"=== Record {i} (raw_log_id={d.raw_log_id}, score={d.score}, ts={d.timestamp}) ===\n"
            f"SUMMARY:\n{d.summary}\n\nRAW:\n{d.raw_text}"
        )

    evidence_str = "\n\n".join(parts)

    prompt = GUIDED_RESEARCH_PROMPT.format(
        records=evidence_str,
        question=query,
    )
    return llm.generate(prompt)


class GuidedResearcher:
    """
    GAM GuidedResearcher 包装器。

    约定被包裹的 `gam_researcher` 提供：
    - run_guided(query, memory_context, evidence_docs, allow_search: bool) -> str
      其中 evidence_docs 可以直接用 EvidenceDoc 或者由调用方转换。
    """

    def __init__(self, gam_researcher) -> None:
        self.gam = gam_researcher

    def run(self, query: str, memory_context: str, evidence_docs: List[EvidenceDoc]) -> str:
        """
        调用 GAM 内部的 planner/reasoner，但禁用 search。
        """

        if not hasattr(self.gam, "run_guided"):
            raise AttributeError("gam_researcher must implement run_guided(...) for GuidedResearcher.")

        # 将 EvidenceDoc 转为 dict，便于 GAM 侧消费
        evidence_as_dicts = [
            {
                "raw_log_id": d.raw_log_id,
                "summary": d.summary,
                "raw_text": d.raw_text,
                "score": d.score,
                "timestamp": d.timestamp,
            }
            for d in evidence_docs
        ]

        return self.gam.run_guided(
            query=query,
            memory_context=memory_context,
            evidence_docs=evidence_as_dicts,
            allow_search=False,
        )


__all__ = ["EvidenceDoc", "build_evidence", "guided_research", "GuidedResearcher"]


