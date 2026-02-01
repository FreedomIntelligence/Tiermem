"""
Fast Pipeline（S）：基于 summaries 的快速回答。

严格对应规范中的 fast_answer 接口。
"""

from typing import List, Any

from .summary_index import SummaryHit
from .prompts import FAST_ANSWER_PROMPT
import logging

logger = logging.getLogger(__name__)


def fast_answer(llm: Any, query: str, hits: List[SummaryHit], strict: bool = False) -> str:
    """
    使用 summaries 直接回答。

    strict=True 时：若 memories 明显不足，则返回 "Not enough evidence"。
    """

    if not hits and strict:
        return "Not enough evidence"

    context = "\n".join([f"- {h.summary_text}" for h in hits])

    if strict and not context.strip():
        return "Not enough evidence"

    prompt = FAST_ANSWER_PROMPT.format(
        memories=context,
        question=query,
    )
    answer = llm.generate(prompt)
    logger.info(f"[FastPipeline] prompt: {prompt}, answer: {answer}")
    return answer


__all__ = ["fast_answer"]


