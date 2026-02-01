"""
统一 MemorySystem 接口的轻量封装，复用已有 core.systems.base 定义。

保持 dataclass 签名与 runner 兼容，并暴露统一导出，便于在
`src/memory/*` 以及 router/experiments 中引用。
"""

from core.systems.base import AnswerResult, MemorySystem, ObserveResult, Turn

__all__ = [
    "Turn",
    "ObserveResult",
    "AnswerResult",
    "MemorySystem",
]
