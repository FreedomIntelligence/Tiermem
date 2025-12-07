"""
RawLLMSystem适配器

最简单的baseline：直接将所有历史对话拼接，然后调用LLM回答问题。
用于验证pipeline和作为baseline对比。
"""
import time
from typing import Dict, Any, Optional
from core.systems.base import MemorySystem, Turn, ObserveResult, AnswerResult


class RawLLMSystem(MemorySystem):
    """
    原始LLM系统：无记忆管理，直接拼接所有历史
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.history: List[Turn] = []
        self.api_client = None  # 需要初始化LLM客户端
        
    def reset(self, session_id: str) -> None:
        """重置历史记录"""
        super().reset(session_id)
        self.history = []
    
    def observe(self, turn: Turn) -> ObserveResult:
        """
        简单存储历史对话（无实际memory操作，cost为0）
        """
        start_time = time.time()
        
        # 存储turn
        self.history.append(turn)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return ObserveResult(
            cost_metrics={
                "total_latency_ms": latency_ms,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "api_calls_count": 0
            },
            storage_stats={
                "total_items_count": len(self.history),
                "current_context_window": sum(len(t.text) for t in self.history)  # 简单估算
            }
        )
    
    def answer(self, query: str, meta: Optional[Dict[str, Any]] = None) -> AnswerResult:
        """
        拼接所有历史+问题，调用LLM
        """
        start_time = time.time()
        
        # 构建prompt
        history_text = "\n".join([
            f"{turn.speaker}: {turn.text}" 
            for turn in self.history
        ])
        
        prompt = f"""Based on the following conversation history, please answer the question.

Conversation History:
{history_text}

Question: {query}

Answer:"""
        
        # TODO: 实际调用LLM API
        # 这里先用mock实现
        answer = self._call_llm_mock(prompt)
        
        # 计算cost（mock值，实际需要从API响应中获取）
        latency_ms = int((time.time() - start_time) * 1000)
        tokens_in = len(prompt.split())  # 简单估算
        tokens_out = len(answer.split())  # 简单估算
        
        return AnswerResult(
            answer=answer,
            cost_metrics={
                "online_tokens_in": tokens_in,
                "online_tokens_out": tokens_out,
                "online_retrieval_latency_ms": 0,  # 无检索
                "online_generation_latency_ms": latency_ms,
                "online_total_latency_ms": latency_ms,
                "online_api_calls": 1
            },
            mechanism_trace={
                "retrieved_contexts": [],
                "history_length": len(self.history)
            }
        )
    
    def _call_llm_mock(self, prompt: str) -> str:
        """
        Mock LLM调用（实际使用时需要替换为真实的API调用）
        """
        # 简单mock：返回固定答案
        return "I need more information to answer this question accurately."
    
    def get_system_name(self) -> str:
        return "rawllm"


