"""
MemorySystem抽象接口定义

所有baseline系统都必须实现这个接口，确保统一评估。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class Turn:
    """对话轮次"""
    speaker: str  # "user" 或 "assistant"
    text: str
    timestamp: Optional[str] = None
    dia_id: Optional[str] = None

@dataclass
class ObserveResult:
    """observe()方法的返回结果"""
    cost_metrics: Dict[str, Any]  # 包含 total_latency_ms, total_tokens_in, total_tokens_out, api_calls_count
    storage_stats: Optional[Dict[str, Any]] = None  # 包含 total_items_count, current_context_window
    mechanism_trace: Optional[Dict[str, Any]] = None  # 系统特定的机制追踪信息


@dataclass
class AnswerResult:
    """answer()方法的返回结果"""
    answer: str
    cost_metrics: Dict[str, Any]  # 包含 online_tokens_in, online_tokens_out, online_total_latency_ms, online_api_calls
    mechanism_trace: Optional[Dict[str, Any]] = None  # 包含 retrieved_contexts, 系统特定的trace信息


class MemorySystem(ABC):
    """
    统一的Memory系统接口
    
    所有baseline（Mem0, MIRIX, LightMem, GAM, RAW+LLM, TierMem）都必须实现这个接口。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化系统
        
        Args:
            config: 系统配置字典，包含模型、API key等
        """
        self.config = config or {}
        self.current_session_id: Optional[str] = None
    
    @abstractmethod
    def reset(self, session_id: str) -> None:
        """
        开始一个新的对话/episode，清空内部状态
        
        Args:
            session_id: 会话ID，用于标识不同的对话会话
        """
        self.current_session_id = session_id
    
    @abstractmethod
    def observe(self, turn: Turn) -> ObserveResult:
        """
        写入阶段：处理一条对话轮次，写入记忆
        
        Args:
            turn: 对话轮次对象
            
        Returns:
            ObserveResult: 包含cost metrics和storage stats
        """
        pass
    
    @abstractmethod
    def answer(self, query: str, meta: Optional[Dict[str, Any]] = None) -> AnswerResult:
        """
        QA阶段：回答一个问题
        
        Args:
            query: 问题文本
            meta: 额外信息，如ground_truth, query_id等
            
        Returns:
            AnswerResult: 包含答案、cost metrics和mechanism trace
        """
        pass
    
    def get_system_name(self) -> str:
        """
        返回系统名称，用于日志记录
        
        子类可以重写此方法返回自定义名称
        """
        return self.__class__.__name__.replace("System", "").lower()


