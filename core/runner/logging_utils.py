"""
统一的日志记录工具

实现experiments.md中定义的两个JSON schema：
1. 记忆写入日志 (memory_write)
2. QA回答日志 (qa)
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from core.systems.base import ObserveResult, AnswerResult

def make_write_record(
    run_id: str,
    system_name: str,
    model_name: str,
    benchmark: str,
    session_id: str,
    turn_id: int,
    raw_input_text: str,
    observe_result: ObserveResult,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建记忆写入日志记录
    
    符合experiments.md中定义的schema：
    {
      "benchmark": "locomo",
      "system": "mem0",
      "session_id": "session_01",
      "turn_id": 5,
      "raw_input_text": "...",
      "timestamp": "...",
      "cost_metrics": {...},
      "storage_stats": {...}
    }
    
    Args:
        run_id: 运行ID
        system_name: 系统名称
        model_name: 模型名称
        benchmark: benchmark名称
        session_id: 会话ID
        turn_id: 轮次ID
        raw_input_text: 原始输入文本
        observe_result: observe()方法返回的结果
        timestamp: 时间戳（可选，默认使用当前时间）
        
    Returns:
        符合schema的字典
    """
    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "system": system_name,
        "model": model_name,
        "phase": "memory_write",
        "session_id": session_id,
        "turn_id": turn_id,
        "raw_input_text": raw_input_text,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "cost_metrics": observe_result.cost_metrics,
        "storage_stats": observe_result.storage_stats or {},
    }


def make_qa_record(
    run_id: str,
    system_name: str,
    model_name: str,
    benchmark: str,
    session_id: str,
    query_id: str,
    question: str,
    ground_truth: Any,
    answer_result: AnswerResult,
    score: Optional[float] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建QA回答日志记录
    
    符合experiments.md中定义的schema：
    {
      "benchmark": "locomo",
      "system": "gam",
      "query_id": "q_101",
      "ground_truth": "...",
      "model_response": "...",
      "score": 1.0,
      "cost_metrics": {...},
      "mechanism_trace": {...}
    }
    
    Args:
        run_id: 运行ID
        system_name: 系统名称
        model_name: 模型名称
        benchmark: benchmark名称
        query_id: 查询ID
        question: 问题文本
        ground_truth: 标准答案
        answer_result: answer()方法返回的结果
        score: 评分（可选，后续计算）
        timestamp: 时间戳（可选，默认使用当前时间）
        
    Returns:
        符合schema的字典
    """
    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "system": system_name,
        "model": model_name,
        "phase": "qa",
        "session_id": session_id,
        "query_id": query_id,
        "question": question,
        "ground_truth": ground_truth,
        "model_response": answer_result.answer,
        "score": score,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "cost_metrics": answer_result.cost_metrics,
        "mechanism_trace": answer_result.mechanism_trace or {},
    }


def append_log_to_jsonl(log: Dict[str, Any], filepath: str) -> None:
    """
    追加单条日志到JSONL文件（实时写入）
    
    Args:
        log: 单条日志记录
        filepath: 输出文件路径
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log, ensure_ascii=False) + '\n')


def save_logs_to_jsonl(logs: List[Dict[str, Any]], filepath: str) -> None:
    """
    将日志列表保存为JSONL格式
    
    Args:
        logs: 日志记录列表
        filepath: 输出文件路径
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')


def load_processed_sessions(filepath: str, session_key: str = "session_id") -> set:
    """
    从JSONL文件中加载已处理的session_id集合（用于断点续跑）
    
    Args:
        filepath: JSONL文件路径
        session_key: session ID的键名
        
    Returns:
        已处理的session_id集合
    """
    import json
    import os
    
    processed = set()
    
    if not os.path.exists(filepath):
        return processed
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)
                    session_id = log.get(session_key)
                    if session_id:
                        processed.add(session_id)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"⚠ 警告: 读取已处理session列表时出错: {e}")
    
    return processed


