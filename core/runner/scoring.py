"""
统一的评分函数

实现各benchmark的评估指标：
- HotpotQA: EM (Exact Match) + F1 Score
- LoCoMo: 参考官方evaluation
- MemoryAgentBench: 参考官方evaluation
"""
from typing import List, Dict, Any, Union
import re
from collections import Counter
import string


def normalize_answer(s: str) -> str:
    """
    标准化答案字符串（用于EM和F1计算）
    
    参考HotpotQA的标准化方法
    
    关键修复：LoCoMo的gold answer可能是int类型（如年份2020），需要先转换为字符串
    """
    # 关键修复：先把非字符串统一成字符串，并处理None
    if s is None:
        return ""
    s = str(s)
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    import string
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    计算Exact Match分数
    
    Args:
        prediction: 模型预测答案
        ground_truth: 标准答案（可以是字符串或列表）
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # 防御性检查：处理None和类型转换
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    
    if isinstance(ground_truth, list):
        return max(exact_match_score(prediction, gt) for gt in ground_truth)
    
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    计算F1分数
    
    Args:
        prediction: 模型预测答案
        ground_truth: 标准答案（可以是字符串或列表）
        
    Returns:
        F1分数 (0.0 to 1.0)
    """
    # 防御性检查：处理None和类型转换
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    
    if isinstance(ground_truth, list):
        return max(f1_score(prediction, gt) for gt in ground_truth)
    
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_metrics(
    predictions: List[str],
    ground_truths: List[Union[str, List[str]]],
    benchmark: str = "hotpotqa"
) -> Dict[str, float]:
    """
    计算统一的评估指标
    
    Args:
        predictions: 预测答案列表
        ground_truths: 标准答案列表（每个可以是字符串或列表）
        benchmark: benchmark名称，用于选择评估方法
        
    Returns:
        包含各项指标的字典
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) length mismatch")
    
    if benchmark == "hotpotqa":
        em_scores = [exact_match_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        f1_scores = [f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        
        return {
            "exact_match": sum(em_scores) / len(em_scores),
            "f1": sum(f1_scores) / len(f1_scores),
            "num_samples": len(predictions)
        }
    
    elif benchmark == "locomo":
        # LoCoMo使用F1和BLEU-1（官方评估方式），不使用exact_match
        # 注意：这里应该使用eval_result.py中的f1_score_locomo和bleu1_score
        # 但为了保持接口统一，我们在这里调用eval_result模块的函数
        from core.runner.eval_result import f1_score_locomo, bleu1_score
        
        f1_scores = [f1_score_locomo(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        bleu1_scores = [bleu1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        
        return {
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
            "num_samples": len(predictions)
        }
    
    elif benchmark == "memory_agent_bench":
        # MemoryAgentBench使用EM和F1（与HotpotQA类似）
        em_scores = [exact_match_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        f1_scores = [f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        
        return {
            "exact_match": sum(em_scores) / len(em_scores),
            "f1": sum(f1_scores) / len(f1_scores),
            "num_samples": len(predictions)
        }
    
    else:
        from core.runner.eval_result import f1_score_locomo, bleu1_score
        
        f1_scores = [f1_score_locomo(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        bleu1_scores = [bleu1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        
        return {
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
            "num_samples": len(predictions)
        }

