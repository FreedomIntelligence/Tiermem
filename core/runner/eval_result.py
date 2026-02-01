"""
评估结果处理和报告生成

参考 GAM 官方的评估逻辑，提供：
1. 按 category 分类的评估
2. BLEU-1 指标
3. 详细的评估报告
"""
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from pathlib import Path
import json


def normalize_text_locomo(s: str) -> str:
    """
    LoCoMo 专用的文本标准化函数
    
    参考 GAM 官方的 normalize_text 实现
    """
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)  # drop english articles
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_locomo(s: str) -> List[str]:
    """将文本转换为 token 列表（LoCoMo 风格）"""
    s = normalize_text_locomo(s)
    return s.split() if s else []


def f1_score_locomo(pred: str, gold: str) -> float:
    """
    LoCoMo 风格的 F1 Score 计算
    
    参考 GAM 官方实现
    
    注意：pred和gold可能是int类型（如年份），normalize_text_locomo会处理
    """
    # 防御性检查：处理None
    if pred is None:
        pred = ""
    if gold is None:
        gold = ""
    
    gtoks = tokens_locomo(gold)
    ptoks = tokens_locomo(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    """
    计算 BLEU-1 Score
    
    参考 GAM 官方实现
    
    注意：pred和gold可能是int类型（如年份），normalize_text_locomo会处理
    """
    # 防御性检查：处理None
    if pred is None:
        pred = ""
    if gold is None:
        gold = ""
    
    gtoks = tokens_locomo(gold)
    ptoks = tokens_locomo(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision


def compute_metrics_by_category(
    qa_logs: List[Dict[str, Any]],
    category_key: str = "category"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按 category 计算评估指标（过滤category==5，官方不计入评估）
    
    Args:
        qa_logs: QA日志列表，每个日志应包含：
            - "model_response": 模型预测答案
            - "ground_truth": 标准答案
            - category_key: category字段（默认"category"）
        category_key: category字段的键名
        
    Returns:
        (summary, details):
        - summary: 按category汇总的指标列表
        - details: 每个QA的详细指标列表
    """
    agg = defaultdict(list)
    details = []
    
    for idx, log in enumerate(qa_logs, 1):
        # 提取category
        cat = log.get(category_key, "NA")
        if cat is None:
            cat = "NA"
        
        # 关键：过滤category==5（官方不计入评估）
        if cat == 5:
            continue
        
        # 提取预测和标准答案
        pred = log.get("model_response", "")
        gold = log.get("ground_truth", "")
        
        # 计算指标
        f1 = f1_score_locomo(pred, gold)
        b1 = bleu1_score(pred, gold)
        
        agg[cat].append((f1, b1))
        
        details.append({
            "q_idx": idx,
            "query_id": log.get("query_id", f"q_{idx}"),
            "category": cat,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "BLEU1": b1
        })
    
    # 生成汇总
    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        if scores:
            f1_avg = sum(s[0] for s in scores) / len(scores)
            b1_avg = sum(s[1] for s in scores) / len(scores)
            summary.append({
                "category": cat,
                "count": len(scores),
                "F1_avg": f1_avg,
                "BLEU1_avg": b1_avg
            })
    
    return summary, details


def compute_overall_metrics(qa_logs: List[Dict[str, Any]], category_key: str = "category", benchmark: str = "locomo") -> Dict[str, float]:
    """
    计算整体指标（不按category分类，但过滤category==5）
    
    Args:
        qa_logs: QA日志列表
        category_key: category字段的键名
        benchmark: benchmark名称，用于选择评估方法
        
    Returns:
        包含整体指标的字典
    """
    if not qa_logs:
        return {
            "overall_f1_avg": 0.0,
            "overall_bleu1_avg": 0.0,
            "num_samples": 0
        }
    
    # 根据benchmark选择评估方法
    if benchmark == "locomo":
        # LoCoMo使用专用的F1和BLEU-1计算方法
        f1_scores = []
        bleu1_scores = []
        
        for log in qa_logs:
            # 过滤category==5（官方不计入评估）
            cat = log.get(category_key)
            if cat == 5:
                continue
            
            pred = log.get("model_response", "")
            gold = log.get("ground_truth", "")
            f1 = f1_score_locomo(pred, gold)
            b1 = bleu1_score(pred, gold)
            f1_scores.append(f1)
            bleu1_scores.append(b1)
        
        return {
            "overall_f1_avg": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "overall_bleu1_avg": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
            "num_samples": len(f1_scores)
        }
    else:
        # 其他benchmark（如MemoryAgentBench）使用标准的F1计算方法
        from core.runner.scoring import f1_score
        
        f1_scores = []
        
        for log in qa_logs:
            # MemoryAgentBench可能没有category字段，不需要过滤
            pred = log.get("model_response", "")
            gold = log.get("ground_truth", "")
            f1 = f1_score(pred, gold)  # 使用标准的F1计算方法
            f1_scores.append(f1)
        
        return {
            "overall_f1_avg": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "num_samples": len(f1_scores)
        }


def generate_eval_report(
    qa_logs: List[Dict[str, Any]],
    benchmark: str,
    output_path: Path,
    category_key: str = "category"
) -> Dict[str, Any]:
    """
    生成完整的评估报告
    
    Args:
        qa_logs: QA日志列表
        benchmark: benchmark名称
        output_path: 输出目录
        category_key: category字段的键名
        
    Returns:
        评估报告字典
    """
    report = {
        "benchmark": benchmark,
        "num_samples": len(qa_logs)
    }
    
    # 计算整体指标（传入benchmark名称以选择正确的评估方法）
    overall_metrics = compute_overall_metrics(qa_logs, category_key=category_key, benchmark=benchmark)
    report.update(overall_metrics)
    
    # 如果是 LoCoMo，按 category 分类
    if benchmark == "locomo":
        summary, details = compute_metrics_by_category(qa_logs, category_key=category_key)
        report["by_category"] = summary
        report["details"] = details
        
        # 保存详细结果到CSV（可选）
        if output_path:
            details_path = output_path / "eval_details.json"
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump(details, f, ensure_ascii=False, indent=2)
    
    return report


def print_eval_report(report: Dict[str, Any], benchmark: str = "locomo"):
    """
    打印评估报告到控制台
    
    Args:
        report: 评估报告字典
        benchmark: benchmark名称
    """
    print("\n" + "="*60)
    print("评估报告")
    print("="*60)
    print(f"Benchmark: {report.get('benchmark', benchmark)}")
    print(f"样本数量: {report.get('num_samples', 0)}")
    print()
    
    # 整体指标
    if "overall_f1_avg" in report:
        print(f"整体 F1: {report['overall_f1_avg']:.4f}")
    if "overall_bleu1_avg" in report:
        print(f"整体 BLEU-1: {report['overall_bleu1_avg']:.4f}")
    print()
    
    # 按 category 分类
    if "by_category" in report and report["by_category"]:
        print("按 Category 分类:")
        print("-" * 60)
        for cat_result in report["by_category"]:
            cat = cat_result["category"]
            count = cat_result["count"]
            f1 = cat_result["F1_avg"]
            b1 = cat_result["BLEU1_avg"]
            print(f"  Category {cat}: n={count}, F1={f1:.4f}, BLEU-1={b1:.4f}")
        print()
    
    print("="*60)

