#!/usr/bin/env python3
"""
分析 router 统计信息脚本

统计内容：
1. 每个 session_id 走了多少次 S，平均消耗多少 token
2. 走了多少次 R，消耗了多少 token
3. 第一次走了 query 后，选了 S 还是 R，分别消耗了多少 token
4. 这四个内容的准确率（S的准确率、R的准确率、第一次query后选S的准确率、第一次query后选R的准确率）
5. S vs R 对比统计：S对R对、S对R错、S错R对、S错R错的数量
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse


def load_qa_logs(sessions_dir: Path) -> List[Dict]:
    """加载所有 session 的 QA 日志"""
    all_logs = []
    for qa_file in sorted(sessions_dir.glob("*_qa.jsonl")):
        if not qa_file.exists() or qa_file.stat().st_size == 0:
            continue
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)
                    all_logs.append(log)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {qa_file}: {e}")
                    continue
    
    return all_logs


def load_eval_results(results_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    加载评估结果（从 results/sessions/*_eval.json 或 sessions/*_eval.json）
    
    支持两种路径格式：
    1. results_dir = results/ -> 会在 results/sessions/ 下查找
    2. results_dir = results/sessions/ -> 直接在此目录下查找
    
    Returns:
        Dict[session_id, Dict[query_id, eval_data]]
    """
    eval_data = {}
    
    # 先尝试作为 sessions 目录（直接查找 *_eval.json）
    if results_dir.exists() and any(results_dir.glob("*_eval.json")):
        sessions_dir = results_dir
    else:
        # 否则尝试作为 results 目录（查找 sessions 子目录）
        sessions_dir = results_dir / "sessions"
    
    if not sessions_dir.exists():
        return eval_data
    
    for eval_file in sorted(sessions_dir.glob("*_eval.json")):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_id = session_data.get("session_id", "")
            details = session_data.get("details", [])
            
            if session_id not in eval_data:
                eval_data[session_id] = {}
            
            for detail in details:
                query_id = detail.get("query_id", "")
                if query_id:
                    eval_data[session_id][query_id] = detail
        except Exception as e:
            print(f"Warning: Failed to load eval file {eval_file}: {e}")
            continue
    
    return eval_data


def load_eval_details_json(eval_details_path: Path) -> Dict[str, Dict[str, Dict]]:
    """
    加载 eval_details_with_llm_judge.json 文件
    
    Returns:
        Dict[session_id, Dict[query_id, eval_data]]
    """
    eval_data = {}
    
    if not eval_details_path.exists():
        return eval_data
    
    try:
        with open(eval_details_path, 'r', encoding='utf-8') as f:
            details_list = json.load(f)
        
        for detail in details_list:
            session_id = detail.get("session_id", "")
            query_id = detail.get("query_id", "")
            
            if session_id and query_id:
                if session_id not in eval_data:
                    eval_data[session_id] = {}
                eval_data[session_id][query_id] = detail
    except Exception as e:
        print(f"Warning: Failed to load eval details file {eval_details_path}: {e}")
    
    return eval_data


def compare_s_vs_r(s_eval_data: Dict[str, Dict[str, Dict]], 
                   r_eval_data: Dict[str, Dict[str, Dict]]) -> Dict:
    """
    比较 S 和 R 的结果
    
    Returns:
        包含四种情况的统计：S对R对、S对R错、S错R对、S错R错
    """
    comparison_stats = {
        "S_correct_R_correct": 0,  # S对R对
        "S_correct_R_wrong": 0,    # S对R错
        "S_wrong_R_correct": 0,    # S错R对
        "S_wrong_R_wrong": 0,      # S错R错
        "total_matched": 0          # 总共匹配到的记录数
    }
    
    # 遍历所有 session_id 和 query_id
    all_keys = set()
    for session_id in s_eval_data.keys():
        all_keys.update((session_id, qid) for qid in s_eval_data[session_id].keys())
    for session_id in r_eval_data.keys():
        all_keys.update((session_id, qid) for qid in r_eval_data[session_id].keys())
    
    for session_id, query_id in all_keys:
        s_detail = s_eval_data.get(session_id, {}).get(query_id)
        r_detail = r_eval_data.get(session_id, {}).get(query_id)
        
        # 只有当两个都存在时才进行比较
        if s_detail is None or r_detail is None:
            continue
        
        # 获取 llm_judge_score（1表示正确，0表示错误）
        s_score = s_detail.get("llm_judge_score", None)
        r_score = r_detail.get("llm_judge_score", None)
        
        # 如果两个都有分数，则进行比较
        if s_score is not None and r_score is not None:
            s_correct = (s_score == 1)
            r_correct = (r_score == 1)
            
            comparison_stats["total_matched"] += 1
            
            if s_correct and r_correct:
                comparison_stats["S_correct_R_correct"] += 1
            elif s_correct and not r_correct:
                comparison_stats["S_correct_R_wrong"] += 1
            elif not s_correct and r_correct:
                comparison_stats["S_wrong_R_correct"] += 1
            else:
                comparison_stats["S_wrong_R_wrong"] += 1
    
    return comparison_stats


def merge_qa_with_eval(qa_logs: List[Dict], eval_data: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    """
    合并 QA 日志和评估结果
    
    优先使用 LLM judge 的结果（如果存在），否则使用原始的 score
    """
    merged_logs = []
    
    for log in qa_logs:
        session_id = log.get("session_id", "")
        query_id = log.get("query_id", "")
        
        # 尝试从 eval_data 获取 LLM judge 结果
        if session_id in eval_data and query_id in eval_data[session_id]:
            eval_detail = eval_data[session_id][query_id]
            # 使用 LLM judge 的结果（更准确）
            llm_judge_score = eval_detail.get("llm_judge_score", None)
            if llm_judge_score is not None:
                # llm_judge_score: 1 表示正确，0 表示错误
                log["score"] = float(llm_judge_score)
                log["llm_judge_label"] = eval_detail.get("llm_judge_label", "")
        
        merged_logs.append(log)
    
    return merged_logs


def has_query_in_history(router_history: List[Dict]) -> bool:
    """检查 router_history 中是否有 QUERY action"""
    if not router_history:
        return False
    
    for entry in router_history:
        action = entry.get("action", "")
        if action.startswith("QUERY:") or action == "QUERY":
            return True
    return False


def get_first_query_final_action(router_history: List[Dict]) -> Optional[str]:
    """获取第一次 QUERY 后的最终 action（S 或 R）"""
    if not router_history:
        return None
    
    found_query = False
    for entry in router_history:
        action = entry.get("action", "")
        if action.startswith("QUERY:") or action == "QUERY":
            found_query = True
        elif found_query and action in ("S", "R"):
            return action
    
    # 如果找到了 QUERY 但没有后续的 S/R，返回 None
    return None


def is_first_action(router_history: List[Dict], target_action: str) -> bool:
    """检查是否是第一次直接选 S 或 R（没有经过 QUERY）"""
    if not router_history:
        return False
    
    # 检查是否有 QUERY
    has_query = has_query_in_history(router_history)
    if has_query:
        return False
    
    # 如果没有 QUERY，检查第一个非 initial_search 的 action 是否是目标 action
    for entry in router_history:
        action = entry.get("action", "")
        if action == "initial_search":
            continue
        return action == target_action
    
    return False


def init_s_vs_r_stats():
    """初始化 S vs R 对比统计"""
    return {
        "S_correct_R_correct": 0,
        "S_correct_R_wrong": 0,
        "S_wrong_R_correct": 0,
        "S_wrong_R_wrong": 0,
        "total_matched": 0
    }


def update_s_vs_r_comparison(comparison_stats: Dict, session_id: str, query_id: str,
                            s_eval_data: Optional[Dict[str, Dict[str, Dict]]],
                            r_eval_data: Optional[Dict[str, Dict[str, Dict]]]):
    """
    更新 S vs R 对比统计
    
    Args:
        comparison_stats: 要更新的统计字典
        session_id: session ID
        query_id: query ID
        s_eval_data: S路径的评估数据
        r_eval_data: R路径的评估数据
    """
    if not s_eval_data or not r_eval_data:
        return
    
    s_detail = s_eval_data.get(session_id, {}).get(query_id)
    r_detail = r_eval_data.get(session_id, {}).get(query_id)
    
    # 只有当两个都存在时才进行比较
    if s_detail is None or r_detail is None:
        return
    
    # 获取 llm_judge_score（1表示正确，0表示错误）
    s_score = s_detail.get("llm_judge_score", None)
    r_score = r_detail.get("llm_judge_score", None)
    
    # 如果两个都有分数，则进行比较
    if s_score is not None and r_score is not None:
        s_correct = (s_score == 1)
        r_correct = (r_score == 1)
        
        comparison_stats["total_matched"] += 1
        
        if s_correct and r_correct:
            comparison_stats["S_correct_R_correct"] += 1
        elif s_correct and not r_correct:
            comparison_stats["S_correct_R_wrong"] += 1
        elif not s_correct and r_correct:
            comparison_stats["S_wrong_R_correct"] += 1
        else:
            comparison_stats["S_wrong_R_wrong"] += 1


def analyze_router_stats(logs: List[Dict], 
                        s_eval_data: Optional[Dict[str, Dict[str, Dict]]] = None,
                        r_eval_data: Optional[Dict[str, Dict[str, Dict]]] = None) -> Dict:
    """
    分析 router 统计信息
    
    Args:
        logs: QA 日志列表
        s_eval_data: S路径的评估数据，Dict[session_id, Dict[query_id, eval_data]]
        r_eval_data: R路径的评估数据，Dict[session_id, Dict[query_id, eval_data]]
    """
    
    # 按 session_id 分组
    by_session = defaultdict(list)
    for log in logs:
        session_id = log.get("session_id", "unknown")
        by_session[session_id].append(log)
    
    # 全局统计
    stats = {
        "total_qa_pairs": len(logs),
        "total_sessions": len(by_session),
        "by_session": {},
        "global": {
            "total": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0
            },
            "S": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "R": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_S": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_R": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_query_then_S": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_query_then_R": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            }
        }
    }
    
    # 逐个 session 分析
    for session_id, session_logs in sorted(by_session.items()):
        session_stats = {
            "session_id": session_id,
            "total_qa_pairs": len(session_logs),
            "S": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "R": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_S": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_R": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_query_then_S": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            },
            "first_query_then_R": {
                "count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_retrieval_latency_ms": 0,
                "total_generation_latency_ms": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "s_vs_r_comparison": init_s_vs_r_stats()
            }
        }
        
        for log in session_logs:
            # 获取路由信息
            mechanism_trace = log.get("mechanism_trace", {})
            final_action = mechanism_trace.get("final_action") or mechanism_trace.get("route", "unknown")
            router_history = mechanism_trace.get("router_history", [])
            
            # 获取 token 信息
            cost_metrics = log.get("cost_metrics", {})
            tokens_in = cost_metrics.get("llm_prompt_tokens", 0)
            tokens_out = cost_metrics.get("llm_completion_tokens", 0)
            total_tokens = tokens_in + tokens_out
            
            # 获取耗时信息
            total_latency_ms = cost_metrics.get("online_total_latency_ms", 0)
            retrieval_latency_ms = cost_metrics.get("online_retrieval_latency_ms", 0)
            generation_latency_ms = cost_metrics.get("online_generation_latency_ms", 0)
            
            # 获取准确率
            score = log.get("score", 0.0)
            is_correct = (score == 1.0)
            
            # 获取 session_id 和 query_id 用于 S vs R 对比
            session_id_for_comparison = log.get("session_id", "")
            query_id = log.get("query_id", "")
            
            # 判断是否有 QUERY
            has_query = has_query_in_history(router_history)
            first_query_action = get_first_query_final_action(router_history) if has_query else None
            is_first_s = is_first_action(router_history, "S")
            is_first_r = is_first_action(router_history, "R")
            
            # 统计总计（所有路径）
            stats["global"]["total"]["count"] += 1
            stats["global"]["total"]["total_tokens_in"] += tokens_in
            stats["global"]["total"]["total_tokens_out"] += tokens_out
            stats["global"]["total"]["total_tokens"] += total_tokens
            stats["global"]["total"]["total_latency_ms"] += total_latency_ms
            stats["global"]["total"]["total_retrieval_latency_ms"] += retrieval_latency_ms
            stats["global"]["total"]["total_generation_latency_ms"] += generation_latency_ms
            if is_correct:
                stats["global"]["total"]["correct"] += 1
            else:
                stats["global"]["total"]["wrong"] += 1
            
            # 统计 S 路径
            if final_action == "S":
                session_stats["S"]["count"] += 1
                session_stats["S"]["total_tokens_in"] += tokens_in
                session_stats["S"]["total_tokens_out"] += tokens_out
                session_stats["S"]["total_tokens"] += total_tokens
                session_stats["S"]["total_latency_ms"] += total_latency_ms
                session_stats["S"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                session_stats["S"]["total_generation_latency_ms"] += generation_latency_ms
                if is_correct:
                    session_stats["S"]["correct"] += 1
                else:
                    session_stats["S"]["wrong"] += 1
                
                # 全局统计
                stats["global"]["S"]["count"] += 1
                stats["global"]["S"]["total_tokens_in"] += tokens_in
                stats["global"]["S"]["total_tokens_out"] += tokens_out
                stats["global"]["S"]["total_tokens"] += total_tokens
                stats["global"]["S"]["total_latency_ms"] += total_latency_ms
                stats["global"]["S"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                stats["global"]["S"]["total_generation_latency_ms"] += generation_latency_ms
                if is_correct:
                    stats["global"]["S"]["correct"] += 1
                else:
                    stats["global"]["S"]["wrong"] += 1
                
                # 更新 S vs R 对比统计
                update_s_vs_r_comparison(stats["global"]["S"]["s_vs_r_comparison"], 
                                        session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                update_s_vs_r_comparison(session_stats["S"]["s_vs_r_comparison"], 
                                        session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                
                # 统计 first_S（第一次直接选 S，没有 QUERY）
                if is_first_s:
                    session_stats["first_S"]["count"] += 1
                    session_stats["first_S"]["total_tokens_in"] += tokens_in
                    session_stats["first_S"]["total_tokens_out"] += tokens_out
                    session_stats["first_S"]["total_tokens"] += total_tokens
                    session_stats["first_S"]["total_latency_ms"] += total_latency_ms
                    session_stats["first_S"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                    session_stats["first_S"]["total_generation_latency_ms"] += generation_latency_ms
                    if is_correct:
                        session_stats["first_S"]["correct"] += 1
                    else:
                        session_stats["first_S"]["wrong"] += 1
                    
                    # 全局统计
                    stats["global"]["first_S"]["count"] += 1
                    stats["global"]["first_S"]["total_tokens_in"] += tokens_in
                    stats["global"]["first_S"]["total_tokens_out"] += tokens_out
                    stats["global"]["first_S"]["total_tokens"] += total_tokens
                    stats["global"]["first_S"]["total_latency_ms"] += total_latency_ms
                    stats["global"]["first_S"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                    stats["global"]["first_S"]["total_generation_latency_ms"] += generation_latency_ms
                    if is_correct:
                        stats["global"]["first_S"]["correct"] += 1
                    else:
                        stats["global"]["first_S"]["wrong"] += 1
                    
                    # 更新 S vs R 对比统计
                    update_s_vs_r_comparison(stats["global"]["first_S"]["s_vs_r_comparison"], 
                                            session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                    update_s_vs_r_comparison(session_stats["first_S"]["s_vs_r_comparison"], 
                                            session_id_for_comparison, query_id, s_eval_data, r_eval_data)
            
            # 统计 R 路径
            elif final_action == "R":
                session_stats["R"]["count"] += 1
                session_stats["R"]["total_tokens_in"] += tokens_in
                session_stats["R"]["total_tokens_out"] += tokens_out
                session_stats["R"]["total_tokens"] += total_tokens
                session_stats["R"]["total_latency_ms"] += total_latency_ms
                session_stats["R"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                session_stats["R"]["total_generation_latency_ms"] += generation_latency_ms
                if is_correct:
                    session_stats["R"]["correct"] += 1
                else:
                    session_stats["R"]["wrong"] += 1
                
                # 全局统计
                stats["global"]["R"]["count"] += 1
                stats["global"]["R"]["total_tokens_in"] += tokens_in
                stats["global"]["R"]["total_tokens_out"] += tokens_out
                stats["global"]["R"]["total_tokens"] += total_tokens
                stats["global"]["R"]["total_latency_ms"] += total_latency_ms
                stats["global"]["R"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                stats["global"]["R"]["total_generation_latency_ms"] += generation_latency_ms
                if is_correct:
                    stats["global"]["R"]["correct"] += 1
                else:
                    stats["global"]["R"]["wrong"] += 1
                
                # 更新 S vs R 对比统计
                update_s_vs_r_comparison(stats["global"]["R"]["s_vs_r_comparison"], 
                                        session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                update_s_vs_r_comparison(session_stats["R"]["s_vs_r_comparison"], 
                                        session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                
                # 统计 first_R（第一次直接选 R，没有 QUERY）
                if is_first_r:
                    session_stats["first_R"]["count"] += 1
                    session_stats["first_R"]["total_tokens_in"] += tokens_in
                    session_stats["first_R"]["total_tokens_out"] += tokens_out
                    session_stats["first_R"]["total_tokens"] += total_tokens
                    session_stats["first_R"]["total_latency_ms"] += total_latency_ms
                    session_stats["first_R"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                    session_stats["first_R"]["total_generation_latency_ms"] += generation_latency_ms
                    if is_correct:
                        session_stats["first_R"]["correct"] += 1
                    else:
                        session_stats["first_R"]["wrong"] += 1
                    
                    # 全局统计
                    stats["global"]["first_R"]["count"] += 1
                    stats["global"]["first_R"]["total_tokens_in"] += tokens_in
                    stats["global"]["first_R"]["total_tokens_out"] += tokens_out
                    stats["global"]["first_R"]["total_tokens"] += total_tokens
                    stats["global"]["first_R"]["total_latency_ms"] += total_latency_ms
                    stats["global"]["first_R"]["total_retrieval_latency_ms"] += retrieval_latency_ms
                    stats["global"]["first_R"]["total_generation_latency_ms"] += generation_latency_ms
                    if is_correct:
                        stats["global"]["first_R"]["correct"] += 1
                    else:
                        stats["global"]["first_R"]["wrong"] += 1
                    
                    # 更新 S vs R 对比统计
                    update_s_vs_r_comparison(stats["global"]["first_R"]["s_vs_r_comparison"], 
                                            session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                    update_s_vs_r_comparison(session_stats["first_R"]["s_vs_r_comparison"], 
                                            session_id_for_comparison, query_id, s_eval_data, r_eval_data)
            
            # 统计第一次 QUERY 后的路径
            if has_query and first_query_action:
                key = f"first_query_then_{first_query_action}"
                session_stats[key]["count"] += 1
                session_stats[key]["total_tokens_in"] += tokens_in
                session_stats[key]["total_tokens_out"] += tokens_out
                session_stats[key]["total_tokens"] += total_tokens
                session_stats[key]["total_latency_ms"] += total_latency_ms
                session_stats[key]["total_retrieval_latency_ms"] += retrieval_latency_ms
                session_stats[key]["total_generation_latency_ms"] += generation_latency_ms
                if is_correct:
                    session_stats[key]["correct"] += 1
                else:
                    session_stats[key]["wrong"] += 1
                
                # 全局统计
                stats["global"][key]["count"] += 1
                stats["global"][key]["total_tokens_in"] += tokens_in
                stats["global"][key]["total_tokens_out"] += tokens_out
                stats["global"][key]["total_tokens"] += total_tokens
                stats["global"][key]["total_latency_ms"] += total_latency_ms
                stats["global"][key]["total_retrieval_latency_ms"] += retrieval_latency_ms
                stats["global"][key]["total_generation_latency_ms"] += generation_latency_ms
                if is_correct:
                    stats["global"][key]["correct"] += 1
                else:
                    stats["global"][key]["wrong"] += 1
                
                # 更新 S vs R 对比统计
                update_s_vs_r_comparison(stats["global"][key]["s_vs_r_comparison"], 
                                        session_id_for_comparison, query_id, s_eval_data, r_eval_data)
                update_s_vs_r_comparison(session_stats[key]["s_vs_r_comparison"], 
                                        session_id_for_comparison, query_id, s_eval_data, r_eval_data)
        
        # 计算 session 级别的准确率和平均 token/latency
        for key in ["S", "R", "first_S", "first_R", "first_query_then_S", "first_query_then_R"]:
            if session_stats[key]["count"] > 0:
                session_stats[key]["accuracy"] = session_stats[key]["correct"] / session_stats[key]["count"]
                session_stats[key]["avg_tokens_in"] = session_stats[key]["total_tokens_in"] / session_stats[key]["count"]
                session_stats[key]["avg_tokens_out"] = session_stats[key]["total_tokens_out"] / session_stats[key]["count"]
                session_stats[key]["avg_tokens"] = session_stats[key]["total_tokens"] / session_stats[key]["count"]
                session_stats[key]["avg_latency_ms"] = session_stats[key]["total_latency_ms"] / session_stats[key]["count"]
                session_stats[key]["avg_retrieval_latency_ms"] = session_stats[key]["total_retrieval_latency_ms"] / session_stats[key]["count"]
                session_stats[key]["avg_generation_latency_ms"] = session_stats[key]["total_generation_latency_ms"] / session_stats[key]["count"]
        
        stats["by_session"][session_id] = session_stats
    
    # 计算全局准确率和平均 token/latency
    for key in ["total", "S", "R", "first_S", "first_R", "first_query_then_S", "first_query_then_R"]:
        if stats["global"][key]["count"] > 0:
            stats["global"][key]["accuracy"] = stats["global"][key]["correct"] / stats["global"][key]["count"]
            stats["global"][key]["avg_tokens_in"] = stats["global"][key]["total_tokens_in"] / stats["global"][key]["count"]
            stats["global"][key]["avg_tokens_out"] = stats["global"][key]["total_tokens_out"] / stats["global"][key]["count"]
            stats["global"][key]["avg_tokens"] = stats["global"][key]["total_tokens"] / stats["global"][key]["count"]
            stats["global"][key]["avg_latency_ms"] = stats["global"][key]["total_latency_ms"] / stats["global"][key]["count"]
            stats["global"][key]["avg_retrieval_latency_ms"] = stats["global"][key]["total_retrieval_latency_ms"] / stats["global"][key]["count"]
            stats["global"][key]["avg_generation_latency_ms"] = stats["global"][key]["total_generation_latency_ms"] / stats["global"][key]["count"]
    
    return stats


def print_stats(stats: Dict):
    """打印统计结果"""
    print("=" * 80)
    print("Router 统计信息")
    print("=" * 80)
    print(f"\n总 Session 数: {stats['total_sessions']}")
    print(f"总 QA 对数: {stats['total_qa_pairs']}")
    
    print("\n" + "=" * 80)
    print("全局统计")
    print("=" * 80)
    
    def print_path_stats(path_name: str, path_stats: Dict):
        """打印路径统计信息的辅助函数"""
        print(f"\n【{path_name}】")
        print(f"  数量: {path_stats['count']}")
        if path_stats['count'] > 0:
            print(f"  平均输入 token: {path_stats['avg_tokens_in']:.1f}")
            print(f"  平均输出 token: {path_stats['avg_tokens_out']:.1f}")
            print(f"  平均总 token: {path_stats['avg_tokens']:.1f}")
            print(f"  平均总耗时: {path_stats['avg_latency_ms']:.1f} ms ({path_stats['avg_latency_ms']/1000:.2f} s)")
            print(f"  平均检索耗时: {path_stats['avg_retrieval_latency_ms']:.1f} ms")
            print(f"  平均生成耗时: {path_stats['avg_generation_latency_ms']:.1f} ms")
            print(f"  总输入 token: {path_stats['total_tokens_in']}")
            print(f"  总输出 token: {path_stats['total_tokens_out']}")
            print(f"  总 token: {path_stats['total_tokens']}")
            print(f"  总耗时: {path_stats['total_latency_ms']:.1f} ms ({path_stats['total_latency_ms']/1000:.2f} s)")
            print(f"  正确: {path_stats['correct']}, 错误: {path_stats['wrong']}")
            print(f"  准确率: {path_stats['accuracy']:.2%}")
            
            # 打印 S vs R 对比统计
            if "s_vs_r_comparison" in path_stats:
                comp = path_stats["s_vs_r_comparison"]
                if comp["total_matched"] > 0:
                    print(f"  S vs R 对比:")
                    print(f"    匹配记录数: {comp['total_matched']}")
                    print(f"    S对R对: {comp['S_correct_R_correct']} ({comp['S_correct_R_correct']/comp['total_matched']*100:.2f}%)")
                    print(f"    S对R错: {comp['S_correct_R_wrong']} ({comp['S_correct_R_wrong']/comp['total_matched']*100:.2f}%)")
                    print(f"    S错R对: {comp['S_wrong_R_correct']} ({comp['S_wrong_R_correct']/comp['total_matched']*100:.2f}%)")
                    print(f"    S错R错: {comp['S_wrong_R_wrong']} ({comp['S_wrong_R_wrong']/comp['total_matched']*100:.2f}%)")
    
    # 总计统计（所有路径）
    print_path_stats("总计（全部路径）", stats["global"]["total"])
    
    # S 路径统计
    print_path_stats("S 路径", stats["global"]["S"])
    
    # R 路径统计
    print_path_stats("R 路径", stats["global"]["R"])
    
    # 第一次直接选 S（没有 QUERY）
    print_path_stats("第一次直接选 S（无 QUERY）", stats["global"]["first_S"])
    
    # 第一次直接选 R（没有 QUERY）
    print_path_stats("第一次直接选 R（无 QUERY）", stats["global"]["first_R"])
    
    # 第一次 QUERY 后选 S
    print_path_stats("第一次 QUERY 后选 S", stats["global"]["first_query_then_S"])
    
    # 第一次 QUERY 后选 R
    print_path_stats("第一次 QUERY 后选 R", stats["global"]["first_query_then_R"])
    
    # S vs R 对比统计
    if "s_vs_r_comparison" in stats:
        print("\n" + "=" * 80)
        print("S vs R 对比统计")
        print("=" * 80)
        comp = stats["s_vs_r_comparison"]
        print(f"\n总共匹配的记录数: {comp['total_matched']}")
        print(f"S对R对: {comp['S_correct_R_correct']} ({comp['S_correct_R_correct']/comp['total_matched']*100:.2f}%)" if comp['total_matched'] > 0 else "S对R对: 0")
        print(f"S对R错: {comp['S_correct_R_wrong']} ({comp['S_correct_R_wrong']/comp['total_matched']*100:.2f}%)" if comp['total_matched'] > 0 else "S对R错: 0")
        print(f"S错R对: {comp['S_wrong_R_correct']} ({comp['S_wrong_R_correct']/comp['total_matched']*100:.2f}%)" if comp['total_matched'] > 0 else "S错R对: 0")
        print(f"S错R错: {comp['S_wrong_R_wrong']} ({comp['S_wrong_R_wrong']/comp['total_matched']*100:.2f}%)" if comp['total_matched'] > 0 else "S错R错: 0")
    
    print("\n" + "=" * 80)
    print("按 Session 统计（前 10 个）")
    print("=" * 80)
    
    # 打印前 10 个 session 的统计
    for i, (session_id, session_stats) in enumerate(list(stats["by_session"].items())[:10]):
        print(f"\n【Session: {session_id}】")
        print(f"  总 QA 对数: {session_stats['total_qa_pairs']}")
        
        s = session_stats["S"]
        if s["count"] > 0:
            print(f"  S: {s['count']} 次, 平均 {s['avg_tokens']:.1f} tokens, {s['avg_latency_ms']:.1f} ms, 准确率 {s['accuracy']:.2%}")
        
        r = session_stats["R"]
        if r["count"] > 0:
            print(f"  R: {r['count']} 次, 平均 {r['avg_tokens']:.1f} tokens, {r['avg_latency_ms']:.1f} ms, 准确率 {r['accuracy']:.2%}")
        
        first_s = session_stats["first_S"]
        if first_s["count"] > 0:
            print(f"  First S: {first_s['count']} 次, 平均 {first_s['avg_tokens']:.1f} tokens, {first_s['avg_latency_ms']:.1f} ms, 准确率 {first_s['accuracy']:.2%}")
        
        first_r = session_stats["first_R"]
        if first_r["count"] > 0:
            print(f"  First R: {first_r['count']} 次, 平均 {first_r['avg_tokens']:.1f} tokens, {first_r['avg_latency_ms']:.1f} ms, 准确率 {first_r['accuracy']:.2%}")
        
        qs = session_stats["first_query_then_S"]
        if qs["count"] > 0:
            print(f"  QUERY→S: {qs['count']} 次, 平均 {qs['avg_tokens']:.1f} tokens, {qs['avg_latency_ms']:.1f} ms, 准确率 {qs['accuracy']:.2%}")
        
        qr = session_stats["first_query_then_R"]
        if qr["count"] > 0:
            print(f"  QUERY→R: {qr['count']} 次, 平均 {qr['avg_tokens']:.1f} tokens, {qr['avg_latency_ms']:.1f} ms, 准确率 {qr['accuracy']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="分析 router 统计信息")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./results/locomo/linked_view/test_run",
        help="基础目录路径（会自动构建 sessions_dir、results_dir 和 output 路径）"
    )
    parser.add_argument(
        "--sessions_dir",
        type=str,
        default=None,
        help="Sessions 目录路径（包含 *_qa.jsonl 文件），如果未指定则使用 base_dir/sessions"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results 目录路径（包含 sessions/*_eval.json 文件，可选，用于获取 LLM judge 结果），如果未指定则使用 base_dir/results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（可选），如果未指定则使用 base_dir/router_stats.json"
    )
    parser.add_argument(
        "--s-results-dir",
        type=str,
        default="./results/locomo/linked_view/test_summary_only/results/sessions",
        help="走S路径的results目录路径（包含sessions/*_eval.json文件，用于S vs R对比）"
    )
    parser.add_argument(
        "--r-results-dir",
        type=str,
        default="./results/locomo/linked_view/test_raw_only/results/sessions",
        help="走R路径的results目录路径（包含sessions/*_eval.json文件，用于S vs R对比）"
    )
    
    args = parser.parse_args()
    
    # 根据 base_dir 自动构建路径
    base_dir = Path(args.base_dir)
    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else base_dir / "sessions"
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"
    output_path = Path(args.output) if args.output else base_dir / "router_stats.json"
    
    if not sessions_dir.exists():
        print(f"Error: 目录不存在: {sessions_dir}")
        return
    
    print(f"正在加载日志文件从: {sessions_dir}")
    logs = load_qa_logs(sessions_dir)
    print(f"加载了 {len(logs)} 条 QA 记录")
    
    # 如果提供了 results_dir，尝试加载 LLM judge 评估结果
    eval_data = {}
    if results_dir.exists():
        print(f"正在加载评估结果从: {results_dir}")
        eval_data = load_eval_results(results_dir)
        print(f"加载了 {sum(len(v) for v in eval_data.values())} 条评估记录")
        # 合并数据
        logs = merge_qa_with_eval(logs, eval_data)
        print("已合并 QA 日志和 LLM judge 评估结果")
    else:
        print(f"Warning: Results 目录不存在: {results_dir}，将使用原始 score")
    
    # 加载S和R的评估数据（如果提供了）
    s_eval_data = None
    r_eval_data = None
    
    if args.s_results_dir:
        s_results_dir = Path(args.s_results_dir)
        if s_results_dir.exists():
            print(f"\n正在加载S路径评估结果从: {s_results_dir}")
            s_eval_data = load_eval_results(s_results_dir)
            print(f"加载了 {sum(len(v) for v in s_eval_data.values())} 条S路径评估记录")
        else:
            print(f"Warning: S路径results目录不存在: {s_results_dir}")
    
    if args.r_results_dir:
        r_results_dir = Path(args.r_results_dir)
        if r_results_dir.exists():
            print(f"正在加载R路径评估结果从: {r_results_dir}")
            r_eval_data = load_eval_results(r_results_dir)
            print(f"加载了 {sum(len(v) for v in r_eval_data.values())} 条R路径评估记录")
        else:
            print(f"Warning: R路径results目录不存在: {r_results_dir}")
    
    print("正在分析统计信息...")
    stats = analyze_router_stats(logs, s_eval_data=s_eval_data, r_eval_data=r_eval_data)
    
    print_stats(stats)
    
    # 保存到 JSON 文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

