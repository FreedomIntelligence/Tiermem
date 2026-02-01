#!/usr/bin/env python3
"""
对 sessions 目录下的所有 qa.jsonl 文件进行 LLM-as-Judge 评分

功能：
- 读取所有 sessions/*_qa.jsonl 文件
- 对每个 QA 进行 LLM-as-Judge 评分
- 计算每个 session 的统计（llm_judge, F1, BLEU1 等）
- 计算总体统计
- 保存结果到 results/ 目录
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import concurrent.futures
from threading import Lock
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.runner.eval_result import f1_score_locomo, bleu1_score

# 检查 OpenAI 是否可用
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: 未安装 openai 库，请运行: pip install openai")

# LLM Judge Prompt（与项目中使用的相同）
LLM_JUDGE_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(
    question: str, 
    gold_answer: str, 
    generated_answer: str,
    client: Optional[Any] = None, 
    model: str = "gpt-4.1"
) -> Dict[str, Any]:
    """
    使用 LLM 评估回答正确性
    
    Returns:
        {
            "score": 1 (CORRECT) or 0 (WRONG) or -1 (error),
            "label": "CORRECT" or "WRONG" or "ERROR",
            "explanation": str or None,
            "error": str or None
        }
    """
    if not OPENAI_AVAILABLE or client is None:
        return {
            "score": -1,
            "label": "ERROR",
            "explanation": None,
            "error": "OpenAI client not available"
        }
    
    try:
        prompt = LLM_JUDGE_PROMPT.format(
            question=str(question),
            gold_answer=str(gold_answer),
            generated_answer=str(generated_answer)
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        label = result.get("label", "").upper()
        
        score = 1 if label == "CORRECT" else 0
        
        return {
            "score": score,
            "label": label,
            "explanation": result.get("explanation"),
            "error": None
        }
    except Exception as e:
        return {
            "score": -1,
            "label": "ERROR",
            "explanation": None,
            "error": str(e)
        }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {i} 行不是合法 JSON: {e}") from e
    return rows


def process_qa_item(
    item: Dict[str, Any],
    client: Optional[Any],
    judge_model: str
) -> Dict[str, Any]:
    """处理单个 QA 项目，添加 LLM-as-Judge 评分和其他指标"""
    query_id = item.get("query_id", "")
    question = item.get("question", "")
    ground_truth = item.get("ground_truth", "")
    model_response = item.get("model_response", "")
    
    # 处理 ground_truth 可能是列表的情况
    if isinstance(ground_truth, list) and len(ground_truth) > 0:
        gold_answer = str(ground_truth[0])
    elif isinstance(ground_truth, str):
        gold_answer = ground_truth
    else:
        gold_answer = str(ground_truth) if ground_truth else ""
    
    # 计算 F1 和 BLEU1
    f1 = f1_score_locomo(model_response, gold_answer)
    bleu1 = bleu1_score(model_response, gold_answer)
    
    # LLM-as-Judge 评分
    llm_result = evaluate_llm_judge(
        question=question,
        gold_answer=gold_answer,
        generated_answer=model_response,
        client=client,
        model=judge_model
    )
    
    # 构建结果记录
    result = {
        "query_id": query_id,
        "session_id": item.get("session_id", ""),
        "question": question,
        "gold_answer": gold_answer,
        "ground_truth": ground_truth,  # 保留原始 ground_truth
        "prediction": model_response,
        "F1": f1,
        "BLEU1": bleu1,
        "llm_judge_score": llm_result["score"],
        "llm_judge_label": llm_result["label"],
        "category": item.get("category"),
        "score": item.get("score"),  # 保留原有的 score（如果有）
    }
    
    if llm_result.get("explanation"):
        result["llm_judge_explanation"] = llm_result["explanation"]
    if llm_result.get("error"):
        result["llm_judge_error"] = llm_result["error"]
    
    return result


def compute_session_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算单个 session 的统计指标"""
    if not results:
        return {
            "num_samples": 0,
            "llm_judge": {"correct": 0, "wrong": 0, "error": 0, "accuracy": 0.0},
            "f1_avg": 0.0,
            "bleu1_avg": 0.0,
        }
    
    # 过滤 category==5（官方不计入评估）
    valid_results = [r for r in results if r.get("category") != 5]
    
    if not valid_results:
        return {
            "num_samples": 0,
            "llm_judge": {"correct": 0, "wrong": 0, "error": 0, "accuracy": 0.0},
            "f1_avg": 0.0,
            "bleu1_avg": 0.0,
        }
    
    # LLM-as-Judge 统计
    llm_scores = [r.get("llm_judge_score", -1) for r in valid_results]
    correct = sum(1 for s in llm_scores if s == 1)
    wrong = sum(1 for s in llm_scores if s == 0)
    error = sum(1 for s in llm_scores if s == -1)
    accuracy = correct / len(valid_results) if valid_results else 0.0
    
    # F1 和 BLEU1 平均
    f1_scores = [r.get("F1", 0.0) for r in valid_results]
    bleu1_scores = [r.get("BLEU1", 0.0) for r in valid_results]
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    bleu1_avg = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
    
    return {
        "num_samples": len(valid_results),
        "llm_judge": {
            "correct": correct,
            "wrong": wrong,
            "error": error,
            "accuracy": accuracy,
            "correct_rate": accuracy * 100
        },
        "f1_avg": f1_avg,
        "bleu1_avg": bleu1_avg,
    }


def process_single_session(
    qa_file: Path,
    output_dir: Path,
    openai_client: Any,
    judge_model: str,
    max_workers: int,
    reuse_existing: bool
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    处理单个 session 文件
    
    Returns:
        (session_id, session_data, processed_results)
    """
    session_id = qa_file.stem.replace("_qa", "")
    
    # 如果选择复用已有结果，并且该 session 的结果文件已存在，则直接加载
    session_file = output_dir / "sessions" / f"{session_id}_eval.json"
    if reuse_existing and session_file.exists():
        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        processed_results = data.get("details", [])
        session_metrics = data.get("metrics", compute_session_metrics(processed_results))
        session_data = {
            "session_id": session_id,
            "metrics": session_metrics,
            "details": processed_results,
        }
        return session_id, session_data, processed_results

    # 读取 QA 日志
    qa_logs = read_jsonl(qa_file)
    print(f"  [Session {session_id}] 开始处理 {len(qa_logs)} 条 QA 记录...")

    # 并行处理所有 QA 项目
    processed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_qa_item, item, openai_client, judge_model): item
            for item in qa_logs
        }

        # 禁用内层进度条，避免多个 session 并发时输出混乱
        # 只在外层显示整体进度
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                processed_results.append(result)
            except Exception as e:
                # 即使出错也保留原记录（不包含 LLM 评分）
                item = futures[future]
                processed_results.append({
                    "query_id": item.get("query_id", ""),
                    "session_id": session_id,
                    "question": item.get("question", ""),
                    "gold_answer": str(item.get("ground_truth", "")),
                    "prediction": item.get("model_response", ""),
                    "F1": f1_score_locomo(item.get("model_response", ""), str(item.get("ground_truth", ""))),
                    "BLEU1": bleu1_score(item.get("model_response", ""), str(item.get("ground_truth", ""))),
                    "llm_judge_score": -1,
                    "llm_judge_label": "ERROR",
                    "llm_judge_error": str(e),
                    "category": item.get("category"),
                })

    # 计算 session 统计
    session_metrics = compute_session_metrics(processed_results)
    session_data = {
        "session_id": session_id,
        "metrics": session_metrics,
        "details": processed_results
    }
    
    return session_id, session_data, processed_results


def compute_metrics_by_category(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """按 category 计算统计指标"""
    # 按 category 分组
    by_category = defaultdict(list)
    
    for r in results:
        cat = r.get("category")
        # 过滤 category==5（官方不计入评估）
        if cat == 5:
            continue
        if cat is None:
            cat = "NA"
        by_category[cat].append(r)
    
    # 计算每个 category 的统计
    category_metrics = {}
    for cat in sorted(by_category.keys(), key=lambda x: str(x)):
        cat_results = by_category[cat]
        if not cat_results:
            continue
        
        # LLM-as-Judge 统计
        llm_scores = [r.get("llm_judge_score", -1) for r in cat_results]
        correct = sum(1 for s in llm_scores if s == 1)
        wrong = sum(1 for s in llm_scores if s == 0)
        error = sum(1 for s in llm_scores if s == -1)
        accuracy = correct / len(cat_results) if cat_results else 0.0
        
        # F1 和 BLEU1 平均
        f1_scores = [r.get("F1", 0.0) for r in cat_results]
        bleu1_scores = [r.get("BLEU1", 0.0) for r in cat_results]
        f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        bleu1_avg = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
        
        category_metrics[cat] = {
            "category": cat,
            "num_samples": len(cat_results),
            "llm_judge": {
                "correct": correct,
                "wrong": wrong,
                "error": error,
                "accuracy": accuracy,
                "correct_rate": accuracy * 100
            },
            "f1_avg": f1_avg,
            "bleu1_avg": bleu1_avg,
        }
    
    return category_metrics


def evaluate_sessions(
    sessions_dir: Path,
    output_dir: Path,
    judge_model: str = "gpt-4.1",
    batch_size: int = 20,
    max_workers: int = 20,
    reuse_existing: bool = True,
    session_id_filter: Optional[str] = None
) -> None:
    """
    对 sessions 目录下的所有 qa.jsonl 文件进行 LLM-as-Judge 评分
    
    Args:
        sessions_dir: sessions 目录路径
        output_dir: 输出目录路径
        judge_model: LLM-as-Judge 使用的模型
        batch_size: 批次大小（用于进度显示）
        max_workers: 并行处理的工作线程数
        session_id_filter: 如果指定，只处理匹配的 session_id（例如 "conv-42"）
    """
    # 初始化 OpenAI 客户端
    openai_client = None
    if OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if api_key:
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
            print("✓ OpenAI 客户端已初始化")
        else:
            print("⚠ 警告: 未设置 OPENAI_API_KEY 环境变量")
            return
    else:
        print("⚠ 警告: openai 库未安装")
        return
    
    # 查找所有 qa.jsonl 文件
    qa_files = sorted(sessions_dir.glob("*_qa.jsonl"))
    if not qa_files:
        print(f"⚠ 警告: 在 {sessions_dir} 中未找到 *_qa.jsonl 文件")
        return
    
    # 如果指定了 session_id_filter，只处理匹配的文件
    if session_id_filter:
        filtered_files = []
        for qa_file in qa_files:
            session_id = qa_file.stem.replace("_qa", "")
            if session_id == session_id_filter:
                filtered_files.append(qa_file)
        qa_files = filtered_files
        if not qa_files:
            print(f"⚠ 警告: 未找到匹配的 session_id: {session_id_filter}")
            return
        print(f"✓ 过滤到指定 session_id: {session_id_filter}")
    
    print(f"找到 {len(qa_files)} 个 QA 文件")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sessions").mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果（使用锁保护线程安全）
    all_results = []
    session_results = {}
    results_lock = Lock()
    
    # 分离外层和内层并发数，避免总线程数过大
    # 外层：session 并发数（限制为较小值，避免资源竞争）
    session_max_workers = min(5, len(qa_files))  # 最多同时处理 5 个 session
    # 内层：每个 session 内部的 QA 项目并发数
    qa_max_workers = max(1, max_workers // session_max_workers)  # 平均分配线程数
    
    print(f"\n开始并发处理 {len(qa_files)} 个 session")
    print(f"  - 外层并发数（session）: {session_max_workers}")
    print(f"  - 内层并发数（每个 session 的 QA 项目）: {qa_max_workers}")
    print(f"  - 理论最大总线程数: {session_max_workers * qa_max_workers}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=session_max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_session,
                qa_file,
                output_dir,
                openai_client,
                judge_model,
                qa_max_workers,  # 每个 session 内部的并发数
                reuse_existing
            ): qa_file
            for qa_file in qa_files
        }
        
        # 收集结果
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="处理 sessions"
        ):
            qa_file = futures[future]
            try:
                session_id, session_data, processed_results = future.result()
                
                # 线程安全地更新共享数据结构
                with results_lock:
                    session_results[session_id] = session_data
                    all_results.extend(processed_results)
                
                # 打印完成信息
                session_metrics = session_data["metrics"]
                print(f"\n✓ Session {session_id} 完成:")
                print(f"  - 样本数: {session_metrics['num_samples']}")
                print(f"  - LLM-as-Judge 准确率: {session_metrics['llm_judge']['accuracy']:.1%}")
                print(f"  - F1 平均: {session_metrics['f1_avg']:.4f}")
                print(f"  - BLEU1 平均: {session_metrics['bleu1_avg']:.4f}")
                
            except Exception as e:
                session_id = qa_file.stem.replace("_qa", "")
                print(f"\n✗ Session {session_id} 处理失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 计算总体统计
    print("\n" + "=" * 60)
    print("计算总体统计...")
    overall_metrics = compute_session_metrics(all_results)
    
    # 计算按 category 的统计
    print("计算按 category 的统计...")
    by_category_metrics = compute_metrics_by_category(all_results)

    # 尝试从原始实验目录中读取已有的 summary.json 里的 cost_summary
    cost_summary = None
    try:
        # output_dir 通常是 <实验根目录>/results
        experiment_root = output_dir.parent
        legacy_summary_path = experiment_root / "summary.json"
        if legacy_summary_path.exists():
            with legacy_summary_path.open("r", encoding="utf-8") as f:
                legacy_summary = json.load(f)
            cost_summary = legacy_summary.get("cost_summary")
            if cost_summary is not None:
                print(f"检测到已有 cost_summary，来源: {legacy_summary_path}")
    except Exception as e:
        print(f"读取原始 summary.json 以整合 cost_summary 时出错: {e}")
    
    # 构建最终结果
    final_results = {
        "overall": {
            "num_sessions": len(session_results),
            "num_samples": overall_metrics["num_samples"],
            "metrics": {
                "llm_judge": overall_metrics["llm_judge"],
                "f1_avg": overall_metrics["f1_avg"],
                "bleu1_avg": overall_metrics["bleu1_avg"],
            }
        },
        "by_category": {
            str(cat): metrics for cat, metrics in by_category_metrics.items()
        },
        "by_session": {
            session_id: {
                "session_id": session_id,
                "metrics": session_results[session_id]["metrics"]
            }
            for session_id in session_results
        },
        "judge_model": judge_model,
    }
    if cost_summary is not None:
        final_results["cost_summary"] = cost_summary
    
    # 保存结果
    # 1. 总体统计
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"✓ 总体统计已保存到: {summary_path}")
    
    # 2. 所有详细结果
    details_path = output_dir / "eval_details_with_llm_judge.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✓ 详细结果已保存到: {details_path}")
    
    # 3. 每个 session 的详细结果
    sessions_output_dir = output_dir / "sessions"
    sessions_output_dir.mkdir(parents=True, exist_ok=True)
    for session_id, data in session_results.items():
        session_file = sessions_output_dir / f"{session_id}_eval.json"
        with session_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 各 session 详细结果已保存到: {sessions_output_dir}")
    
    # 打印总体统计
    print("\n" + "=" * 60)
    print("总体统计:")
    print(f"  - Session 数: {final_results['overall']['num_sessions']}")
    print(f"  - 总样本数: {final_results['overall']['num_samples']}")
    print(f"  - LLM-as-Judge 准确率: {final_results['overall']['metrics']['llm_judge']['accuracy']:.1%}")
    print(f"    (CORRECT: {final_results['overall']['metrics']['llm_judge']['correct']}, "
          f"WRONG: {final_results['overall']['metrics']['llm_judge']['wrong']}, "
          f"ERROR: {final_results['overall']['metrics']['llm_judge']['error']})")
    print(f"  - F1 平均: {final_results['overall']['metrics']['f1_avg']:.4f}")
    print(f"  - BLEU1 平均: {final_results['overall']['metrics']['bleu1_avg']:.4f}")
    print(f"  - 使用模型: {judge_model}")
    
    # 打印按 category 的统计
    if by_category_metrics:
        print("\n按 Category 统计:")
        for cat in sorted(by_category_metrics.keys(), key=lambda x: str(x)):
            metrics = by_category_metrics[cat]
            print(f"  Category {cat}:")     
            print(f"    - 样本数: {metrics['num_samples']}")
            print(f"    - LLM-as-Judge 准确率: {metrics['llm_judge']['accuracy']:.1%}")
            print(f"    - F1 平均: {metrics['f1_avg']:.4f}")
            print(f"    - BLEU1 平均: {metrics['bleu1_avg']:.4f}")
    
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="对指定实验目录下的 sessions 进行 LLM-as-Judge 评分")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./results/locomo/linked_view/test_run",
        help="实验根目录路径（内部自动使用 root_dir/sessions 和 root_dir/results 子目录）"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM-as-Judge 使用的模型"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="批次大小（用于进度显示）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="并行处理的工作线程数"
    )
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="不加载已有结果，覆盖之前跑过的，全部重新跑一遍"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="如果指定，只处理匹配的 session_id（例如 conv-42）"
    )
    
    args = parser.parse_args()
    
    # 现在命令行只需要传一个实验根目录，代码内部自动拼接子目录
    root_dir = Path(args.root_dir)
    sessions_dir = root_dir / "sessions"
    output_dir = root_dir / "results"
    
    if not sessions_dir.exists():
        print(f"错误: sessions 目录不存在: {sessions_dir}")
        return
    
    evaluate_sessions(
        sessions_dir=sessions_dir,
        output_dir=output_dir,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        reuse_existing=not args.no_reuse_existing,
        session_id_filter=args.session_id
    )


if __name__ == "__main__":
    main()

