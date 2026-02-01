#!/usr/bin/env python3
"""
准备 GRPO 训练数据 (改进版)

功能:
1. 过滤 s 和 r 都错的数据
2. 使用 LLM judge 重新评估 s_path_correct
3. 检查 ground truth 是否在 summaries 中（字符串匹配）
4. 检查 ground truth 是否是无效答案（"not provided" 等）
5. 统计标记前后的区别
6. 平衡数据
7. 可选: LLM 改写问题
# 使用默认并发数（10 个 worker）
python scripts/router_training/3_prepare_grpo_data.py \
    --offline-data data/router_offline/ \
    --output-dir data/router_grpo_v6/ \
    --use-llm-judge \
    --num-workers 10

# 使用更多并发（加快速度，但注意 API 限流）
python scripts/router_training/3_prepare_grpo_data.py \
    --offline-data data/router_offline/ \
    --output-dir data/router_grpo_v6/ \
    --use-llm-judge \
    --num-workers 20

# 串行处理（不使用并发）
python scripts/router_training/3_prepare_grpo_data.py \
    --offline-data data/router_offline/ \
    --output-dir data/router_grpo_v6/ \
    --use-llm-judge \
    --num-workers 1
Usage:
    python scripts/router_training/3_prepare_grpo_data.py \
        --offline-data data/router_offline/ \
        --output-dir data/router_grpo_v6/ \
        --use-llm-judge \
        --rewrite-questions

python scripts/router_training/filter_grpo_data.py \
    --input-dir data/router_grpo_v6/ \
    --offline-data data/router_offline/ \
    --output-dir data/router_grpo_v7/
"""

import argparse
import json
import random
import re
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: 简单的进度显示
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(f"{desc}...")
        return iterable

# 添加项目路径
PROJECT_ROOT = str(Path(__file__).parent.parent.parent) if 'Path' in dir() else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available, LLM judge will be disabled")


# ============== 配置 ==============

# 无效答案关键词（如果 ground truth 包含这些，s_path 算错）
INVALID_ANSWER_KEYWORDS = [
    "not provided",
    "not available",
    "no information",
    "cannot be determined",
    "unknown",
    "n/a",
    "none",
    "未提供",
    "无法确定",
]

# LLM judge 模型
JUDGE_MODEL = "gpt-4.1-mini"
REWRITE_MODEL = "gpt-4.1-mini"


# ============== Prompt 模板 ==============

ROUTER_PROMPT_TEMPLATE = """You are an expert router for a memory-augmented QA system.

Your task: Analyze the retrieved summaries and decide the best action to answer the question.

Available actions:
1. "S" - Answer using current summaries only 
    Use when: Summaries contain the exact answer linked directly to the question's specific constraints 
    Do not infer, just based on the original data.

2. "QUERY" - Search again with a better query
   Use when: Summaries have related info but miss the specific answer
   Include a more targeted query that might find the answer

3. "R" - Deep research mode (slow path)
   Use when: Summaries don't contain the answer OR are insufficient for a confident answer
   Prefer R over (QUERY and S) when unsure - R is more reliable


Question: {question}

Retrieved Summaries:
{summaries_block}

Output format (JSON only):
- If answering with summaries: {{"action": "S"}}
- If need better search: {{"action": "QUERY", "query": "your refined search query"}}
- If deep research needed: {{"action": "R"}}

Your response:"""


LLM_JUDGE_PROMPT = """You are evaluating whether a QA system can correctly answer a question using retrieved memory summaries.

Question: {question}

Retrieved Summaries:
{summaries_text}

Ground Truth Answer(s): {gt_text}

Can the question be correctly answered using ONLY the retrieved summaries above?
Consider:
1. Do the summaries contain the key information needed?
2. Is the ground truth answer (or equivalent) derivable from the summaries?
3. Are there any missing critical details that would make the answer incomplete or incorrect?

Answer with ONLY "YES" or "NO"."""


QUESTION_REWRITE_PROMPT = """Rewrite the following question in a different way while keeping the same meaning and intent.

Original question: {question}

Provide 2-3 alternative phrasings of the question. Each should:
- Ask the same thing but use different words
- Maintain the same level of specificity
- Be clear and unambiguous

Return as a JSON array of strings, e.g., ["question 1", "question 2", "question 3"]"""


# ============== 数据加载 ==============

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    samples = []
    if not path.exists():
        return samples

    files = [path] if path.is_file() else list(path.glob("*.jsonl"))

    for f in files:
        if f.name in ("all.jsonl", "stats.json"):
            continue
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return samples


def load_as_dict(path: Path, key_field: str = "query_id") -> Dict[str, Dict]:
    """加载 JSONL 为字典 (按 key_field 索引)"""
    samples = load_jsonl(path)
    return {s.get(key_field, ""): s for s in samples if s.get(key_field)}


# ============== Ground Truth 检查 ==============

def is_invalid_answer(ground_truth: Any) -> bool:
    """检查 ground truth 是否是无效答案"""
    if not ground_truth:
        return True
    
    # 转换为字符串
    if isinstance(ground_truth, list):
        gt_text = " ".join(str(g) for g in ground_truth).lower()
    else:
        gt_text = str(ground_truth).lower()
    
    # 检查是否包含无效关键词
    for keyword in INVALID_ANSWER_KEYWORDS:
        if keyword in gt_text:
            return True
    
    return False


def check_gt_in_summaries(ground_truth: Any, summaries: List[str]) -> bool:
    """检查 ground truth 是否在 summaries 中（简单字符串匹配）"""
    if not ground_truth or not summaries:
        return False
    
    # 确保 ground_truth 是列表
    if isinstance(ground_truth, str):
        gt_list = [ground_truth]
    elif isinstance(ground_truth, list):
        gt_list = ground_truth
    else:
        gt_list = [str(ground_truth)]
    
    # 将所有 summaries 合并为文本
    summaries_text = " ".join(summaries).lower()
    
    # 检查每个 ground truth 是否在 summaries 中
    for gt in gt_list:
        gt_clean = str(gt).strip().lower()
        if not gt_clean:
            continue
        
        # 简单字符串匹配
        if gt_clean in summaries_text:
            return True
        
        # 检查关键词匹配（如果 ground truth 较长，检查关键部分）
        gt_words = gt_clean.split()
        if len(gt_words) > 3:
            # 对于较长的答案，检查是否包含关键部分
            key_phrases = [gt_words[i:i+3] for i in range(len(gt_words)-2)]
            for phrase in key_phrases:
                phrase_text = " ".join(phrase)
                if phrase_text in summaries_text:
                    return True
    
    return False


# ============== LLM Judge ==============

def get_openai_client() -> Optional[Any]:
    """获取 OpenAI 客户端"""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, LLM judge will be disabled")
        return None
    
    return OpenAI(api_key=api_key)


def llm_judge_s_path(
    client: Optional[Any],
    question: str,
    summaries: List[str],
    ground_truth: List[str],
    use_cache: bool = True,
    cache: Optional[Dict[str, bool]] = None,
    cache_lock: Optional[threading.Lock] = None
) -> Tuple[bool, Optional[str]]:
    """使用 LLM judge 评估 s_path_correct（线程安全版本）
    
    Returns:
        (is_correct, error_message)
    """
    if not client:
        return False, "OpenAI client not available"
    
    # 构建 prompt
    summaries_text = "\n".join([f"- {s}" for s in summaries])
    gt_text = ", ".join(ground_truth[:3]) if ground_truth else "(no ground truth)"
    
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        summaries_text=summaries_text,
        gt_text=gt_text
    )
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        result = answer == "YES"
        print(f"LLM judge result: {result}")
        print(f"LLM judge prompt: {prompt}")
        print(f"LLM judge answer: {answer}")
        print("=============================")
        return result, None
    except Exception as e:
        return False, str(e)


def rewrite_question(client: Optional[Any], question: str) -> List[str]:
    """使用 LLM 改写问题"""
    if not client:
        return []
    
    prompt = QUESTION_REWRITE_PROMPT.format(question=question)
    
    try:
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # 尝试不同的键名
        if "questions" in result:
            return result["questions"]
        elif "alternatives" in result:
            return result["alternatives"]
        elif isinstance(result, list):
            return result
        else:
            # 如果返回的不是预期格式，尝试解析
            return []
    except Exception as e:
        print(f"Error rewriting question: {e}")
        return []


# ============== 样本评估 ==============

def evaluate_s_path_correct(
    sample: Dict[str, Any],
    client: Optional[Any] = None,
    use_llm_judge: bool = False,
    judge_cache: Optional[Dict[str, bool]] = None,
    cache_lock: Optional[threading.Lock] = None
) -> Tuple[bool, Dict[str, Any]]:
    """评估 s_path_correct（线程安全版本）
    
    Returns:
        (s_path_correct, metadata)
    """
    metadata = {
        "original_s_path_correct": sample.get("s_path_correct", False),
        "checks": {}
    }
    
    question = sample.get("question", "")
    summaries = sample.get("original_summaries", [])
    ground_truth = sample.get("ground_truth", [])
    
    # 确保 ground_truth 是列表
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    elif not isinstance(ground_truth, list):
        ground_truth = [str(ground_truth)]
    # 检查 3: LLM judge（如果启用）
    if use_llm_judge and client:
        llm_result, error = llm_judge_s_path(
            client, question, summaries, ground_truth,
            use_cache=True, cache=judge_cache, cache_lock=cache_lock
        )

        metadata["checks"]["llm_judge"] = llm_result
        metadata["checks"]["llm_judge_error"] = error
        if not llm_result:
            return False, metadata
    
    # 所有检查通过
    return True, metadata


# ============== Prompt 构建 ==============

def build_router_prompt(sample: Dict[str, Any]) -> str:
    """构建 Router prompt"""
    question = sample.get("question", "")
    summaries = sample.get("original_summaries", [])
    scores = sample.get("original_scores", [])
    
    # 构建 summaries block
    lines = []
    for i, (summary, score) in enumerate(zip(summaries[:5], scores[:5])):
        lines.append(f"[{i}] score={score:.3f}\n{summary}")
    summaries_block = "\n\n".join(lines) if lines else "(no summaries retrieved)"
    
    return ROUTER_PROMPT_TEMPLATE.format(
        question=question,
        summaries_block=summaries_block
    )


def get_sample_type(sample: Dict[str, Any]) -> str:
    """获取样本类型: s_ok_r_ok / s_ok_r_fail / s_fail_r_ok / s_fail_r_fail"""
    s = sample.get("s_path_correct", False)
    r = sample.get("r_path_correct", False)
    return f"{'s_ok' if s else 's_fail'}_{'r_ok' if r else 'r_fail'}"


# ============== 样本准备 ==============

def prepare_grpo_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """准备单个 GRPO 样本 (ms-swift messages 格式)"""
    prompt = build_router_prompt(sample)
    
    # 确保 ground_truth 是列表
    gt = sample.get("ground_truth", [])
    if isinstance(gt, str):
        gt = [gt]
    
    return {
        # ms-swift GRPO 需要 messages 格式
        "messages": [{"role": "user", "content": prompt}],
        # reward 函数需要的字段
        "query_id": sample.get("query_id", ""),
        "question": sample.get("question", ""),
        "ground_truth": gt,
        "original_summaries": sample.get("original_summaries", []),
        "original_scores": sample.get("original_scores", []),
        "s_path_correct": sample.get("s_path_correct", False),
        "r_path_correct": sample.get("r_path_correct", False),
    }


# ============== 数据平衡 ==============

def balance_data(
    samples: List[Dict[str, Any]],
    target_ratios: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """平衡数据分布
    
    Args:
        samples: 样本列表
        target_ratios: 目标比例，例如 {"s_ok_r_ok": 0.3, "s_ok_r_fail": 0.1, "s_fail_r_ok": 0.4, "s_fail_r_fail": 0.2}
    """
    if not target_ratios:
        # 默认：不过度平衡，只做轻微调整
        return samples
    
    # 按类型分组
    by_type = {"s_ok_r_ok": [], "s_ok_r_fail": [], "s_fail_r_ok": [], "s_fail_r_fail": []}
    for sample in samples:
        sample_type = get_sample_type(sample)
        if sample_type in by_type:
            by_type[sample_type].append(sample)
    
    # 计算目标数量
    total = len(samples)
    target_counts = {k: int(total * v) for k, v in target_ratios.items()}
    
    # 平衡数据
    balanced = []
    for sample_type, target_count in target_counts.items():
        available = by_type.get(sample_type, [])
        if len(available) >= target_count:
            # 随机选择
            balanced.extend(random.sample(available, target_count))
        else:
            # 全部使用，可能需要过采样
            balanced.extend(available)
            # 如果需要过采样
            if len(available) > 0:
                needed = target_count - len(available)
                balanced.extend(random.choices(available, k=needed))
    
    random.shuffle(balanced)
    return balanced


# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO training data (improved)")
    parser.add_argument("--offline-data", type=str, required=True, help="离线数据目录/文件")
    parser.add_argument("--distilled-data", type=str, default=None, help="蒸馏数据目录/文件")
    parser.add_argument("--augmented-data", type=str, default=None, help="增强数据 (问题改写)")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--oversample-ratio", type=int, default=5, help="s_fail_r_ok 过采样倍数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use-llm-judge", action="store_true", help="使用 LLM judge 评估 s_path_correct")
    parser.add_argument("--rewrite-questions", action="store_true", help="使用 LLM 改写问题（数据增强）")
    parser.add_argument("--max-rewrites", type=int, default=2, help="每个问题最多改写次数")
    parser.add_argument("--num-workers", type=int, default=10, help="LLM judge 并发数（默认 10）")
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 OpenAI 客户端
    client = None
    if args.use_llm_judge or args.rewrite_questions:
        client = get_openai_client()
        if not client:
            print("Warning: Cannot initialize OpenAI client, disabling LLM features")
            args.use_llm_judge = False
            args.rewrite_questions = False
    
    judge_cache = {}
    cache_lock = threading.Lock() if args.use_llm_judge else None
    
    # 1. 加载数据
    print("=== Loading Data ===")
    offline_samples = load_jsonl(Path(args.offline_data))
    print(f"  Offline: {len(offline_samples)} samples")
    
    distilled_dict = {}
    if args.distilled_data:
        distilled_dict = load_as_dict(Path(args.distilled_data))
        print(f"  Distilled: {len(distilled_dict)} samples")
    
    augmented_samples = []
    if args.augmented_data:
        augmented_samples = load_jsonl(Path(args.augmented_data))
        print(f"  Augmented: {len(augmented_samples)} samples")
    
    # 2. 合并离线数据和增强数据
    all_samples = offline_samples + augmented_samples
    
    # 3. 统计原始分布
    print("\n=== Original Sample Distribution ===")
    original_by_type = {"s_ok_r_ok": [], "s_ok_r_fail": [], "s_fail_r_ok": [], "s_fail_r_fail": []}
    for sample in all_samples:
        sample_type = get_sample_type(sample)
        if sample_type in original_by_type:
            original_by_type[sample_type].append(sample)
    
    for st, samples in original_by_type.items():
        print(f"  {st}: {len(samples)}")
    
    # 4. 重新评估 s_path_correct（支持并发）
    print("\n=== Re-evaluating s_path_correct ===")
    if args.use_llm_judge:
        print(f"  Using {args.num_workers} workers for concurrent LLM judge evaluation")
    
    evaluation_stats = {
        "total": 0,
        "unchanged": 0,
        "changed_to_false": 0,
        "changed_to_true": 0,
        "invalid_answer": 0,
        "gt_not_in_summaries": 0,
        "llm_judge_false": 0,
    }
    
    def evaluate_sample(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """评估单个样本（用于并发）"""
        original_s_correct = sample.get("s_path_correct", False)
        
        # 重新评估
        new_s_correct, metadata = evaluate_s_path_correct(
            sample,
            client=client,
            use_llm_judge=args.use_llm_judge,
            judge_cache=judge_cache,
            cache_lock=cache_lock
        )
        
        # 更新样本
        sample["s_path_correct"] = new_s_correct
        sample["_evaluation_metadata"] = metadata
        
        # 返回统计信息
        stats_update = {
            "unchanged": 0,
            "changed_to_false": 0,
            "changed_to_true": 0,
            "invalid_answer": 0,
            "gt_not_in_summaries": 0,
            "llm_judge_false": 0,
        }
        
        if new_s_correct == original_s_correct:
            stats_update["unchanged"] = 1
        elif new_s_correct:
            stats_update["changed_to_true"] = 1
        else:
            stats_update["changed_to_false"] = 1
        
        # 统计失败原因
        checks = metadata.get("checks", {})
        if checks.get("is_invalid_answer"):
            stats_update["invalid_answer"] = 1
        if not checks.get("gt_in_summaries"):
            stats_update["gt_not_in_summaries"] = 1
        if args.use_llm_judge and not checks.get("llm_judge", True):
            stats_update["llm_judge_false"] = 1
        
        return sample, stats_update
    
    # 并发评估（如果使用 LLM judge）
    if args.use_llm_judge and args.num_workers > 1:
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(evaluate_sample, sample): sample 
                for sample in all_samples
            }
            
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(future_to_sample), total=len(all_samples), desc="Evaluating"):
                try:
                    sample, stats_update = future.result()
                    evaluation_stats["total"] += 1
                    for key, value in stats_update.items():
                        evaluation_stats[key] += value
                except Exception as e:
                    print(f"Error evaluating sample: {e}")
                    evaluation_stats["total"] += 1
    else:
        # 串行处理（不使用 LLM judge 或 num_workers=1）
        for sample in tqdm(all_samples, desc="Evaluating"):
            evaluation_stats["total"] += 1
            original_s_correct = sample.get("s_path_correct", False)
            
            # 重新评估
            new_s_correct, metadata = evaluate_s_path_correct(
                sample,
                client=client,
                use_llm_judge=args.use_llm_judge,
                judge_cache=judge_cache,
                cache_lock=cache_lock
            )
            
            # 更新样本
            sample["s_path_correct"] = new_s_correct
            sample["_evaluation_metadata"] = metadata
            
            # 统计
            if new_s_correct == original_s_correct:
                evaluation_stats["unchanged"] += 1
            elif new_s_correct:
                evaluation_stats["changed_to_true"] += 1
            else:
                evaluation_stats["changed_to_false"] += 1
            
            # 统计失败原因
            checks = metadata.get("checks", {})
            if checks.get("is_invalid_answer"):
                evaluation_stats["invalid_answer"] += 1
            if not checks.get("gt_in_summaries"):
                evaluation_stats["gt_not_in_summaries"] += 1
            if args.use_llm_judge and not checks.get("llm_judge", True):
                evaluation_stats["llm_judge_false"] += 1
    
    print(f"\n  Total evaluated: {evaluation_stats['total']}")
    print(f"  Unchanged: {evaluation_stats['unchanged']}")
    print(f"  Changed to False: {evaluation_stats['changed_to_false']}")
    print(f"  Changed to True: {evaluation_stats['changed_to_true']}")
    print(f"  Invalid answer: {evaluation_stats['invalid_answer']}")
    print(f"  GT not in summaries: {evaluation_stats['gt_not_in_summaries']}")
    if args.use_llm_judge:
        print(f"  LLM judge False: {evaluation_stats['llm_judge_false']}")
    
    # 5. 过滤数据
    print("\n=== Filtering Samples ===")
    print("  Filtering rules:")
    print("    1. Remove s_fail_r_fail (both paths wrong)")
    print("    2. Remove r_fail (r_path_correct=False)")
    print("    3. Keep only consistent samples (original_s_path_correct == new_s_path_correct)")
    
    filtered_samples = []
    filter_stats = {
        "total": len(all_samples),
        "removed_s_fail_r_fail": 0,
        "removed_r_fail": 0,
        "removed_inconsistent": 0,
        "kept": 0,
    }
    
    for sample in all_samples:
        # 检查 1: 过滤 s 和 r 都错的数据
        sample_type = get_sample_type(sample)
        if sample_type == "s_fail_r_fail":
            filter_stats["removed_s_fail_r_fail"] += 1
            continue
        
        # 检查 2: 过滤 r_fail 的数据
        if not sample.get("r_path_correct", False):
            filter_stats["removed_r_fail"] += 1
            continue
        
        # 检查 3: 只保留前后一致的数据
        metadata = sample.get("_evaluation_metadata", {})
        original_s_correct = metadata.get("original_s_path_correct", sample.get("s_path_correct", False))
        new_s_correct = sample.get("s_path_correct", False)
        
        if original_s_correct != new_s_correct:
            filter_stats["removed_inconsistent"] += 1
            continue
        
        # 通过所有检查
        filtered_samples.append(sample)
        filter_stats["kept"] += 1
    
    print(f"\n  Before filtering: {filter_stats['total']}")
    print(f"  Removed s_fail_r_fail: {filter_stats['removed_s_fail_r_fail']}")
    print(f"  Removed r_fail: {filter_stats['removed_r_fail']}")
    print(f"  Removed inconsistent: {filter_stats['removed_inconsistent']}")
    print(f"  Kept (consistent): {filter_stats['kept']}")
    print(f"  Total removed: {filter_stats['total'] - filter_stats['kept']}")
    
    # 6. 问题改写（数据增强）
    if args.rewrite_questions and client:
        print("\n=== Rewriting Questions ===")
        rewritten_samples = []
        for sample in filtered_samples[:100]:  # 限制数量，避免成本过高
            question = sample.get("question", "")
            if not question:
                continue
            
            rewrites = rewrite_question(client, question)
            for rewrite in rewrites[:args.max_rewrites]:
                new_sample = sample.copy()
                new_sample["question"] = rewrite
                new_sample["query_id"] = f"{sample.get('query_id', '')}_rewrite_{len(rewritten_samples)}"
                rewritten_samples.append(new_sample)
        
        filtered_samples.extend(rewritten_samples)
        print(f"  Added {len(rewritten_samples)} rewritten samples")
    
    # 7. 重新分类样本（基于新的 s_path_correct）
    print("\n=== New Sample Distribution ===")
    by_type = {"s_ok_r_ok": [], "s_ok_r_fail": [], "s_fail_r_ok": [], "s_fail_r_fail": []}
    
    for sample in filtered_samples:
        sample_type = get_sample_type(sample)
        if sample_type in by_type:
            by_type[sample_type].append(sample)
    
    for st, samples in by_type.items():
        print(f"  {st}: {len(samples)}")
    
    # 8. 构建 GRPO 样本 + 过采样
    grpo_samples = []
    
    # 添加所有有效样本
    for st, samples in by_type.items():
        for sample in samples:
            grpo_samples.append(prepare_grpo_sample(sample))
    
    # 过采样 s_fail_r_ok (关键路由决策)
    s_fail_r_ok = [prepare_grpo_sample(s) for s in by_type["s_fail_r_ok"]]
    for _ in range(args.oversample_ratio - 1):
        grpo_samples.extend(s_fail_r_ok)
    
    print(f"\n=== After Oversampling ===")
    print(f"  Total: {len(grpo_samples)} samples")
    print(f"  s_fail_r_ok oversampled {args.oversample_ratio}x")
    
    # 9. 拆分 train/val
    random.shuffle(grpo_samples)
    val_size = int(len(grpo_samples) * args.val_ratio)
    val_samples = grpo_samples[:val_size]
    train_samples = grpo_samples[val_size:]
    
    print(f"\n=== Train/Val Split ===")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    
    # 10. 保存
    train_path = output_dir / "train_grpo.jsonl"
    val_path = output_dir / "val_grpo.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            # 移除评估元数据（训练不需要）
            s_clean = {k: v for k, v in s.items() if not k.startswith("_")}
            f.write(json.dumps(s_clean, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            s_clean = {k: v for k, v in s.items() if not k.startswith("_")}
            f.write(json.dumps(s_clean, ensure_ascii=False) + "\n")
    
    # 11. 统计
    train_s_correct = sum(1 for s in train_samples if s["s_path_correct"])
    train_r_only = sum(1 for s in train_samples if not s["s_path_correct"] and s["r_path_correct"])
    
    stats = {
        "total_train": len(train_samples),
        "total_val": len(val_samples),
        "train_s_correct": train_s_correct,
        "train_r_only_correct": train_r_only,
        "oversample_ratio": args.oversample_ratio,
        "evaluation_stats": evaluation_stats,
        "filter_stats": filter_stats,
        "original_distribution": {k: len(v) for k, v in original_by_type.items()},
        "new_distribution": {k: len(v) for k, v in by_type.items()},
        "use_llm_judge": args.use_llm_judge,
        "rewrite_questions": args.rewrite_questions,
    }
    
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n=== Saved ===")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Stats: {output_dir / 'stats.json'}")
    
    # 12. 打印对比统计
    print(f"\n=== Comparison: Before vs After ===")
    print("Original distribution:")
    for st, count in stats["original_distribution"].items():
        print(f"  {st}: {count}")
    print("New distribution:")
    for st, count in stats["new_distribution"].items():
        print(f"  {st}: {count}")


if __name__ == "__main__":
    main()
