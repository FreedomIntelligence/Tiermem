#!/usr/bin/env python3
"""
准备 SFT 训练数据（V2 版本 - S/R 二分类，适配 Thinking 模型）

数据来源：
- data/router_distilled_v2/*.jsonl（增强后的数据，包含 gpt5_decision_v2 和 optimal_label_v2）

训练标签策略：
- 使用 optimal_label_v2 作为 ground truth（S/R 二分类）
- 旧数据中的 QUERY 标签会被映射到 R
- 使用 gpt5_decision_v2 的思维链生成 <think>...</think> 格式

输出格式：
- messages 格式：[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
- response 格式：<think>思考过程</think>\n\n{"action": "S/R"}

Usage:
python scripts/router_training/2_prepare_sft_data_v2.py \
    --input-dir data/router_distilled_s_r/ \
    --output data/router_sft_s_r/train.jsonl \
    --label-field optimal_label_v2 \
    --only-quality-thinking \
    --balance
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter


# ============ V2 Router Prompt（S/R 二分类）============

ROUTER_PROMPT_V2 = """You are an expert router for a memory-augmented QA system.

Your task: Analyze the retrieved summaries and decide the best action to answer the question.

Available actions:
1. "S" - Answer using current summaries only
    Use when: Summaries contain the EXPLICIT answer to the specific question.
    CRITICAL: Do not infer causes from effects (e.g., "benefits of X" is NOT "reason for starting X").
    If the exact answer is not verbatim in the text, or the question needs efficient details do not use S.

2. "R" - Deep research mode (slow path)
    Use when: Summaries are ambiguous.
    If the question requires completeness (e.g., "how many times", "list all", "what are all", "Both of them"), prefer R to ensure comprehensive answers.


Question: {question}

Retrieved Summaries:
{summaries_block}

Output format (JSON only):
- If answering with summaries: {{"action": "S"}}
- If deep research needed: {{"action": "R"}}

Your response:"""


def load_v2_data(data_path: Path) -> List[Dict[str, Any]]:
    """
    加载 V2 增强数据

    Args:
        data_path: 文件或目录路径

    Returns:
        样本列表
    """
    samples = []

    if data_path.is_file():
        files = [data_path]
    else:
        files = list(data_path.glob("*.jsonl"))

    for f in files:
        # 跳过 stats 和其他非数据文件
        if "stats" in f.name or "cache" in f.name:
            continue
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 只保留有 V2 字段的样本
                        if "gpt5_decision_v2" in data or "optimal_label_v2" in data:
                            samples.append(data)
                    except json.JSONDecodeError:
                        continue

    return samples


def build_router_prompt_v2(sample: Dict[str, Any]) -> str:
    """
    构建 V2 Router 的输入 prompt

    使用与 run_gpt5_router_experiment_v2.py 一致的 prompt 格式
    """
    question = sample.get("question", "")

    # 获取 summaries 和 scores
    # 兼容两种字段名
    summaries = sample.get("summaries") or sample.get("original_summaries", [])
    scores = sample.get("scores") or sample.get("original_scores", [])

    # 构建 summaries block
    summaries_lines = []
    for i, (summary, score) in enumerate(zip(summaries[:5], scores[:5])):
        summaries_lines.append(f"[{i}] score={score:.3f}\n{summary}")

    summaries_block = "\n\n".join(summaries_lines) if summaries_lines else "(no summaries retrieved)"

    return ROUTER_PROMPT_V2.format(
        question=question,
        summaries_block=summaries_block,
    )


def build_target_output_v2(sample: Dict[str, Any], optimal_action: str = None) -> str:
    """
    构建 V2 目标输出（适配 Thinking 模型格式 - S/R 二分类）

    Args:
        sample: 原始样本数据
        optimal_action: ground truth action，如果为 None 则从 sample 读取

    格式：<think>连贯的思考过程</think>\n\n{"action": "S/R"}
    """
    # 获取 ground truth action
    if optimal_action is None:
        optimal_action = sample.get("optimal_label_v2", "R")

    # 确保 optimal_action 只能是 S 或 R
    if optimal_action not in ("S", "R"):
        optimal_action = "R"

    # 获取 GPT 决策（优先 v2，fallback 到原始 gpt5_decision）
    gpt5_v2 = sample.get("gpt5_decision_v2") or sample.get("gpt5_decision") or {}

    # 获取原始 GPT 决策的 action
    gpt_action = gpt5_v2.get("action", "")
    if isinstance(gpt_action, str):
        gpt_action = gpt_action.upper().strip()
        # 将 QUERY 映射到 R
        if gpt_action == "QUERY":
            gpt_action = "R"

    # 获取 question 和 summaries 用于生成 thinking
    question = sample.get("question", "")
    summaries = sample.get("summaries") or sample.get("original_summaries", [])

    # 构建连贯的 thinking 内容
    thinking_text = ""

    # 如果 GPT 决策与 optimal_action 一致，使用其 thinking
    if gpt5_v2 and gpt_action == optimal_action:
        # 新格式：thinking 是字符串
        if isinstance(gpt5_v2.get("thinking"), str) and gpt5_v2.get("thinking"):
            thinking_text = gpt5_v2["thinking"]
        # 旧格式：thinking 是嵌套对象，需要转换
        elif isinstance(gpt5_v2.get("thinking"), dict):
            thinking = gpt5_v2["thinking"]
            parts = []
            if thinking.get("question_needs"):
                parts.append(f"The question asks for {thinking['question_needs']}.")
            if thinking.get("summaries_have"):
                parts.append(f"The summaries contain {thinking['summaries_have']}.")
            if thinking.get("coverage"):
                coverage = thinking['coverage']
                if coverage == "EXPLICIT":
                    parts.append("The answer is explicitly stated in the summaries.")
                elif coverage == "PARTIAL":
                    parts.append("The summaries have partial clues but not the direct answer.")
                else:
                    parts.append(f"The summaries have {coverage.lower()} coverage.")
            if thinking.get("red_flags"):
                flags = thinking['red_flags']
                if flags:
                    parts.append(f"However, there are concerns: {', '.join(flags)}.")
            thinking_text = " ".join(parts)
        # 兼容旧的分字段格式
        elif gpt5_v2.get("question_needs"):
            parts = []
            parts.append(f"The question asks for {gpt5_v2['question_needs']}.")
            if gpt5_v2.get("summaries_have"):
                parts.append(f"The summaries contain {gpt5_v2['summaries_have']}.")
            coverage = gpt5_v2.get("coverage", "")
            if coverage == "EXPLICIT":
                parts.append("The answer is explicitly stated in the summaries.")
            elif coverage == "PARTIAL":
                parts.append("The summaries have partial clues but not the direct answer.")
            elif coverage:
                parts.append(f"The summaries have {coverage.lower()} coverage.")
            red_flags = gpt5_v2.get("red_flags", [])
            if red_flags:
                parts.append(f"However, there are concerns: {', '.join(red_flags)}.")
            thinking_text = " ".join(parts)

    # 如果没有可用的 thinking，根据 coverage_judgement 和 optimal_action 生成
    if not thinking_text:
        coverage_judge = sample.get("coverage_judgement", {})
        reason = coverage_judge.get("reason", "")

        if optimal_action == "S":
            thinking_text = f"The question asks about specific information. Looking at the summaries, the answer is explicitly stated. {reason} Since the summaries directly contain the answer, I should use them."
        else:  # R
            thinking_text = f"The question needs specific information. However, the summaries don't contain sufficient information to answer this. {reason} Deep research is needed to find the complete answer."

    # 添加决策说明
    if optimal_action == "S":
        thinking_text += " Therefore, I'll answer using the summaries directly."
    else:
        thinking_text += " Therefore, deep research mode is needed."

    # 构建输出 JSON
    json_output = json.dumps({"action": optimal_action}, ensure_ascii=False)

    # 组合成 thinking 模型格式
    return f"<think>\n{thinking_text}\n</think>\n\n{json_output}"


def prepare_sft_sample_v2(
    sample: Dict[str, Any],
    label_field: str = "optimal_label_v2",
) -> Optional[Dict[str, Any]]:
    """
    准备单个 V2 SFT 训练样本（S/R 二分类）

    Args:
        sample: 原始样本数据
        label_field: 使用哪个标签字段 (optimal_label_v2 / optimal_label_v3)

    Returns:
        {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], "metadata": dict}
    """
    # 检查必要字段
    summaries = sample.get("summaries") or sample.get("original_summaries", [])
    if not summaries:
        return None

    # 获取标签（支持多个字段名）
    optimal_label = sample.get(label_field)
    if not optimal_label:
        # fallback 顺序
        for fallback in ["optimal_label_v2", "optimal_label", "optimal_label_v3"]:
            optimal_label = sample.get(fallback)
            if optimal_label:
                break
    if not optimal_label:
        return None

    # 确保标签只能是 S 或 R（将 QUERY 映射到 R）
    if optimal_label not in ("S", "R"):
        optimal_label = "R"

    # 构建 prompt
    prompt = build_router_prompt_v2(sample)

    # 构建 response（需要传入正确的 label）
    response = build_target_output_v2(sample, optimal_label)

    # 检查是否有高质量的 thinking（GPT 决策与 optimal_label 一致，且有 thinking 内容）
    gpt5_decision = sample.get("gpt5_decision_v2") or sample.get("gpt5_decision") or {}
    has_quality_thinking = False

    # 获取并规范化 action
    gpt_action = gpt5_decision.get("action", "")
    if isinstance(gpt_action, str):
        gpt_action = gpt_action.upper().strip()
        # 将 QUERY 映射到 R
        if gpt_action == "QUERY":
            gpt_action = "R"

    if gpt_action == optimal_label:
        # 原始决策与 optimal_label 一致的情况
        # thinking 是字符串
        if isinstance(gpt5_decision.get("thinking"), str) and gpt5_decision.get("thinking"):
            has_quality_thinking = True
        # thinking 是嵌套对象
        elif isinstance(gpt5_decision.get("thinking"), dict) and gpt5_decision["thinking"]:
            has_quality_thinking = True
        # 旧格式有 question_needs 字段
        elif gpt5_decision.get("question_needs"):
            has_quality_thinking = True

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "query_id": sample.get("query_id", ""),
            "session_id": sample.get("session_id", ""),
            "action": optimal_label,
            "coverage_has_info": sample.get("coverage_judgement", {}).get("has_sufficient_info", False),
            "has_gpt5_v2": "gpt5_decision_v2" in sample or "gpt5_decision" in sample,
            "has_quality_thinking": has_quality_thinking,
            "benchmark": sample.get("benchmark", "unknown"),
            "sample_type": sample.get("sample_type", "unknown"),
            "label_field": label_field,
        }
    }


def balance_dataset(samples: List[Dict[str, Any]], target_ratio: float = 1.0) -> List[Dict[str, Any]]:
    """
    平衡数据集，通过过采样少数类（S/R 二分类）

    target_ratio: 目标比例，1.0 表示所有类别数量相同
    """
    # 按 action 分组
    by_action = {"S": [], "R": []}
    for sample in samples:
        action = sample["metadata"]["action"]
        if action in by_action:
            by_action[action].append(sample)

    s_count = len(by_action["S"])
    r_count = len(by_action["R"])

    print(f"Original distribution: S={s_count}, R={r_count}")

    # 计算目标数量（以最大类为基准）
    max_count = max(s_count, r_count)

    balanced = []
    for action, group in by_action.items():
        if len(group) == 0:
            continue
        # 过采样到最大类的数量
        if len(group) < max_count:
            oversampled = group * (max_count // len(group)) + random.sample(group, max_count % len(group))
            balanced.extend(oversampled)
        else:
            balanced.extend(group)

    random.shuffle(balanced)

    # 统计平衡后的分布
    new_counts = Counter(s["metadata"]["action"] for s in balanced)
    print(f"Balanced distribution: {dict(new_counts)}")

    return balanced


def filter_quality_thinking(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤只保留高质量 thinking 的样本
    （gpt5_decision_v2 的 action 与 optimal_label_v2 一致，且有完整的 thinking）
    """
    filtered = []
    for sample in samples:
        if sample["metadata"].get("has_quality_thinking", False):
            filtered.append(sample)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data (V2 - S/R Binary Classification)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/router_distilled_v2",
        help="Path to V2 distilled data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/router_sft_v2/train.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance the dataset by oversampling minority classes"
    )
    parser.add_argument(
        "--only-quality-thinking",
        action="store_true",
        help="Only include samples where gpt5_v2 action matches optimal_label_v2"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train/val split ratio (default 0.9 means 90%% train, 10%% val)"
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="optimal_label_v2",
        choices=["optimal_label_v2", "optimal_label_v3"],
        help="Which label field to use as ground truth"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print(f"Settings:")
    print(f"  Label field: {args.label_field}")
    print(f"  Mode: S/R binary classification (QUERY removed)")

    # 加载数据
    input_path = Path(args.input_dir)
    print(f"\nLoading V2 data from: {input_path}")
    raw_samples = load_v2_data(input_path)
    print(f"  Loaded {len(raw_samples)} samples")

    # 构建 SFT 样本
    sft_samples = []
    stats = {
        "total": 0,
        "skipped_no_summaries": 0,
        "skipped_no_label": 0,
        "actions": Counter(),
        "quality_thinking": 0,
        "by_benchmark": Counter(),
    }

    for sample in raw_samples:
        sft_sample = prepare_sft_sample_v2(sample, label_field=args.label_field)

        if sft_sample is None:
            if not (sample.get("summaries") or sample.get("original_summaries")):
                stats["skipped_no_summaries"] += 1
            else:
                stats["skipped_no_label"] += 1
            continue

        action = sft_sample["metadata"]["action"]

        sft_samples.append(sft_sample)
        stats["total"] += 1
        stats["actions"][action] += 1
        stats["by_benchmark"][sft_sample["metadata"]["benchmark"]] += 1
        if sft_sample["metadata"]["has_quality_thinking"]:
            stats["quality_thinking"] += 1

    print(f"\nGenerated {stats['total']} SFT samples")
    print(f"  With quality thinking: {stats['quality_thinking']}")
    print(f"  Skipped (no summaries): {stats['skipped_no_summaries']}")
    print(f"  Skipped (no label): {stats['skipped_no_label']}")
    print(f"  Action distribution: {dict(stats['actions'])}")
    print(f"  By benchmark: {dict(stats['by_benchmark'])}")

    # 过滤只保留高质量 thinking 的样本
    if args.only_quality_thinking:
        print("\nFiltering to only quality thinking samples...")
        sft_samples = filter_quality_thinking(sft_samples)
        print(f"  Remaining: {len(sft_samples)} samples")
        # 更新 action 统计
        action_counts = Counter(s["metadata"]["action"] for s in sft_samples)
        print(f"  Action distribution: {dict(action_counts)}")

    # 平衡数据集
    if args.balance:
        print("\nBalancing dataset...")
        sft_samples = balance_dataset(sft_samples)

    # 分割训练集和验证集
    random.shuffle(sft_samples)
    split_idx = int(len(sft_samples) * args.train_split)
    train_samples = sft_samples[:split_idx]
    val_samples = sft_samples[split_idx:]

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(train_samples)} train samples to: {output_path}")

    # 保存验证集
    val_path = output_path.parent / output_path.name.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(val_samples)} val samples to: {val_path}")

    # 保存统计信息
    final_train_counts = Counter(s["metadata"]["action"] for s in train_samples)
    final_val_counts = Counter(s["metadata"]["action"] for s in val_samples)

    stats_path = output_path.parent / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(sft_samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "train_actions": dict(final_train_counts),
            "val_actions": dict(final_val_counts),
            "balanced": args.balance,
            "only_quality_thinking": args.only_quality_thinking,
            "quality_thinking_count": stats["quality_thinking"],
            "label_field": args.label_field,
            "mode": "S/R binary classification",
        }, f, indent=2, ensure_ascii=False)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
