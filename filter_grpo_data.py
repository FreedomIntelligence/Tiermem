#!/usr/bin/env python3
"""
过滤已有的 GRPO 数据

功能:
1. 读取已有的 train_grpo.jsonl 和 val_grpo.jsonl
2. 从原始 offline 数据中获取原始的 s_path_correct
3. 过滤规则:
   - 删除 r_fail (r_path_correct=False)
   - 删除前后不一致的数据 (original_s_path_correct != new_s_path_correct)
   - 删除 s_fail_r_fail (两条路都错的)
4. 重新生成训练数据

Usage:
    python scripts/router_training/filter_grpo_data.py \
        --input-dir data/router_grpo_v6/ \
        --offline-data data/router_offline/ \
        --output-dir data/router_grpo_v7/
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter


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


def load_offline_data(offline_path: Path) -> Dict[str, Dict[str, Any]]:
    """加载原始 offline 数据，返回 {query_id: sample}"""
    samples = {}
    
    files = [offline_path] if offline_path.is_file() else list(offline_path.glob("*.jsonl"))
    
    for f in files:
        if f.name == "stats.json":
            continue
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        query_id = data.get("query_id", "")
                        if query_id:
                            samples[query_id] = data
                    except json.JSONDecodeError:
                        continue
    
    return samples


def get_sample_type(sample: Dict[str, Any]) -> str:
    """获取样本类型"""
    s = sample.get("s_path_correct", False)
    r = sample.get("r_path_correct", False)
    return f"{'s_ok' if s else 's_fail'}_{'r_ok' if r else 'r_fail'}"


def main():
    parser = argparse.ArgumentParser(description="Filter existing GRPO data")
    parser.add_argument("--input-dir", type=str, required=True, help="输入目录（包含 train_grpo.jsonl 和 val_grpo.jsonl）")
    parser.add_argument("--offline-data", type=str, required=True, help="原始 offline 数据目录/文件")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--oversample-ratio", type=int, default=1, help="s_fail_r_ok 过采样倍数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载已有数据
    print("=== Loading Existing Data ===")
    train_samples = load_jsonl(input_dir / "train_grpo.jsonl")
    val_samples = load_jsonl(input_dir / "val_grpo.jsonl")
    all_grpo_samples = train_samples + val_samples
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Total: {len(all_grpo_samples)}")
    
    # 2. 加载原始 offline 数据（获取原始的 s_path_correct）
    print("\n=== Loading Original Offline Data ===")
    offline_dict = load_offline_data(Path(args.offline_data))
    print(f"  Loaded {len(offline_dict)} offline samples")
    
    # 3. 合并数据并获取原始 s_path_correct
    print("\n=== Matching with Original Data ===")
    matched_samples = []
    unmatched = 0
    
    for sample in all_grpo_samples:
        query_id = sample.get("query_id", "")
        if query_id in offline_dict:
            original_sample = offline_dict[query_id]
            # 添加原始的 s_path_correct
            sample["_original_s_path_correct"] = original_sample.get("s_path_correct", False)
            matched_samples.append(sample)
        else:
            unmatched += 1
            # 如果没有匹配到，假设原始值就是当前值（保守处理）
            sample["_original_s_path_correct"] = sample.get("s_path_correct", False)
            matched_samples.append(sample)
    
    if unmatched > 0:
        print(f"  Warning: {unmatched} samples not found in offline data, using current value as original")
    
    print(f"  Matched: {len(matched_samples)}")
    
    # 4. 统计原始分布
    print("\n=== Original Distribution (Before Filtering) ===")
    original_by_type = {"s_ok_r_ok": [], "s_ok_r_fail": [], "s_fail_r_ok": [], "s_fail_r_fail": []}
    for sample in matched_samples:
        sample_type = get_sample_type(sample)
        if sample_type in original_by_type:
            original_by_type[sample_type].append(sample)
    
    for st, samples in original_by_type.items():
        print(f"  {st}: {len(samples)}")
    
    # 5. 过滤数据
    print("\n=== Filtering Samples ===")
    print("  Filtering rules:")
    print("    1. Remove s_fail_r_fail (both paths wrong)")
    print("    2. Remove r_fail (r_path_correct=False)")
    print("    3. Keep only consistent samples (original_s_path_correct == new_s_path_correct)")
    
    filtered_samples = []
    filter_stats = {
        "total": len(matched_samples),
        "removed_s_fail_r_fail": 0,
        "removed_r_fail": 0,
        "removed_inconsistent": 0,
        "kept": 0,
    }
    
    for sample in matched_samples:
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
        original_s_correct = sample.get("_original_s_path_correct", sample.get("s_path_correct", False))
        new_s_correct = sample.get("s_path_correct", False)
        
        if original_s_correct != new_s_correct:
            filter_stats["removed_inconsistent"] += 1
            continue
        
        # 通过所有检查，移除临时字段
        sample_clean = {k: v for k, v in sample.items() if not k.startswith("_")}
        filtered_samples.append(sample_clean)
        filter_stats["kept"] += 1
    
    print(f"\n  Before filtering: {filter_stats['total']}")
    print(f"  Removed s_fail_r_fail: {filter_stats['removed_s_fail_r_fail']}")
    print(f"  Removed r_fail: {filter_stats['removed_r_fail']}")
    print(f"  Removed inconsistent: {filter_stats['removed_inconsistent']}")
    print(f"  Kept (consistent): {filter_stats['kept']}")
    print(f"  Total removed: {filter_stats['total'] - filter_stats['kept']}")
    
    # 6. 重新分类样本
    print("\n=== New Sample Distribution ===")
    by_type = {"s_ok_r_ok": [], "s_ok_r_fail": [], "s_fail_r_ok": [], "s_fail_r_fail": []}
    
    for sample in filtered_samples:
        sample_type = get_sample_type(sample)
        if sample_type in by_type:
            by_type[sample_type].append(sample)
    
    for st, samples in by_type.items():
        print(f"  {st}: {len(samples)}")
    
    # 7. 过采样 s_fail_r_ok
    print(f"\n=== Oversampling s_fail_r_ok ===")
    grpo_samples = []
    
    # 添加所有有效样本
    for st, samples in by_type.items():
        grpo_samples.extend(samples)
    
    # 过采样 s_fail_r_ok
    s_fail_r_ok = by_type["s_fail_r_ok"]
    for _ in range(args.oversample_ratio - 1):
        grpo_samples.extend(s_fail_r_ok)
    
    print(f"  Total after oversampling: {len(grpo_samples)}")
    print(f"  s_fail_r_ok oversampled {args.oversample_ratio}x")
    
    # 8. 拆分 train/val
    random.shuffle(grpo_samples)
    val_size = int(len(grpo_samples) * args.val_ratio)
    val_samples = grpo_samples[:val_size]
    train_samples = grpo_samples[val_size:]
    
    print(f"\n=== Train/Val Split ===")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    
    # 9. 保存
    train_path = output_dir / "train_grpo.jsonl"
    val_path = output_dir / "val_grpo.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    # 10. 统计
    train_s_correct = sum(1 for s in train_samples if s["s_path_correct"])
    train_r_only = sum(1 for s in train_samples if not s["s_path_correct"] and s["r_path_correct"])
    
    stats = {
        "total_train": len(train_samples),
        "total_val": len(val_samples),
        "train_s_correct": train_s_correct,
        "train_r_only_correct": train_r_only,
        "oversample_ratio": args.oversample_ratio,
        "filter_stats": filter_stats,
        "original_distribution": {k: len(v) for k, v in original_by_type.items()},
        "new_distribution": {k: len(v) for k, v in by_type.items()},
    }
    
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n=== Saved ===")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Stats: {output_dir / 'stats.json'}")
    
    # 11. 打印对比
    print(f"\n=== Comparison: Before vs After ===")
    print("Original distribution:")
    for st, count in stats["original_distribution"].items():
        print(f"  {st}: {count}")
    print("New distribution:")
    for st, count in stats["new_distribution"].items():
        print(f"  {st}: {count}")


if __name__ == "__main__":
    main()






