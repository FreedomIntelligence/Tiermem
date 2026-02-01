#!/usr/bin/env python3
"""
构建离线数据集：合并 S-path 和 R-path 的实验结果

从 test_amb_summary (action=S) 和 test_amb_raw (action=R) 的结果中，
提取每个问题的预计算信息，生成统一的离线数据集。

重要：从 Qdrant 数据库中查询真实的 summaries！

Usage:
    python scripts/router_training/1_build_offline_dataset.py \
      --s-results results/memory_agent_bench/linked_view/test_amb_summary \
      --r-results results/memory_agent_bench/linked_view/test_amb_raw \
      --output data/router_offline/ \
      --benchmark mab
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_qa_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    加载 QA 结果，返回 {query_id: result} 的字典

    优先从 eval_details_with_llm_judge.json 读取（包含 llm_judge_score）
    如果不存在，则从 sessions/*_qa.jsonl 读取
    """
    results = {}

    # 尝试从 eval_details_with_llm_judge.json 读取（包含 llm_judge 结果）
    eval_details_file = results_dir / "results" / "eval_details_with_llm_judge.json"
    if eval_details_file.exists():
        print(f"  Loading from eval_details_with_llm_judge.json (with llm_judge)")
        with open(eval_details_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                query_id = item.get("query_id", "")
                if query_id:
                    results[query_id] = item
        return results

    # Fallback: 从 sessions/*_qa.jsonl 读取
    sessions_dir = results_dir / "sessions"
    if not sessions_dir.exists():
        print(f"Warning: {sessions_dir} does not exist")
        return {}

    qa_files = list(sessions_dir.glob("*_qa.jsonl"))

    for qa_file in qa_files:
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    query_id = data.get("query_id", "")
                    if query_id:
                        results[query_id] = data
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {qa_file}: {e}")

    return results


class SummaryFetcher:
    """从 Qdrant 获取 summaries 的工具类，支持缓存"""

    def __init__(self, benchmark: str = "memory_agent_bench", cache_file: Optional[str] = None):
        self.benchmark = benchmark
        self._mem0_cache: Dict[str, Any] = {}
        self._summary_cache: Dict[str, tuple] = {}  # query_id -> (summaries, scores)

        # 从缓存文件加载已有的 summaries
        if cache_file:
            self._load_cache(cache_file)

    def _load_cache(self, cache_path: str):
        """从之前的 offline 数据集加载 summaries 缓存"""
        cache_path = Path(cache_path)
        if cache_path.is_file():
            files = [cache_path]
        elif cache_path.is_dir():
            files = list(cache_path.glob("*.jsonl"))
        else:
            print(f"  Cache path not found: {cache_path}")
            return

        loaded = 0
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        query_id = data.get("query_id", "")
                        summaries = data.get("original_summaries", [])
                        scores = data.get("original_scores", [])
                        if query_id and summaries:
                            self._summary_cache[query_id] = (summaries, scores)
                            loaded += 1
            except Exception as e:
                print(f"  Warning: Failed to load cache from {f}: {e}")

        if loaded > 0:
            print(f"  Loaded {loaded} cached summaries from {cache_path}")

    def get_cached(self, query_id: str) -> Optional[tuple]:
        """获取缓存的 summaries"""
        return self._summary_cache.get(query_id)

    def get_mem0(self, session_id: str) -> Any:
        """获取或创建 mem0 实例"""
        if session_id in self._mem0_cache:
            return self._mem0_cache[session_id]

        try:
            from src.mem0 import Memory
            from src.mem0.configs.base import MemoryConfig

            collection_name = f"TierMem_{self.benchmark}_{session_id}"

            config = MemoryConfig(
                llm={
                    "provider": "openai",
                    "config": {"model": "gpt-4.1-mini"},
                },
                vector_store={
                    "provider": "qdrant",
                    "config": {
                        "host": "localhost",
                        "port": 6333,
                        "collection_name": collection_name,
                    },
                },
            )
            mem0 = Memory(config=config)
            self._mem0_cache[session_id] = mem0
            return mem0
        except Exception as e:
            print(f"Warning: Failed to create mem0 for {session_id}: {e}")
            return None

    def search(self, session_id: str, question: str, query_id: str = "", top_k: int = 5) -> tuple:
        """
        搜索获取 summaries 和 scores

        优先使用缓存，缓存没有才查询 Qdrant

        Returns:
            (summaries, scores): 两个列表
        """
        # 先检查缓存
        if query_id:
            cached = self.get_cached(query_id)
            if cached:
                return cached

        mem0 = self.get_mem0(session_id)
        if mem0 is None:
            return [], []

        try:
            user_id = f"{session_id}:user"
            results = mem0.search(query=question, user_id=user_id, limit=top_k)

            summaries = []
            scores = []

            if isinstance(results, dict) and "results" in results:
                results = results["results"]

            for r in results:
                if isinstance(r, dict):
                    memory_text = r.get("memory", "") or r.get("text", "")
                    score = r.get("score", 0.5)
                else:
                    memory_text = str(r)
                    score = 0.5

                if memory_text:
                    summaries.append(memory_text)
                    scores.append(float(score))

            return summaries, scores
        except Exception as e:
            # Collection 可能不存在
            return [], []


def is_strictly_correct(result: Optional[Dict[str, Any]], f1_threshold: float = 0.3) -> bool:
    """
    严格的正确性判断：f1_score > threshold AND llm_judge_score == 1

    只有同时满足两个条件才认为正确，保证数据质量
    """
    if result is None:
        return False

    # 获取 F1 score (可能字段名为 F1, f1, score 等)
    f1_score = result.get("F1", result.get("f1", result.get("score", 0)))

    # 获取 llm_judge_score
    llm_judge = result.get("llm_judge_score", None)

    # 必须同时满足: f1 > threshold AND llm_judge == 1
    if llm_judge is not None:
        return f1_score > f1_threshold and llm_judge == 1
    else:
        # 如果没有 llm_judge，fallback 到仅用 f1 (不推荐)
        return f1_score > f1_threshold


def build_offline_sample(
    query_id: str,
    s_result: Optional[Dict[str, Any]],
    r_result: Optional[Dict[str, Any]],
    fetcher: Optional[SummaryFetcher] = None,
    f1_threshold: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """
    构建单个离线样本

    保留每条路径的详细信息：answer, f1_score, llm_judge_score
    使用严格的正确性判断：f1 > 0.3 AND llm_judge == 1
    """
    # 至少需要一个结果
    base_result = s_result or r_result
    if base_result is None:
        return None

    session_id = base_result.get("session_id", "")
    question = base_result.get("question", "")
    ground_truth = base_result.get("ground_truth", [])

    # 从 Qdrant 获取真实的 summaries（优先使用缓存）
    original_summaries = []
    original_scores = []
    if fetcher and session_id and question:
        original_summaries, original_scores = fetcher.search(session_id, question, query_id=query_id)

    # 如果 Qdrant 查询失败，使用 mechanism_trace 中的 scores
    if not original_scores:
        s_trace = s_result.get("mechanism_trace", {}) if s_result else {}
        r_trace = r_result.get("mechanism_trace", {}) if r_result else {}
        original_scores = s_trace.get("scores", []) or r_trace.get("scores", [])

    # S-path 详细信息
    s_answer = ""
    s_f1 = 0.0
    s_llm_judge = None
    s_correct = False
    s_cost = {}
    if s_result:
        s_answer = s_result.get("model_response", "") or s_result.get("prediction", "")
        s_f1 = s_result.get("F1", s_result.get("f1", s_result.get("score", 0)))
        s_llm_judge = s_result.get("llm_judge_score", None)
        s_correct = is_strictly_correct(s_result, f1_threshold)
        s_cost = s_result.get("cost_metrics", {})

    # R-path 详细信息
    r_answer = ""
    r_f1 = 0.0
    r_llm_judge = None
    r_correct = False
    r_cost = {}
    r_summary = ""
    if r_result:
        r_answer = r_result.get("model_response", "") or r_result.get("prediction", "")
        r_f1 = r_result.get("F1", r_result.get("f1", r_result.get("score", 0)))
        r_llm_judge = r_result.get("llm_judge_score", None)
        r_correct = is_strictly_correct(r_result, f1_threshold)
        r_trace = r_result.get("mechanism_trace", {})
        r_cost = r_result.get("cost_metrics", {})
        r_summary = r_trace.get("research_summary", "")

    # 计算最优 action
    if s_correct and not r_correct:
        optimal_action = "S"
    elif not s_correct and r_correct:
        optimal_action = "R"
    elif s_correct and r_correct:
        optimal_action = "S"  # 都对时选更高效的路径
    else:
        optimal_action = "R"  # 都错时仍走 R（深度路径）

    sample = {
        "query_id": query_id,
        "session_id": session_id,
        "question": question,
        "ground_truth": ground_truth,

        # 原始检索信息（从 Qdrant 获取）
        "original_summaries": original_summaries,
        "original_scores": original_scores,

        # S-path 详细结果
        "s_path_correct": s_correct,
        "s_path_answer": s_answer,
        "s_path_f1": s_f1,
        "s_path_llm_judge": s_llm_judge,
        "s_path_cost": s_cost,

        # R-path 详细结果
        "r_path_correct": r_correct,
        "r_path_answer": r_answer,
        "r_path_f1": r_f1,
        "r_path_llm_judge": r_llm_judge,
        "r_path_cost": r_cost,
        "r_path_summary": r_summary[:500] if r_summary else "",

        # 最优决策
        "optimal_action": optimal_action,
    }

    return sample


def main():
    parser = argparse.ArgumentParser(description="Build offline dataset from S/R results")
    parser.add_argument(
        "--s-results",
        type=str,
        default="results/memory_agent_bench/linked_view/test_amb_summary",
        help="Path to S-path results directory"
    )
    parser.add_argument(
        "--r-results",
        type=str,
        default="results/memory_agent_bench/linked_view/test_amb_raw",
        help="Path to R-path results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/router_offline",
        help="Output directory for offline dataset"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="memory_agent_bench",
        help="Benchmark name for Qdrant collection prefix (memory_agent_bench or locomo)"
    )
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Skip Qdrant lookup (use only scores from mechanism_trace)"
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.3,
        help="F1 score threshold for strict correctness (default: 0.3)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to existing offline dataset to use as cache for summaries (avoid re-querying Qdrant)"
    )

    args = parser.parse_args()

    s_results_dir = Path(args.s_results)
    r_results_dir = Path(args.r_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading S-path results from: {s_results_dir}")
    s_results = load_qa_results(s_results_dir)
    print(f"  Loaded {len(s_results)} samples")

    print(f"Loading R-path results from: {r_results_dir}")
    r_results = load_qa_results(r_results_dir)
    print(f"  Loaded {len(r_results)} samples")

    # 合并所有 query_id
    all_query_ids = set(s_results.keys()) | set(r_results.keys())
    print(f"Total unique query IDs: {len(all_query_ids)}")

    # 创建 SummaryFetcher（如果需要从 Qdrant 获取 summaries）
    fetcher = None
    if not args.no_qdrant:
        print(f"Initializing SummaryFetcher for benchmark: {args.benchmark}")
        if args.cache:
            print(f"  Using cache from: {args.cache}")
        fetcher = SummaryFetcher(benchmark=args.benchmark, cache_file=args.cache)

    # 按 session_id 分组统计
    session_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats = {
        "total": 0,
        "s_only": 0,
        "r_only": 0,
        "both": 0,
        "s_correct": 0,
        "r_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "s_better": 0,  # S✓ R✗
        "r_better": 0,  # S✗ R✓
        "with_summaries": 0,  # 成功获取 summaries 的样本
    }

    # 按 session_id 排序，优化 Qdrant 缓存命中
    def get_session_id(query_id):
        result = s_results.get(query_id) or r_results.get(query_id)
        return result.get("session_id", "") if result else ""

    sorted_query_ids = sorted(all_query_ids, key=get_session_id)

    print(f"\nBuilding offline samples (f1_threshold={args.f1_threshold})...")
    for query_id in tqdm(sorted_query_ids, desc="Processing"):
        s_result = s_results.get(query_id)
        r_result = r_results.get(query_id)

        sample = build_offline_sample(
            query_id, s_result, r_result,
            fetcher=fetcher,
            f1_threshold=args.f1_threshold
        )
        if sample is None:
            continue

        # 提取 split 信息（从 session_id）
        session_id = sample["session_id"]
        # session_id 格式: mab_Accurate_Retrieval_0
        parts = session_id.split("_")
        if len(parts) >= 3:
            split_name = "_".join(parts[1:-1])  # 例如 "Accurate_Retrieval"
        else:
            split_name = "unknown"

        sample["split"] = split_name
        session_samples[split_name].append(sample)

        # 统计
        stats["total"] += 1
        if s_result and r_result:
            stats["both"] += 1
        elif s_result:
            stats["s_only"] += 1
        else:
            stats["r_only"] += 1

        if sample["s_path_correct"]:
            stats["s_correct"] += 1
        if sample["r_path_correct"]:
            stats["r_correct"] += 1
        if sample.get("original_summaries"):
            stats["with_summaries"] += 1

        if sample["s_path_correct"] and sample["r_path_correct"]:
            stats["both_correct"] += 1
        elif not sample["s_path_correct"] and not sample["r_path_correct"]:
            stats["both_wrong"] += 1
        elif sample["s_path_correct"] and not sample["r_path_correct"]:
            stats["s_better"] += 1
        elif not sample["s_path_correct"] and sample["r_path_correct"]:
            stats["r_better"] += 1

    # 保存按 split 分组的数据
    for split_name, samples in session_samples.items():
        output_file = output_dir / f"mab_{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  Saved {len(samples)} samples to {output_file}")

    # 保存合并的完整数据集
    all_samples = []
    for samples in session_samples.values():
        all_samples.extend(samples)

    all_output_file = output_dir / "all.jsonl"
    with open(all_output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_samples)} samples to {all_output_file}")

    # 保存统计信息
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # 打印统计
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print(f"(Strict correctness: F1 > {args.f1_threshold} AND llm_judge == 1)")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"  - Both S and R: {stats['both']}")
    print(f"  - S only: {stats['s_only']}")
    print(f"  - R only: {stats['r_only']}")
    print(f"  - With summaries from Qdrant: {stats['with_summaries']} ({100*stats['with_summaries']/max(1,stats['total']):.1f}%)")
    print()
    print(f"S-path correct: {stats['s_correct']} ({100*stats['s_correct']/max(1,stats['total']):.1f}%)")
    print(f"R-path correct: {stats['r_correct']} ({100*stats['r_correct']/max(1,stats['total']):.1f}%)")
    print()
    print(f"Both correct (S✓ R✓): {stats['both_correct']} ({100*stats['both_correct']/max(1,stats['total']):.1f}%)")
    print(f"Both wrong (S✗ R✗): {stats['both_wrong']} ({100*stats['both_wrong']/max(1,stats['total']):.1f}%)")
    print(f"S better (S✓ R✗): {stats['s_better']} ({100*stats['s_better']/max(1,stats['total']):.1f}%)")
    print(f"R better (S✗ R✓): {stats['r_better']} ({100*stats['r_better']/max(1,stats['total']):.1f}%)")
    print()
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
