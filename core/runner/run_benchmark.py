"""
Unified Benchmark Runner

Run TierMem (LinkedViewSystem) on supported benchmarks.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.systems.base import MemorySystem
from core.runner.write_phase import run_write_phase
from core.runner.qa_phase import run_qa_phase
from core.runner.summary_phase import run_summary_phase


def run_benchmark(
    system: MemorySystem,
    dataset_module,
    benchmark_name: str,
    run_id: str,
    config: Dict[str, Any],
    output_dir: str = "results",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run benchmark evaluation

    Args:
        system: MemorySystem instance
        dataset_module: Dataset module with iter_sessions function
        benchmark_name: Benchmark name
        run_id: Run ID
        config: Configuration dictionary
        output_dir: Output directory
        limit: Limit number of sessions (for testing)

    Returns:
        Dictionary with evaluation results
    """
    model_name = config.get("model_name", "unknown")
    split = config.get("split", "test")

    # Prepare output path
    output_path = Path(output_dir) / benchmark_name / system.get_system_name() / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate all sessions
    sessions = list(dataset_module.iter_sessions(split=split, limit=limit))
    print(f"Processing {len(sessions)} sessions...")

    # 1. Memory Write Phase
    print("\n" + "="*60)
    print("Phase 1: Memory Write")
    print("="*60)
    processed_sessions_write = run_write_phase(
        system=system,
        sessions=sessions,
        run_id=run_id,
        benchmark_name=benchmark_name,
        model_name=model_name,
        output_path=output_path
    )

    # 2. QA Phase
    print("\n" + "="*60)
    print("Phase 2: QA")
    print("="*60)
    processed_sessions_qa = run_qa_phase(
        system=system,
        sessions=sessions,
        run_id=run_id,
        benchmark_name=benchmark_name,
        model_name=model_name,
        output_path=output_path
    )

    # 3. Summary Phase
    print("\n" + "="*60)
    print("Phase 3: Summary")
    print("="*60)
    summary = run_summary_phase(
        benchmark_name=benchmark_name,
        system_name=system.get_system_name(),
        model_name=model_name,
        run_id=run_id,
        config=config,
        output_path=output_path,
        num_sessions=len(sessions)
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--system", type=str, default="linked_view",
                        choices=["linked_view"],
                        help="System name (currently only linked_view is supported)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["locomo", "longmemeval", "memory_agent_bench", "hotpotqa"],
                        help="Benchmark name")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generated)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini", help="Model name")

    args = parser.parse_args()

    # Generate run_id
    if args.run_id is None:
        args.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Import dataset module
    if args.benchmark == "locomo":
        from core.datasets import locomo as dataset_module
    elif args.benchmark == "hotpotqa":
        from core.datasets import hotpotqa as dataset_module
    elif args.benchmark == "memory_agent_bench":
        from core.datasets import memory_agent_bench as dataset_module
    elif args.benchmark == "longmemeval":
        from core.datasets import longmemeval as dataset_module
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    # Import and configure LinkedViewSystem
    from src.memory.linked_view_system import LinkedViewSystem

    lv_cfg = {
        "mem0_config": {
            "backend": "mem0",
            "llm": {
                "provider": "openai",
                "config": {
                    "model": args.model_name,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                },
            },
        },
        "router_threshold": 0.5,
        "fast_model": args.model_name,
        "slow_model": args.model_name,
        "top_k": 5,
        "max_research_iters": 3,
    }
    system = LinkedViewSystem(lv_cfg)

    # Run evaluation
    config = {
        "model_name": args.model_name,
        "split": args.split
    }

    summary = run_benchmark(
        system=system,
        dataset_module=dataset_module,
        benchmark_name=args.benchmark,
        run_id=args.run_id,
        config=config,
        output_dir=args.output_dir,
        limit=args.limit
    )

    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
