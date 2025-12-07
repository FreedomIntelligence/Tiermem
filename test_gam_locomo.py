#!/usr/bin/env python3
"""
测试GAM在LoCoMo上的效果

运行示例：
python test_gam_locomo.py --limit 1
"""
import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.runner.run_benchmark import run_benchmark
from core.datasets import locomo
from core.systems.gam_adapter import GAMSystem


def main():
    parser = argparse.ArgumentParser(description="Test GAM on LoCoMo")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # 检查GAM是否可用
    try:
        from gam import MemoryAgent, ResearchAgent
        print("✓ GAM library is available")
    except ImportError:
        print("✗ GAM library not found!")
        print("Please install GAM from /share/home/qmzhu/AGMS/relatedwork/general-agentic-memory")
        print("Or add it to PYTHONPATH:")
        print("  export PYTHONPATH=$PYTHONPATH:/share/home/qmzhu/AGMS/relatedwork/general-agentic-memory")
        return 1
    
    # 检查API key
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not set!")
        print("Please set it:")
        print("  export OPENAI_API_KEY=your_key_here")
        return 1
    
    print(f"\n{'='*60}")
    print("Testing GAM on LoCoMo")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit} sessions")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # 创建GAM系统
    try:
        system = GAMSystem(config={
            "memory_model": args.model,
            "research_model": args.model,
            "working_model": args.model,
            "max_research_iters": 3
        })
        print("✓ GAM system created successfully")
    except Exception as e:
        print(f"✗ Failed to create GAM system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 运行评估
    try:
        run_id = args.run_id or f"gam_locomo_test_{args.limit}"
        
        summary = run_benchmark(
            system=system,
            dataset_module=locomo,
            benchmark_name="locomo",
            run_id=run_id,
            config={
                "model_name": args.model,
                "split": "test"
            },
            output_dir=args.output_dir,
            limit=args.limit
        )
        
        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {args.output_dir}/locomo/gam/{run_id}/")
        print(f"\nMetrics:")
        for key, value in summary.get("metrics", {}).items():
            print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
