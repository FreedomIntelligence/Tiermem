#!/usr/bin/env python3
"""
并发版本的 LinkedViewSystem (TierMem) 在 MemoryAgentBench 上的测试脚本

采用"每个 Session 作为一个原子任务（Write + QA）"的并发模式：
- 每个 Session 先执行 Write，然后立即执行 QA
- Session 内部的 QA 串行执行，避免并发嵌套导致 API 限流
- 这样可以快速反馈与止损、提高数据完整性与断点续跑能力

运行示例：
  python test_TierMem_memoryagentbench.py --split Accurate_Retrieval --sub-dataset longmemeval_s* --limit 2 --max-workers 4
  python test_TierMem_memoryagentbench.py --split Accurate_Retrieval,Test_Time_Learning --limit 10 --max-workers 4

说明：
- 数据集：core.datasets.memory_agent_bench
- 系统：src.memory.linked_view_system.LinkedViewSystem
- 底层通过 run_benchmark_multi 并发评估
"""
import argparse
import sys
import os
from types import SimpleNamespace

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str = None, log_dir: str = "logs"):
    """配置日志，同时输出到控制台和文件"""
    # 创建日志目录
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
    else:
        log_path = None
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # 清除已有的handlers（避免重复）
    root_logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件输出handler（如果指定了日志文件）
    if log_path:
        # 使用RotatingFileHandler，自动轮转日志文件
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,  # 保留5个备份文件
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置src模块的日志级别
    logging.getLogger("src").setLevel(logging.INFO)
    
    return log_path


# 初始化日志（稍后会在main函数中根据run_id重新配置）
logging.basicConfig(level=logging.WARNING, force=True)  # 根 logger：WARNING+
logging.getLogger("src").setLevel(logging.INFO) 
# 确保项目根目录在 sys.path 中，方便以 core / src 方式导入
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = THIS_DIR  # 当前这个仓库就是根目录
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runner.run_benchmark_multi import run_benchmark_multi
from core.datasets import memory_agent_bench as mab
from src.memory.linked_view_system import LinkedViewSystem


def main() -> int:
    parser = argparse.ArgumentParser(description="Test LinkedViewSystem (TierMem) on MemoryAgentBench (Concurrent)")
    parser.add_argument("--limit", type=int, default=2, help="Limit number of sessions")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name for fast/slow LLMs")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generated)")
    parser.add_argument("--collection-name", type=str, default=None,
                       help="Qdrant collection name (default: auto-generated from username and run_id)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of concurrent workers")
    parser.add_argument("--executor", type=str, default="thread", choices=["thread", "process"],
                       help="Executor type: thread (default) or process")
    parser.add_argument(
        "--split",
        type=str,
        default="Accurate_Retrieval",
        help="MemoryAgentBench split(s), support comma-separated list (e.g., 'Accurate_Retrieval,Test_Time_Learning')"
    )
    parser.add_argument(
        "--sub-dataset",
        type=str,
        default=None,
        help="Sub-dataset source name/pattern, support comma-separated list; None = no filter"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local dataset path (load_from_disk). If not set, use HF."
    )

    args = parser.parse_args()

    # 检查 OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not set!")
        print("Please set it, e.g.:")
        print("  export OPENAI_API_KEY=your_key_here")
        return 1

    # 解析 splits（支持逗号分隔的多个 split）
    valid_splits = ["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"]
    splits = [s.strip() for s in args.split.split(",")]
    for split in splits:
        if split not in valid_splits:
            print(f"✗ Invalid split: {split}")
            print(f"Valid splits: {', '.join(valid_splits)}")
            return 1

    # 生成 run_id（用于日志和结果目录）
    if args.run_id:
        run_id = args.run_id
    else:
        # 使用时间戳生成更简洁的 run_id
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        splits_str = "_".join(splits)
        run_id = f"tiermem_mab_{splits_str}"

    log_file = f"{run_id}.log"
    log_path = setup_logging(log_file=log_file, log_dir="logs")

    # 生成唯一的 collection 名称，避免与其他运行冲突
    if args.collection_name:
        # 用户显式指定 collection 名称
        collection_name = args.collection_name
    else:
        # 自动生成：使用更简洁的命名（只包含 username 和 run_id，不包含 limit/workers）
        username = os.getenv("USER", "unknown")
        collection_name = f"mem0_linked_{username}_{run_id}"

    print(f"\n{'='*60}")
    print("Testing LinkedViewSystem (TierMem) on MemoryAgentBench (Concurrent)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Sub-dataset: {args.sub_dataset}")
    print(f"Limit: {args.limit} sessions per split" if args.limit else "Limit: None (all sessions)")
    print(f"Workers: {args.max_workers} ({args.executor} executor)")
    print(f"Output: {args.output_dir}")
    print(f"Qdrant Collection: {collection_name}")
    if log_path:
        print(f"Log file: {log_path}")
    print(f"{'='*60}\n")

    # 构造 LinkedViewSystem 的配置
    lv_cfg = {
        "benchmark_name": "memory_agent_bench",  # 传入 benchmark_name，用于生成 collection 名称
        "mem0_config": {
            "backend": "mem0",
            "llm": {
                "provider": "openai",
                "config": {
                    "model": args.model,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    # 使用共享 Qdrant 服务（推荐用于并发）
                    # collection_name 会在 reset/load 时根据 session_id 动态生成
                    # 格式：TierMem_{benchmark_name}_{session_id}
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": collection_name,  # 这个值会被动态覆盖，但保留作为默认值
                },
            },
        },
        "page_size": 4000,
        # === Enhanced Router 配置 ===
        # 是否启用 query 改写（默认 True）
        "use_query_rewriter": False,
        # 是否启用双路召回（原始query + 改写query）
        "use_dual_retrieval": False,
        # 改写指南更新频率（每N次answer调用更新一次，0=不自动更新）
        # 推荐：10-20（每10-20个问题更新一次，平衡质量和成本）
        "rewriter_guide_update_freq": 0,

        # === 现有配置 ===
        "router_threshold": 0.5,
        "fast_model": args.model,
        "slow_model": args.model,
        "top_k": 5,
        "max_research_iters": 3,
    }

    # 创建系统
    try:
        system = LinkedViewSystem(lv_cfg)
        print("✓ LinkedViewSystem created successfully")
    except Exception as e:
        print(f"✗ Failed to create LinkedViewSystem: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 循环处理每个 split
    all_summaries = {}
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}\n")

        # 组装 dataset_module，补充 sub_dataset 与 data_dir 参数
        # 注意：run_benchmark_multi 会从 config 中获取 split 并传递给 iter_sessions
        # 所以 lambda 需要接受 split 和 limit 参数（而不是预先绑定）
        dataset_module = SimpleNamespace(
            iter_sessions=lambda split, limit: mab.iter_sessions(
                data_dir=args.data_dir,
                split=split,
                sub_dataset=args.sub_dataset,
                limit=limit,
            )
        )

        # 为每个 split 生成唯一的 sub_run_id
        sub_run_id = f"{run_id}_{split}" if len(splits) > 1 else run_id
        print(f"sub_run_id: {sub_run_id}")
        # 运行并发评估
        try:
            summary = run_benchmark_multi(
                system=system,
                dataset_module=dataset_module,
                benchmark_name="memory_agent_bench",
                run_id=sub_run_id,
                config={
                    "model_name": args.model,
                    "split": split
                },
                output_dir=args.output_dir,
                limit=args.limit,
                max_workers=args.max_workers,
                executor_type=args.executor,
                system_config=lv_cfg,  # 显式传入配置
            )

            all_summaries[split] = summary

            print(f"\n{'='*60}")
            print(f"Split '{split}' Complete!")
            print(f"{'='*60}")
            result_path = f"{args.output_dir}/memory_agent_bench/linked_view/{sub_run_id}/"
            print(f"Results saved to: {result_path}")
            print(f"  - Session logs: {result_path}sessions/ (each session has its own log files)")
            print(f"  - Summary: {result_path}summary.json")
            print("\nMetrics:")
            for key, value in summary.get("metrics", {}).items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"\n✗ Evaluation failed for split '{split}': {e}")
            import traceback
            traceback.print_exc()
            all_summaries[split] = {"error": str(e)}

    # 打印总体汇总
    print(f"\n{'='*60}")
    print("All Splits Complete!")
    print(f"{'='*60}")
    for split, summary in all_summaries.items():
        print(f"\n{split}:")
        if "error" in summary:
            print(f"  Error: {summary['error']}")
        else:
            for key, value in summary.get("metrics", {}).items():
                print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())







