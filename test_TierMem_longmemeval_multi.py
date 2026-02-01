#!/usr/bin/env python3
"""
并发版本的 LinkedViewSystem (TierMem) 测试脚本 - LongMemEval

采用"每个 Session 作为一个原子任务（Write + QA）"的并发模式：
- 每个 Session 先执行 Write，然后立即执行 QA
- Session 内部的 QA 串行执行，避免并发嵌套导致 API 限流
- 这样可以快速反馈与止损、提高数据完整性与断点续跑能力

运行示例：
  # 使用 OpenAI 作为 Router（默认）
  python test_TierMem_longmemeval_multi.py --limit 10 --max-workers 4

  # 调整 Session 内部的并发数（避免 Qdrant 过载）
  python test_TierMem_longmemeval_multi.py --limit 10 --max-workers 4 \
    --write-max-workers 2 --qa-max-workers 2

  # 使用 vLLM 作为 Router
  python test_TierMem_longmemeval_multi.py --limit 10 --router-type vllm \
    --router-base-url http://localhost:8000/v1 \
    --router-model Qwen3-4B-Thinking-2507

配置文档：docs/linked_view_config.md
"""
import argparse
import sys
import os

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str = None, log_dir: str = "logs"):
    """配置日志，同时输出到控制台和文件"""
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
    else:
        log_path = None

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_path:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("src").setLevel(logging.INFO)

    return log_path


logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger("src").setLevel(logging.INFO)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = THIS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runner.run_benchmark_multi import run_benchmark_multi
from core.datasets import longmemeval
from src.memory.linked_view_system import LinkedViewSystem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test LinkedViewSystem (TierMem) on LongMemEval (Concurrent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用 OpenAI gpt-4.1-mini 作为 Router（默认）
  python test_TierMem_longmemeval_multi.py --limit 10 --max-workers 4

  # 调整 Session 内部的并发数（避免 Qdrant 过载）
  python test_TierMem_longmemeval_multi.py --limit 10 --max-workers 4 \\
    --write-max-workers 2 --qa-max-workers 2

  # 使用 vLLM Qwen3-Thinking 作为 Router
  python test_TierMem_longmemeval_multi.py --limit 10 \\
    --router-type vllm \\
    --router-base-url http://localhost:8000/v1 \\
    --router-model Qwen3-4B-Thinking-2507

  # 使用默认 LLMRouter（不使用 OpenAI client）
  python test_TierMem_longmemeval_multi.py --limit 10 --router-type llm
"""
    )

    # === 基本参数 ===
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                       help="Memory system model (for mem0, answer generation)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generated)")
    parser.add_argument("--collection-name", type=str, default=None,
                       help="Qdrant collection name (default: auto-generated)")
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                       help="Qdrant server host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                       help="Qdrant server port (default: 6333)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=50,
                       help="Maximum number of concurrent workers (session level)")
    parser.add_argument("--write-max-workers", type=int, default=50,
                       help="Maximum number of concurrent turns per session (default: 4)")
    parser.add_argument("--qa-max-workers", type=int, default=3,
                       help="Maximum number of concurrent QA pairs per session (default: 4)")
    parser.add_argument("--executor", type=str, default="thread", choices=["thread", "process"],
                       help="Executor type: thread (default) or process")

    # === Router 配置 ===
    router_group = parser.add_argument_group("Router Configuration")
    router_group.add_argument("--router-type", type=str, default="vllm",
                             choices=["openai", "vllm", "llm"],
                             help="Router type: openai (default), vllm, or llm")
    router_group.add_argument("--router-model", type=str, default="Qwen3-0.6B",
                             help="Router model name (default: same as --model for openai, "
                                  "Qwen3-0.6B for vllm)")
    router_group.add_argument("--router-base-url", type=str, default="http://localhost:8000/v1",
                             help="Router API base URL (required for vllm)")
    router_group.add_argument("--router-api-key", type=str, default="vllm-api-key",
                             help="Router API key (optional, uses env var for openai)")
    router_group.add_argument("--router-thinking", action="store_true", default=True,
                             help="Enable thinking mode for vLLM (default: True)")
    router_group.add_argument("--no-router-thinking", dest="router_thinking", action="store_false",
                             help="Disable thinking mode for vLLM")

    # === Reranker 配置 ===
    reranker_group = parser.add_argument_group("Reranker Configuration")
    reranker_group.add_argument("--use-reranker", action="store_true", default=False,
                               help="Enable Qwen3-Reranker for hit reranking (requires GPU)")
    reranker_group.add_argument("--reranker-top-k", type=int, default=5,
                               help="Number of top hits to keep after reranking (default: 5)")
    reranker_group.add_argument("--reranker-model-path", type=str,
                               default="Qwen/Qwen3-Reranker-0.6B",
                               help="Path to Qwen3-Reranker model (HuggingFace model name or local path)")

    # === 优化版 R 路径配置 ===
    optimization_group = parser.add_argument_group("R-Path Optimization")
    optimization_group.add_argument("--use-optimized-r-path", action="store_true", default=True,
                                   help="Use optimized R path (1 LLM call per iteration instead of 3)")
    optimization_group.add_argument("--no-optimized-r-path", dest="use_optimized_r_path", action="store_false",
                                   help="Disable optimized R path (use original 3-call approach)")

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not set!")
        print("Please set it, e.g.:")
        print("  export OPENAI_API_KEY=your_key_here")
        return 1

    # vLLM router 必须指定 base_url
    if args.router_type == "vllm" and not args.router_base_url:
        print("✗ --router-base-url is required when using --router-type vllm")
        print("Example: --router-base-url http://localhost:8000/v1")
        return 1

    # 生成 run_id
    if args.run_id:
        run_id = args.run_id
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"tiermem_longmemeval_multi_{timestamp}"

    log_file = f"{run_id}.log"
    log_path = setup_logging(log_file=log_file, log_dir="logs")

    # 生成唯一的 collection 名称
    if args.collection_name:
        collection_name = args.collection_name
    else:
        username = os.getenv("USER", "unknown")
        collection_name = f"mem0_linked_{username}_{run_id}"

    # === 构建 Router 配置 ===
    router_config = {"type": args.router_type}

    if args.router_type == "vllm":
        router_config["base_url"] = args.router_base_url
        router_config["model"] = args.router_model or "Qwen3-4B-Thinking-2507"
        router_config["api_key"] = args.router_api_key or "vllm-api-key"
        router_config["is_thinking_model"] = args.router_thinking
    elif args.router_type == "openai":
        router_config["model"] = args.router_model or args.model
        if args.router_api_key:
            router_config["api_key"] = args.router_api_key
        if args.router_base_url:
            router_config["base_url"] = args.router_base_url
    # router_type == "llm" 不需要额外配置

    print(f"\n{'='*60}")
    print("Testing LinkedViewSystem (TierMem) on LongMemEval (Concurrent)")
    print(f"{'='*60}")
    print(f"Memory System Model: {args.model}")
    print(f"Router Type: {args.router_type}")
    if args.router_type == "vllm":
        print(f"  - Model: {router_config['model']}")
        print(f"  - Base URL: {router_config['base_url']}")
        print(f"  - Thinking Mode: {router_config['is_thinking_model']}")
    elif args.router_type == "openai":
        print(f"  - Model: {router_config['model']}")
    print(f"Limit: {args.limit} sessions")
    print(f"Session-level Workers: {args.max_workers} ({args.executor} executor)")
    print(f"Write Workers (per session): {args.write_max_workers}")
    print(f"QA Workers (per session): {args.qa_max_workers}")
    print(f"Output: {args.output_dir}")
    print(f"Qdrant Server: {args.qdrant_host}:{args.qdrant_port}")
    print(f"Qdrant Collection: {collection_name}")
    print(f"Reranker: {'Enabled' if args.use_reranker else 'Disabled'}")
    if args.use_reranker:
        print(f"  - Model: {args.reranker_model_path}")
        print(f"  - Top K: {args.reranker_top_k}")
    print(f"Optimized R-Path: {'Enabled' if args.use_optimized_r_path else 'Disabled'}")
    if log_path:
        print(f"Log file: {log_path}")
    print(f"{'='*60}\n")

    # 构造 LinkedViewSystem 的配置
    lv_cfg = {
        "benchmark_name": "longmemeval_chunk_500",
        "write_facts_to_database": True,
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
                    "host": args.qdrant_host,
                    "port": args.qdrant_port,
                    "collection_name": collection_name,
                },
            },
        },

        # === LLM 配置 ===
        "memory_system_model": args.model,  # 用于 mem0、答案生成等

        # === Router 配置 ===
        "router_config": router_config,

        # === QueryRewriter 配置 ===
        "use_query_rewriter": False,
        "use_dual_retrieval": False,
        "rewriter_guide_update_freq": 10,

        # === 其他配置 ===
        "router_threshold": 0.5,
        "top_k": 5,
        "max_research_iters": 3,
        "page_size": 100,
        # === Reranker 配置 ===
        "use_reranker": args.use_reranker,
        "reranker_top_k": args.reranker_top_k,
        "reranker_model_path": args.reranker_model_path,

        # === 优化版 R 路径配置 ===
        "use_optimized_r_path": args.use_optimized_r_path,
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

    # 运行并发评估
    try:
        summary = run_benchmark_multi(
            system=system,
            dataset_module=longmemeval,
            benchmark_name="longmemeval_chunk_500",
            run_id=run_id,
            config={
                "model_name": args.model,
                "split": "test",
            },
            output_dir=args.output_dir,
            limit=args.limit,
            max_workers=args.max_workers,
            executor_type=args.executor,
            system_config=lv_cfg,
            write_max_workers=args.write_max_workers,
            qa_max_workers=args.qa_max_workers,
            load_only=True,
        )

        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}")
        result_path = f"{args.output_dir}/longmemeval/linked_view/{run_id}/"
        print(f"Results saved to: {result_path}")
        print(f"  - Session logs: {result_path}sessions/")
        print(f"  - Summary: {result_path}summary.json")
        print("\nMetrics:")
        for key, value in summary.get("metrics", {}).items():
            print(f"  {key}: {value}")

        return 0

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

