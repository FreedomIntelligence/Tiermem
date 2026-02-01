#!/usr/bin/env python3
"""
Mem0 Search Service - FastAPI 服务

在 GAM 环境中运行此服务，swift 环境的 reward plugin 通过 HTTP 请求调用。

启动方式:
    conda activate GAM
    cd <PROJECT_ROOT>
    uvicorn scripts.router_training.mem0_service:app --host 0.0.0.0 --port 8765

测试:
    curl -X POST http://localhost:8765/search \
        -H "Content-Type: application/json" \
        -d '{"session_id": "mab_Accurate_Retrieval_0", "query": "test", "user_id": "mab_Accurate_Retrieval_0:user", "top_k": 5}'
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 添加项目路径
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.mem0 import Memory
from src.mem0.configs.base import MemoryConfig

# 创建日志目录
LOG_DIR = Path(PROJECT_ROOT) / "logs/mem0_service"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件路径（按日期命名）
LOG_FILE = LOG_DIR / f"mem0_service_{datetime.now().strftime('%Y%m%d')}.log"

# 配置日志 - 同时输出到控制台和文件
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)

# 文件处理器
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)

# 添加处理器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Logging to file: {LOG_FILE}")

app = FastAPI(title="Mem0 Search Service", version="1.0.0")

# 缓存 mem0 实例
_mem0_cache = {}


class SearchRequest(BaseModel):
    session_id: str
    query: str
    user_id: str
    top_k: int = 5


class SearchResult(BaseModel):
    memory: str
    score: float


class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResult]
    error: Optional[str] = None


def get_mem0_for_session(session_id: str) -> Memory:
    """获取或创建 session 对应的 mem0 实例"""
    if session_id in _mem0_cache:
        return _mem0_cache[session_id]

    collection_name = f"TierMem_memory_agent_bench_{session_id}"
    logger.info(f"Creating mem0 instance for collection: {collection_name}")

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
                "collection_name": collection_name
            },
        },
    )

    mem0 = Memory(config=config)
    _mem0_cache[session_id] = mem0
    return mem0


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "cached_sessions": len(_mem0_cache)}


def _search_with_retry(mem0, query, user_id, top_k, request_id, max_retries=3, retry_delay=1.0):
    """执行带重试的搜索（同步函数，在线程池中运行）"""
    for attempt in range(max_retries):
        try:
            results = mem0.search(
                query=query,
                user_id=user_id,
                limit=top_k
            )
            # 成功则返回结果
            return results
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # 检查是否是超时或连接错误
            is_retryable = (
                "408" in error_msg or 
                "Timeout" in error_type or 
                "timeout" in error_msg.lower() or
                "Connection" in error_type or
                "UnexpectedResponse" in error_type
            )
            
            if attempt < max_retries - 1 and is_retryable:
                wait_time = retry_delay * (2 ** attempt)  # 指数退避
                logger.warning(
                    f"[RETRY {request_id}] Attempt {attempt + 1}/{max_retries} failed: {error_type}: {error_msg}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                # 最后一次尝试或不可重试的错误，直接抛出
                raise


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """执行 mem0 搜索（带重试机制和超时保护）"""
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # 记录请求详情
    request_log = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "type": "request",
        "session_id": request.session_id,
        "user_id": request.user_id,
        "query": request.query,
        "top_k": request.top_k,
    }
    logger.info(f"[REQUEST {request_id}] {json.dumps(request_log, ensure_ascii=False)}")

    # 超时配置（秒）- 总超时时间，包括重试
    total_timeout = 120.0  # 2分钟总超时
    
    try:
        mem0 = get_mem0_for_session(request.session_id)

        # 在线程池中执行搜索，避免阻塞事件循环
        # 同时设置总超时时间
        loop = asyncio.get_event_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # 使用默认线程池
                    _search_with_retry,
                    mem0,
                    request.query,
                    request.user_id,
                    request.top_k,
                    request_id,
                ),
                timeout=total_timeout
            )
        except asyncio.TimeoutError:
            error_msg = f"Search request timed out after {total_timeout}s"
            logger.error(f"[TIMEOUT {request_id}] {error_msg}")
            raise TimeoutError(error_msg)

        # 解析结果
        search_results = []
        if isinstance(results, dict) and "results" in results:
            results = results["results"]

        for r in results:
            if isinstance(r, dict):
                memory_text = r.get("memory", "") or r.get("text", "")
                score = r.get("score", 0.5)
            else:
                memory_text = str(r)
                score = 0.5

            search_results.append(SearchResult(memory=memory_text, score=float(score)))

        # 记录响应详情
        response_log = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "type": "response",
            "success": True,
            "session_id": request.session_id,
            "query": request.query,
            "num_results": len(search_results),
            "results": [
                {
                    "memory": result.memory[:200] + "..." if len(result.memory) > 200 else result.memory,
                    "score": result.score,
                    "memory_length": len(result.memory)
                }
                for result in search_results
            ],
        }
        logger.info(f"[RESPONSE {request_id}] {json.dumps(response_log, ensure_ascii=False)}")

        return SearchResponse(success=True, results=search_results)

    except Exception as e:
        # 记录错误响应
        error_log = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "success": False,
            "session_id": request.session_id,
            "query": request.query,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        logger.error(f"[ERROR {request_id}] {json.dumps(error_log, ensure_ascii=False)}", exc_info=True)
        return SearchResponse(success=False, results=[], error=str(e))


@app.post("/clear_cache")
async def clear_cache():
    """清除缓存的 mem0 实例"""
    global _mem0_cache
    count = len(_mem0_cache)
    _mem0_cache = {}
    logger.info(f"Cleared {count} cached mem0 instances")
    return {"cleared": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
