"""
BM25 检索器，适配 PageStore 的 Page 结构。

在 PageStore 加载时，为所有页建立 BM25 索引，支持关键词检索。
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pyserini.search.lucene import LuceneSearcher
    logger.debug("[BM25Retriever] pyserini.search.lucene.LuceneSearcher imported successfully")
except ImportError:
    LuceneSearcher = None  # type: ignore
    logger.warning("[BM25Retriever] pyserini not available")

# 使用字符串类型提示避免循环导入
if TYPE_CHECKING:
    from .page_store import Page, PageStore


def _safe_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    """
    安全地删除目录树，带重试机制
    
    Args:
        path: 要删除的目录路径
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    if not os.path.exists(path):
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            # 确保目录真的被删除了
            if not os.path.exists(path):
                return
            time.sleep(delay)
        except OSError as e:
            if attempt == max_retries - 1:
                # 最后一次尝试仍然失败，强制删除
                try:
                    # 尝试更激进的删除方式
                    subprocess.run(['rm', '-rf', path], check=False, capture_output=True)
                    if not os.path.exists(path):
                        return
                except Exception:
                    pass
                raise OSError(f"无法删除目录 {path}: {e}")
            time.sleep(delay)


class PageStoreBM25Retriever:
    """
    BM25 检索器，适配 PageStore 的 Page 结构。
    
    使用 pyserini 构建 Lucene 索引，支持对 PageStore 中的页进行 BM25 检索。
    
    config 需要:
    {
        "index_dir": "xxx",   # 用来放 index/ 和 documents/
        "threads": 4          # 构建索引时的线程数
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 BM25 检索器。
        
        Args:
            config: 配置字典，包含 index_dir 和 threads
        """
        if LuceneSearcher is None:
            raise ImportError("PageStoreBM25Retriever requires pyserini to be installed. "
                            "Install with: pip install pyserini")
        
        self.config = config
        self.index_dir = Path(config["index_dir"])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.searcher: Optional[LuceneSearcher] = None
        self.pages: List["Page"] = []  # 使用字符串类型提示避免循环导入
        self.page_id_to_index: Dict[str, int] = {}  # page_id -> 索引位置（已废弃，现在直接用 page_id）

    def _lucene_dir(self) -> Path:
        """Lucene 索引目录"""
        return self.index_dir / "index"

    def _docs_dir(self) -> Path:
        """
        文档目录（用于构建索引）
        注意：pyserini 需要 jsonl 格式的输入，所以需要创建 documents.jsonl
        但索引目录结构为：index_dir/index/（Lucene索引）和 index_dir/documents/（临时文档）
        """
        return self.index_dir / "documents"

    def build(self, pages: List["Page"]) -> None:
        """
        为给定的页列表构建 BM25 索引。
        
        Args:
            pages: Page 列表
        """
        if not pages:
            print("[PageStoreBM25Retriever] 警告：没有页需要索引")
            return

        # 0. 清理旧的索引和文档
        _safe_rmtree(str(self._lucene_dir()))
        _safe_rmtree(str(self._docs_dir()))
        
        # 1. 创建必要的目录
        self._docs_dir().mkdir(parents=True, exist_ok=True)

        # 2. 将 pages 转换为 documents.jsonl（pyserini 需要 jsonl 格式输入）
        # 注意：只对 content 字段做 BM25 索引，不包含 summary
        # 使用 page_id 作为文档 id，方便后续映射回原始 page
        docs_path = self._docs_dir() / "documents.jsonl"
        with open(docs_path, "w", encoding="utf-8") as f:
            for page in pages:
                # 只使用 page.content 作为检索内容（不包含 summary）
                text = page.content.strip()
                
                # 使用 page_id 作为文档 id，这样检索结果可以直接映射回 page
                json.dump({"id": page.page_id, "contents": text}, f, ensure_ascii=False)
                f.write("\n")

        # 3. 确保 lucene index 目录是干净的
        self._lucene_dir().mkdir(parents=True, exist_ok=True)

        # 4. 调用 pyserini 构建 Lucene 索引
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(self._docs_dir()),
            "--index", str(self._lucene_dir()),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.config.get("threads", 1)),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        
        # 添加重试机制，防止偶发的构建失败
        max_build_retries = 2
        for attempt in range(max_build_retries):
            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_build_retries - 1:
                    print(f"[PageStoreBM25Retriever] 错误：Pyserini 索引构建失败:")
                    print(f"  stdout: {e.stdout}")
                    print(f"  stderr: {e.stderr}")
                    raise
                print(f"[PageStoreBM25Retriever] 警告：Pyserini 索引构建失败，重试 {attempt + 1}/{max_build_retries}...")
                # 清理失败的索引
                _safe_rmtree(str(self._lucene_dir()))
                self._lucene_dir().mkdir(parents=True, exist_ok=True)
                time.sleep(1)

        # 5. 保存页列表（page_id_to_index 不再需要，因为现在直接用 page_id 作为文档 id）
        self.pages = pages
        
        # 6. 加载 searcher
        self.searcher = LuceneSearcher(str(self._lucene_dir()))  # type: ignore
        
        print(f"[PageStoreBM25Retriever] 成功构建索引，共 {len(pages)} 页")
        print(f"[PageStoreBM25Retriever] 索引目录: {self._lucene_dir()}")

    def load(self) -> None:
        """
        从磁盘加载已构建的索引。
        
        注意：需要先调用 build() 构建索引，或者确保索引已存在。
        """
        if not self._lucene_dir().exists():
            raise RuntimeError("BM25 索引未找到，需要先调用 build() 构建索引")
        
        self.searcher = LuceneSearcher(str(self._lucene_dir()))  # type: ignore
        
        # 注意：load() 时无法恢复 pages 列表，因为索引中只存储了文档内容
        # 需要在外部维护 pages 列表，或者从 PageStore 重新加载
        print(f"[PageStoreBM25Retriever] 成功加载索引")
        print(f"[PageStoreBM25Retriever] 索引目录: {self._lucene_dir()}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        pages: Optional[List["Page"]] = None
    ) -> List[Tuple["Page", float]]:
        """
        使用 BM25 检索相关页。

        Args:
            query: 查询字符串
            top_k: 返回前 k 个结果
            pages: 页列表（如果为 None，使用 build() 时保存的 pages）

        Returns:
            List[Tuple[Page, score]]: 按分数降序排列的 (Page, BM25分数) 列表
        """
        if self.searcher is None:
            logger.error("[BM25Retriever] searcher is None, not initialized")
            raise RuntimeError("检索器未初始化，请先调用 build() 或 load()")

        query = query.strip()
        if not query:
            logger.warning("[BM25Retriever] Empty query, returning []")
            return []

        # 使用 BM25 检索
        logger.debug(f"[BM25Retriever] Searching: query='{query[:50]}...', top_k={top_k}")
        py_hits = self.searcher.search(query, k=top_k)
        logger.debug(f"[BM25Retriever] Lucene returned {len(py_hits)} hits")

        # 确定使用哪个页列表
        page_list = pages if pages is not None else self.pages
        logger.debug(f"[BM25Retriever] page_list has {len(page_list) if page_list else 0} pages")

        # 创建 page_id 到 page 的映射
        page_id_to_page = {page.page_id: page for page in page_list}

        results: List[Tuple["Page", float]] = []
        missing_page_ids = []
        for h in py_hits:
            # h.docid 现在是 page_id（不是索引位置）
            page_id = h.docid
            if page_id in page_id_to_page:
                page = page_id_to_page[page_id]
                results.append((page, float(h.score)))
            else:
                missing_page_ids.append(page_id)

        if missing_page_ids:
            logger.warning(f"[BM25Retriever] {len(missing_page_ids)} page_ids from index not found in page_list: {missing_page_ids[:5]}...")

        logger.debug(f"[BM25Retriever] Final results: {len(results)} pages matched")
        return results

    def search_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 10,
        pages: Optional[List["Page"]] = None
    ) -> List[Tuple["Page", float]]:
        """
        根据关键词列表进行 BM25 检索。
        
        Args:
            keywords: 关键词列表
            top_k: 返回前 k 个结果
            pages: 页列表（如果为 None，使用 build() 时保存的 pages）
            
        Returns:
            List[Tuple[Page, score]]: 按分数降序排列的 (Page, BM25分数) 列表
        """
        # 将关键词列表组合成查询字符串
        query = " ".join(keywords)
        return self.search(query, top_k=top_k, pages=pages)

    def get_snippet(
        self,
        page: "Page",
        query: str,
        window_size: int = 150
    ) -> str:
        """
        从页内容中提取包含查询关键词的片段。
        
        Args:
            page: 页对象
            query: 查询字符串
            window_size: 关键词前后保留的字符数
            
        Returns:
            截取后的文本片段
        """
        content = page.content
        if not content:
            return ""
        
        # 将查询拆分为关键词
        keywords = [k.strip() for k in query.split() if k.strip()]
        if not keywords:
            return content[:window_size * 2] + "..." if len(content) > window_size * 2 else content
        
        content_lower = content.lower()
        search_terms = [k.lower() for k in keywords if k]
        
        # 找到第一个匹配关键词的位置
        first_idx = -1
        for term in search_terms:
            idx = content_lower.find(term)
            if idx != -1:
                if first_idx == -1 or idx < first_idx:
                    first_idx = idx
        
        if first_idx == -1:
            # 没找到关键词，返回开头
            return content[:window_size * 2] + "..." if len(content) > window_size * 2 else content

        # 计算截取范围
        start = max(0, first_idx - window_size)
        end = min(len(content), first_idx + len(search_terms[0]) + window_size)
        
        snippet = content[start:end]
        
        # 添加省略号
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        
        # 清理断裂的行
        return f"{prefix}{snippet}{suffix}".replace("\n", " ").strip()

    def update(self, pages: List["Page"]) -> None:
        """
        更新索引（全量重建）。
        
        Args:
            pages: 新的页列表
        """
        # Lucene 没有好用的"增量追加+可删改文档"的轻量接口
        # 对现在这个原型我们可以直接全量重建，保持简单可靠
        self.build(pages)

