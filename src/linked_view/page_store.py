"""
PageStore：管理分页逻辑。

将原始对话内容按页组织，每页达到一定长度时触发总结。
支持持久化存储到本地文件系统。
"""

import os
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, field, asdict
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

# 延迟导入 BM25 检索器，避免在没有 pyserini 时出错
try:
    from .retrieve_bm25 import PageStoreBM25Retriever
    BM25_AVAILABLE = True
    logger.info("[PageStore] BM25 retriever (pyserini) is available")
except ImportError:
    # 如果相对导入失败，尝试绝对导入（用于直接运行脚本的情况）
    try:
        import sys
        from pathlib import Path
        # 添加项目根目录到路径
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.linked_view.retrieve_bm25 import PageStoreBM25Retriever
        BM25_AVAILABLE = True
        logger.info("[PageStore] BM25 retriever (pyserini) is available (via absolute import)")
    except ImportError:
        traceback.print_exc()
        BM25_AVAILABLE = False
        PageStoreBM25Retriever = None  # type: ignore


@dataclass
class Page:
    """一页内容。"""
    page_id: str
    session_id: str
    user_id: str
    raw_log_ids: List[str] = field(default_factory=list)  # 该页包含的所有raw_log_id
    content: str = ""  # 累积的原始内容
    summary: Optional[str] = None  # LLM生成的总结
    created_at: str = ""  # 创建时间戳
    summarized_at: Optional[str] = None  # 总结时间戳
    stored_to_mem0: bool = False  # 是否已存入 mem0
    memories: List[str] = field(default_factory=list)  # mem0.add 返回的 memory 列表

    def to_dict(self) -> Dict:
        """转换为字典以便序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Page":
        """从字典反序列化。"""
        return cls(
            page_id=data["page_id"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            raw_log_ids=data.get("raw_log_ids", []),
            content=data.get("content", ""),
            summary=data.get("summary"),
            created_at=data.get("created_at", ""),
            summarized_at=data.get("summarized_at"),
            stored_to_mem0=data.get("stored_to_mem0", False),
            memories=data.get("memories", []),
        )


class PageStore:
    """
    管理分页逻辑的存储，支持持久化。

    - 累积内容直到达到 page_size（字符数）
    - 提供当前页的访问接口
    - 支持创建新页
    - 持久化存储到本地文件系统
    """

    def __init__(
        self, 
        page_size: int = 2000, 
        storage_dir: str = "./tmp/page_store",
        enable_bm25: bool = True,
        bm25_index_dir: Optional[str] = None,
        bm25_threads: int = 4
    ) -> None:
        """
        初始化 PageStore。

        Args:
            page_size: 每页的最大字符数（默认2000）
            storage_dir: 存储目录路径
            enable_bm25: 是否启用 BM25 检索（需要 pyserini）
            bm25_index_dir: BM25 索引目录（如果为 None，使用 storage_dir/bm25_index）
            bm25_threads: 构建 BM25 索引时的线程数
        """
        self.page_size = page_size
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # 按 session_id 组织，每个 session 有当前页
        self._current_pages: Dict[str, Page] = {}
        # 所有已完成的页（已总结的）
        self._completed_pages: Dict[str, List[Page]] = {}
        # 内存缓存：已加载的 session 数据
        self._loaded_sessions: set = set()
        
        # BM25 检索器（按 session 组织，延迟初始化）
        self._bm25_retrievers: Dict[str, PageStoreBM25Retriever] = {}  # session_id -> retriever
        self._enable_bm25 = enable_bm25 and BM25_AVAILABLE
        self._bm25_base_dir = bm25_index_dir or str(self.storage_dir / "bm25_index")
        self._bm25_threads = bm25_threads
        
        # 线程安全：为每个 session 创建独立的锁
        # 这样可以支持不同 session 的并发，同时保护同一 session 内的并发访问
        self._session_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # 保护 _session_locks 字典本身的锁

    def _get_session_file(self, session_id: str) -> Path:
        """获取 session 对应的文件路径。"""
        # 使用安全的文件名（移除特殊字符）
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_session_id}.json"
    
    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """获取 session 级别的锁（线程安全）。"""
        with self._locks_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    def _load_session(self, session_id: str) -> None:
        """从文件加载指定 session 的数据。"""
        if session_id in self._loaded_sessions:
            return

        session_file = self._get_session_file(session_id)
        
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 加载当前页
                    if "current_page" in data and data["current_page"]:
                        self._current_pages[session_id] = Page.from_dict(data["current_page"])
                    
                    # 加载已完成的页
                    if "completed_pages" in data:
                        self._completed_pages[session_id] = [
                            Page.from_dict(page_data) for page_data in data["completed_pages"]
                        ]
            except Exception as e:
                print(f"[PageStore] Failed to load session {session_id}: {e}")

        self._loaded_sessions.add(session_id)

    def _save_session(self, session_id: str) -> None:
        """将指定 session 的数据保存到文件。"""
        session_file = self._get_session_file(session_id)
        try:
            data = {
                "session_id": session_id,
                "current_page": self._current_pages[session_id].to_dict() if session_id in self._current_pages else None,
                "completed_pages": [
                    page.to_dict() for page in self._completed_pages.get(session_id, [])
                ],
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PageStore] Failed to save session {session_id}: {e}")

    def get_current_page(self, session_id: str, user_id: str) -> Page:
        """获取当前页，如果不存在则创建新页。"""
        # 确保已加载该 session 的数据
        self._load_session(session_id)
        
        if session_id not in self._current_pages:
            self._current_pages[session_id] = Page(
                page_id=uuid4().hex,
                session_id=session_id,
                user_id=user_id,
            )
            # 保存新创建的页
            self._save_session(session_id)
        return self._current_pages[session_id]

    def add_content(
        self,
        session_id: str,
        user_id: str,
        raw_log_id: str,
        content: str,
        timestamp: str,
    ) -> Tuple[Page, bool]:
        """
        添加内容到当前页（线程安全）。

        Returns:
            (page, is_full): 返回当前页和是否已满
        """
        # 使用 session 级别的锁保护整个操作
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            page = self.get_current_page(session_id, user_id)
            
            # 设置创建时间（如果是新页）
            if not page.created_at:
                page.created_at = timestamp

            # 添加内容
            if page.content:
                page.content += "\n"
            page.content += content
            page.raw_log_ids.append(raw_log_id)

            # 检查是否达到一页大小
            is_full = len(page.content) >= self.page_size
            #
            # print(f"[PageStore] Page is full: {is_full},length: {len(page.content)},self.page_size: {self.page_size}")
            if is_full:
                # 将当前页移到已完成列表
                if session_id not in self._completed_pages:
                    self._completed_pages[session_id] = []
                self._completed_pages[session_id].append(page)
                # 清空当前页，下次会自动创建新页
                del self._current_pages[session_id]
                # 保存更改
                self._save_session(session_id)
            else:
                # 页未满，但也需要保存当前页的更新
                self._save_session(session_id)

            return page, is_full

    def finalize_page(self, session_id: str, summary: str, timestamp: str) -> Optional[Page]:
        """
        完成当前页（即使未满），设置总结。

        Returns:
            完成的页，如果当前没有页则返回None
        """
        # 确保已加载该 session 的数据
        self._load_session(session_id)
        
        if session_id not in self._current_pages:
            return None

        page = self._current_pages[session_id]
        page.summary = summary
        page.summarized_at = timestamp

        # 移到已完成列表
        if session_id not in self._completed_pages:
            self._completed_pages[session_id] = []
        self._completed_pages[session_id].append(page)

        # 清空当前页
        del self._current_pages[session_id]

        # 保存更改
        self._save_session(session_id)

        return page

    def update_page_summary(self, page_id: str, summary: str, timestamp: str) -> Optional[Page]:
        """
        更新指定页的summary（页可能在_current_pages或_completed_pages中）。

        Args:
            page_id: 页ID
            summary: 总结文本
            timestamp: 总结时间戳

        Returns:
            更新后的页，如果找不到则返回None
        """
        # 在当前页中查找
        for session_id, page in list(self._current_pages.items()):
            if page.page_id == page_id:
                page.summary = summary
                page.summarized_at = timestamp
                self._save_session(session_id)
                return page

        # 在已完成页中查找
        for session_id, pages in list(self._completed_pages.items()):
            for page in pages:
                if page.page_id == page_id:
                    page.summary = summary
                    page.summarized_at = timestamp
                    self._save_session(session_id)
                    return page

        # 如果内存中没找到，尝试从文件加载
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session_id = data.get("session_id")
                    if not session_id:
                        continue

                    # 检查当前页
                    if "current_page" in data and data["current_page"]:
                        if data["current_page"]["page_id"] == page_id:
                            page = Page.from_dict(data["current_page"])
                            page.summary = summary
                            page.summarized_at = timestamp
                            # 确保session已加载
                            self._load_session(session_id)
                            self._current_pages[session_id] = page
                            self._save_session(session_id)
                            return page

                    # 检查已完成的页
                    if "completed_pages" in data:
                        for i, page_data in enumerate(data["completed_pages"]):
                            if page_data["page_id"] == page_id:
                                page = Page.from_dict(page_data)
                                page.summary = summary
                                page.summarized_at = timestamp
                                # 确保session已加载
                                self._load_session(session_id)
                                if session_id not in self._completed_pages:
                                    self._completed_pages[session_id] = []
                                # 更新列表中的页
                                for j, p in enumerate(self._completed_pages[session_id]):
                                    if p.page_id == page_id:
                                        self._completed_pages[session_id][j] = page
                                        break
                                else:
                                    self._completed_pages[session_id].append(page)
                                self._save_session(session_id)
                                return page
            except Exception:
                continue

        return None

    def update_page_mem0_status(self, page_id: str, memories: List[str]) -> Optional[Page]:
        """
        更新指定页的 mem0 状态和 memories（线程安全）。

        Args:
            page_id: 页ID
            memories: mem0.add 返回的 memory 列表

        Returns:
            更新后的页，如果找不到则返回None
        """
        # 需要找到页所属的 session_id 来获取对应的锁
        # 先快速查找（无锁），找到后再用锁保护修改
        session_id_to_update = None
        
        # 快速查找（无锁读取，使用 list() 创建快照）
        for sid, page in list(self._current_pages.items()):
            if page.page_id == page_id:
                session_id_to_update = sid
                break
        
        if session_id_to_update is None:
            for sid, pages in list(self._completed_pages.items()):
                for page in pages:
                    if page.page_id == page_id:
                        session_id_to_update = sid
                        break
                if session_id_to_update:
                    break
        
        # 如果内存中没找到，尝试从文件加载（需要遍历所有 session 文件）
        if session_id_to_update is None:
            for json_file in self.storage_dir.glob("*.json"):
                try:
                    session_id = json_file.stem.replace("_", "/")  # 恢复原始 session_id
                    self._load_session(session_id)
                    # 重新查找
                    for sid, page in list(self._current_pages.items()):
                        if page.page_id == page_id:
                            session_id_to_update = sid
                            break
                    if session_id_to_update:
                        break
                    for sid, pages in list(self._completed_pages.items()):
                        for page in pages:
                            if page.page_id == page_id:
                                session_id_to_update = sid
                                break
                        if session_id_to_update:
                            break
                    if session_id_to_update:
                        break
                except Exception:
                    continue
        
        if session_id_to_update is None:
            return None
        
        # 使用 session 级别的锁保护修改操作
        session_lock = self._get_session_lock(session_id_to_update)
        with session_lock:
            # 在当前页中查找
            if session_id_to_update in self._current_pages:
                page = self._current_pages[session_id_to_update]
                if page.page_id == page_id:
                    page.stored_to_mem0 = True
                    page.memories = memories
                    self._save_session(session_id_to_update)
                    return page

            # 在已完成页中查找
            if session_id_to_update in self._completed_pages:
                for page in self._completed_pages[session_id_to_update]:
                    if page.page_id == page_id:
                        page.stored_to_mem0 = True
                        page.memories = memories
                        self._save_session(session_id_to_update)
                        return page
        
        return None

    def get_page_by_id(self, page_id: str) -> Optional[Page]:
        """根据page_id查找页（在已完成和当前页中查找）。"""
        # 先在所有已加载的 session 中查找
        # 在已完成页中查找
        for pages in self._completed_pages.values():
            for page in pages:
                if page.page_id == page_id:
                    return page

        # 在当前页中查找
        for page in self._current_pages.values():
            if page.page_id == page_id:
                return page

        # 如果没找到，尝试从所有 session 文件中查找
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 检查当前页
                    if "current_page" in data and data["current_page"]:
                        if data["current_page"]["page_id"] == page_id:
                            return Page.from_dict(data["current_page"])
                    
                    # 检查已完成的页
                    if "completed_pages" in data:
                        for page_data in data["completed_pages"]:
                            if page_data["page_id"] == page_id:
                                return Page.from_dict(page_data)
            except Exception:
                continue

        return None

    def get_pages_by_session(self, session_id: str) -> List[Page]:
        """获取指定session的所有页（包括当前页）。"""
        # 确保已加载该 session 的数据
        self._load_session(session_id)
        
        pages = []
        # 已完成页
        if session_id in self._completed_pages:
            pages.extend(self._completed_pages[session_id])
        # 当前页（如果有）
        if session_id in self._current_pages:
            pages.append(self._current_pages[session_id])
        return pages

    def get_all_pages(self) -> List[Page]:
        """获取所有 session 的所有页（包括当前页和已完成页）。"""
        all_pages: List[Page] = []
        
        # 遍历存储目录中的所有 JSON 文件
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 添加当前页
                    if "current_page" in data and data["current_page"]:
                        all_pages.append(Page.from_dict(data["current_page"]))
                    
                    # 添加已完成的页
                    if "completed_pages" in data:
                        for page_data in data["completed_pages"]:
                            all_pages.append(Page.from_dict(page_data))
            except Exception as e:
                print(f"[PageStore] Failed to load pages from {json_file}: {e}")
                continue
        
        # 也添加内存中已加载但可能还未保存的页
        for page in self._current_pages.values():
            # 避免重复添加（如果已经在文件中）
            if not any(p.page_id == page.page_id for p in all_pages):
                all_pages.append(page)
        
        for pages in self._completed_pages.values():
            for page in pages:
                # 避免重复添加
                if not any(p.page_id == page.page_id for p in all_pages):
                    all_pages.append(page)
        
        return all_pages

    def get_pages_without_summary(self) -> List[Page]:
        """获取所有没有 summary 的页。"""
        all_pages = self.get_all_pages()
        return [page for page in all_pages if not page.summary or not page.summary.strip()]

    def get_pages_not_stored_to_mem0(self, session_id: Optional[str] = None) -> List[Page]:
        """获取所有没有存入 mem0 的页。"""
        if session_id:
            pages = self.get_pages_by_session(session_id)
        else:
            pages = self.get_all_pages()
        return [page for page in pages if not page.stored_to_mem0]

    def reset_session(self, session_id: str) -> None:
        """重置指定session的所有页。"""
        # 确保已加载该 session 的数据
        self._load_session(session_id)
        
        if session_id in self._current_pages:
            del self._current_pages[session_id]
        if session_id in self._completed_pages:
            del self._completed_pages[session_id]
        
        # 删除文件
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            try:
                session_file.unlink()
            except Exception as e:
                print(f"[PageStore] Failed to delete session file {session_file}: {e}")
        
        # 从已加载列表中移除
        self._loaded_sessions.discard(session_id)

    # ==========================================
    #  关键词搜索与片段提取功能
    # ==========================================

    def _get_bm25_retriever(self, session_id: str) -> Optional[PageStoreBM25Retriever]:
        """
        获取指定 session 的 BM25 检索器（按需创建）。
        
        注意：此方法只创建检索器对象，不构建索引。索引构建由 _ensure_bm25_index 负责。
        为确保数据一致性，在创建检索器前会先确保 session 数据已加载。
        
        Args:
            session_id: 会话 ID
            
        Returns:
            BM25 检索器，如果 BM25 未启用则返回 None
        """
        if not self._enable_bm25:
            return None
        
        # 如果已存在，直接返回
        if session_id in self._bm25_retrievers:
            return self._bm25_retrievers[session_id]
        
        # 确保 session 数据已加载（与 get_pages_by_session 的逻辑对齐）
        self._load_session(session_id)
        
        # 创建新的检索器（每个 session 独立的索引目录）
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        bm25_dir = str(Path(self._bm25_base_dir) / safe_session_id)
        
        try:
            retriever = PageStoreBM25Retriever({
                "index_dir": bm25_dir,
                "threads": self._bm25_threads
            })
            self._bm25_retrievers[session_id] = retriever
            logger.info(f"[PageStore] BM25 检索器创建成功 (session={session_id}, index_dir={bm25_dir})")
            return retriever
        except Exception as e:
            logger.warning(f"[PageStore] BM25 检索器初始化失败 (session={session_id}): {e}")
            return None
    
    def _ensure_bm25_index(self, session_id: Optional[str] = None) -> None:
        """
        确保指定 session 的 BM25 索引已构建（如果启用 BM25）。
        如果索引已存在则加载，不存在则构建。
        
        注意：此方法会确保 session 数据已加载后再构建索引，与懒加载机制对齐。
        
        Args:
            session_id: 会话 ID，如果为 None 则使用所有 session
        """
        if not self._enable_bm25:
            return
        
        if session_id:
            # 确保 session 数据已加载（与 get_pages_by_session 的逻辑对齐）
            self._load_session(session_id)
            
            # 单个 session 的索引
            retriever = self._get_bm25_retriever(session_id)
            if retriever is None:
                return
            
            # 如果 searcher 已经加载，说明索引已就绪
            if retriever.searcher is not None:
                return
            
            # 检查索引是否存在
            lucene_index_dir = retriever._lucene_dir()
            index_exists = lucene_index_dir.exists() and any(lucene_index_dir.iterdir())
            
            # 获取该 session 的所有页（确保数据已加载）
            session_pages = self.get_pages_by_session(session_id)
            
            if index_exists:
                try:
                    # 尝试加载现有索引
                    retriever.load()
                    # 需要重新设置 pages 列表（只包含该 session 的页）
                    if session_pages:
                        retriever.pages = session_pages
                    logger.info(f"[PageStore] BM25 索引已加载 (session={session_id}, pages={len(session_pages)})")
                except Exception as e:
                    logger.warning(f"[PageStore] 加载 BM25 索引失败，将重建 (session={session_id}): {e}")
                    # 重建索引
                    if session_pages:
                        retriever.build(session_pages)
                        logger.info(f"[PageStore] BM25 索引重建完成 (session={session_id}, pages={len(session_pages)})")
            else:
                # 索引不存在，构建索引
                if session_pages:
                    logger.info(f"[PageStore] BM25 索引不存在，开始构建 (session={session_id}, pages={len(session_pages)})...")
                    retriever.build(session_pages)
                    logger.info(f"[PageStore] BM25 索引构建完成 (session={session_id})")
                else:
                    logger.warning(f"[PageStore] BM25 索引不存在且没有 pages 可构建 (session={session_id})")
        else:
            # 所有 session 的索引（用于全局搜索）
            # 注意：这种情况较少使用，因为通常搜索会限定 session_id
            all_pages = self.get_all_pages()
            if not all_pages:
                return
            
            # 为所有 session 构建索引
            session_ids = {page.session_id for page in all_pages}
            for sid in session_ids:
                self._ensure_bm25_index(sid)

    def search_pages_by_keywords(
        self, 
        keywords: List[str], 
        session_id: Optional[str] = None, 
        limit: int = 3
    ) -> List[Tuple[Page, float]]:
        """
        根据关键词列表搜索相关的 Page。
        
        策略：
        - 如果启用 BM25：使用 BM25 算法进行检索（更准确）
        - 否则：使用简单的字符串包含匹配 (Case-insensitive)
        
        Args:
            keywords: 关键词列表 (e.g., ["Sweden", "home country"])
            session_id: 可选，限定只搜索某个 session。如果为 None，搜索所有已加载/存储的 session。
            limit: 返回的最大 Page 数量
            
        Returns:
            List[Tuple[Page, score]]: 按分数降序排列的 (Page, 分数) 列表
        """
        # 1. 确定搜索范围
        if session_id:
            # 确保加载
            self._load_session(session_id)
            candidates = self.get_pages_by_session(session_id)
        else:
            # 搜索所有
            candidates = self.get_all_pages()

        if not candidates:
            return []

        # 2. 预处理关键词
        search_terms = [k.strip() for k in keywords if k.strip()]
        if not search_terms:
            return []

        # 3. 如果启用 BM25，使用 BM25 检索
        if self._enable_bm25:
            try:
                # 如果指定了 session_id，使用该 session 的索引
                if session_id:
                    retriever = self._get_bm25_retriever(session_id)
                    if retriever is not None:
                        # 确保索引已构建
                        self._ensure_bm25_index(session_id)

                        # 使用 BM25 检索
                        # 注意：将关键词列表用空格连接成查询字符串
                        # pyserini 的 LuceneSearcher 会自动对查询字符串进行分词和 BM25 检索
                        # 所以可以传入完整的自然语言查询（如 "What is Melanie's hand-painted bowl a reminder of?"）
                        # pyserini 会自动提取关键词并进行检索
                        query = " ".join(search_terms)
                        logger.info(f"[PageStore] BM25 搜索开始: query='{query[:50]}...', session={session_id}, candidates={len(candidates)}")
                        bm25_results = retriever.search(
                            query,
                            top_k=limit * 2,  # 多检索一些，以便后续过滤
                            pages=candidates
                        )
                        logger.info(f"[PageStore] BM25 搜索完成: query='{query[:30]}...', results={len(bm25_results)}")

                        # 按分数降序返回
                        return bm25_results[:limit]
                    else:
                        logger.warning(f"[PageStore] BM25 检索器为 None (session={session_id})，回退到简单匹配")
                else:
                    # 没有指定 session_id，搜索所有 session
                    # 为所有相关 session 构建索引并检索
                    session_ids = {page.session_id for page in candidates}
                    all_results: List[Tuple[Page, float]] = []
                    
                    for sid in session_ids:
                        retriever = self._get_bm25_retriever(sid)
                        if retriever is not None:
                            self._ensure_bm25_index(sid)
                            query = " ".join(search_terms)
                            session_pages = [p for p in candidates if p.session_id == sid]
                            session_results = retriever.search(query, top_k=limit * 2, pages=session_pages)
                            all_results.extend(session_results)
                    
                    # 按分数降序排序并返回
                    all_results.sort(key=lambda x: x[1], reverse=True)
                    return all_results[:limit]
            except Exception as e:
                logger.warning(f"[PageStore] BM25 检索失败，回退到简单匹配: {e}")
                # 回退到简单匹配
                pass
        else:
            logger.debug(f"[PageStore] BM25 未启用 (_enable_bm25={self._enable_bm25})，使用简单匹配")

        # 4. 回退：简单的字符串包含匹配
        logger.debug(f"[PageStore] 使用简单字符串匹配: terms={search_terms}, candidates={len(candidates)}")
        search_terms_lower = {k.lower() for k in search_terms}
        scored_pages: List[Tuple[Page, float]] = []

        for page in candidates:
            if not page.content:
                continue
            
            content_lower = page.content.lower()
            score = 0.0
            
            # 简单打分：匹配到一个词 +1 分
            for term in search_terms_lower:
                if term in content_lower:
                    score += 1.0
            
            if score > 0:
                scored_pages.append((page, score))

        # 5. 排序：分数高在前；分数相同按时间倒序（新的在前）
        scored_pages.sort(key=lambda x: (x[1], x[0].created_at), reverse=True)

        return scored_pages[:limit]
    def get_context_snippet(self, content: str, keywords: List[str], window_size: int = 150) -> str:
        """
        辅助函数：从完整内容中截取包含关键词的片段。
        用于在 Log 或 Token 受限时展示上下文。
        
        如果启用 BM25，会使用 BM25 检索器的 snippet 提取方法（更智能）。
        否则使用简单的字符串匹配。
        
        Args:
            content: 完整文本
            keywords: 关键词列表
            window_size: 关键词前后保留的字符数
            
        Returns:
            截取后的文本片段 (e.g., "...text before [keyword] text after...")
        """
        # 如果启用 BM25 且有检索器，使用 BM25 的 snippet 方法
        if self._enable_bm25 and self._bm25_retriever is not None:
            try:
                # 创建一个临时 Page 对象用于 snippet 提取
                temp_page = Page(
                    page_id="temp",
                    session_id="temp",
                    user_id="temp",
                    content=content
                )
                query = " ".join([k.strip() for k in keywords if k.strip()])
                return self._bm25_retriever.get_snippet(temp_page, query, window_size)
            except Exception as e:
                print(f"[PageStore] 警告：BM25 snippet 提取失败，回退到简单方法: {e}")
                # 回退到简单方法
                pass
        
        # 简单方法：字符串匹配
        content_lower = content.lower()
        search_terms = [k.lower().strip() for k in keywords if k.strip()]
        
        if not search_terms:
            return content[:window_size * 2] + "..." if len(content) > window_size * 2 else content
        
        # 找到第一个匹配关键词的位置
        first_idx = -1
        for term in search_terms:
            idx = content_lower.find(term)
            if idx != -1:
                if first_idx == -1 or idx < first_idx:
                    first_idx = idx
        
        if first_idx == -1:
            # 没找到（理论上不该发生，因为是搜出来的），返回开头
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
    
    def rebuild_bm25_index(self, session_id: Optional[str] = None) -> None:
        """
        重建 BM25 索引（当页内容更新后调用）。
        
        注意：此方法会确保 session 数据已加载后再重建索引，与懒加载机制对齐。
        
        Args:
            session_id: 会话 ID，如果为 None 则重建所有 session 的索引
        """
        if not self._enable_bm25:
            print("[PageStore] BM25 未启用，跳过索引重建")
            return
        
        if session_id:
            # 确保 session 数据已加载（与 get_pages_by_session 的逻辑对齐）
            self._load_session(session_id)
            
            # 重建指定 session 的索引
            retriever = self._get_bm25_retriever(session_id)
            if retriever is None:
                return
            
            # 获取该 session 的所有页（确保数据已加载）
            session_pages = self.get_pages_by_session(session_id)
            if session_pages:
                print(f"[PageStore] 重建 BM25 索引 (session={session_id})，共 {len(session_pages)} 页")
                retriever.build(session_pages)
            else:
                print(f"[PageStore] 没有页需要索引 (session={session_id})")
        else:
            # 重建所有 session 的索引
            all_pages = self.get_all_pages()
            if not all_pages:
                print("[PageStore] 没有页需要索引")
                return
            
            session_ids = {page.session_id for page in all_pages}
            print(f"[PageStore] 重建所有 session 的 BM25 索引，共 {len(session_ids)} 个 session")
            for sid in session_ids:
                self.rebuild_bm25_index(sid)


def main():
    """
    测试函数：直接基于 conv-26.json 构建 BM25 索引，并进行关键词召回测试。
    只对 content 字段做 BM25 索引，每次召回整页的信息。
    索引目录命名为 conv-26_index。
    """
    import json
    from pathlib import Path
    
    # 1. 加载 JSON 文件
    json_file = Path("./tmp/page_store/conv-26.json")
    print(f"[测试] 加载文件: {json_file}")
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. 提取所有页（包括 current_page 和 completed_pages）
    pages = []
    
    # 添加 current_page（如果存在）
    if "current_page" in data and data["current_page"]:
        pages.append(Page.from_dict(data["current_page"]))
        print(f"[测试] 加载当前页: {data['current_page']['page_id']}")
    
    # 添加 completed_pages
    if "completed_pages" in data:
        for page_data in data["completed_pages"]:
            pages.append(Page.from_dict(page_data))
        print(f"[测试] 加载已完成页: {len(data['completed_pages'])} 页")
    
    print(f"[测试] 总共加载 {len(pages)} 页")
    
    # 3. 创建 BM25 检索器并构建索引（只对 content 字段）
    # 索引目录命名为 conv-26_index，放在 JSON 文件所在目录
    if not BM25_AVAILABLE or PageStoreBM25Retriever is None:
        print("[错误] BM25 不可用，请安装 pyserini: pip install pyserini")
        return
    
    # 确保 PageStoreBM25Retriever 已导入
    PageStoreBM25Retriever_class = PageStoreBM25Retriever
    
    # 索引目录：conv-26_index（在 JSON 文件所在目录）
    index_dir = json_file.parent / "conv-26_index"
    print(f"[测试] 索引目录: {index_dir}")
    
    bm25_retriever = PageStoreBM25Retriever_class({
        "index_dir": str(index_dir),
        "threads": 4
    })
    
    print("\n[测试] 开始构建 BM25 索引（只对 content 字段）...")
    bm25_retriever.build(pages)
    print("[测试] BM25 索引构建完成\n")
    
    # 4. 测试关键词召回和自然语言查询
    test_queries = [
        # 关键词查询
        {
            "type": "关键词",
            "query": "LGBTQ support group",
            "keywords": ["LGBTQ", "support group"]
        },
        # 自然语言查询
        {
            "type": "自然语言",
            "query": "What is Melanie's hand-painted bowl a reminder of?",
            "keywords": None
        }
    ]
    
    print("=" * 80)
    print("BM25 检索测试结果（每次召回整页信息）")
    print("=" * 80)
    
    for test_item in test_queries:
        query_type = test_item["type"]
        query = test_item["query"]
        
        print(f"\n查询类型: {query_type}")
        print(f"查询内容: {query}")
        print("-" * 80)
        
        # 执行 BM25 检索（直接使用查询字符串）
        results = bm25_retriever.search(query, top_k=5, pages=pages)
        
        print(f"找到 {len(results)} 个相关页:\n")
        
        for i, (page, score) in enumerate(results, 1):
            print(f"结果 {i} (BM25 分数: {score:.4f}):")
            print(f"  Page ID: {page.page_id}")
            print(f"  Session ID: {page.session_id}")
            print(f"  创建时间: {page.created_at}")
            print(f"  内容长度: {len(page.content)} 字符")
            
            # 显示完整内容（整页信息）
            print(f"  完整内容:")
            print(f"  {'-' * 76}")
            # 按行显示，每行最多76字符
            content_lines = page.content.split('\n')
            for line in content_lines[:15]:  # 显示前15行
                if len(line) > 76:
                    print(f"  {line[:73]}...")
                else:
                    print(f"  {line}")
            if len(content_lines) > 15:
                print(f"  ... (还有 {len(content_lines) - 15} 行)")
            print(f"  {'-' * 76}")
            
            # 如果有 summary，也显示
            if page.summary:
                print(f"  总结: {page.summary}")
            
            print()
        
        print("-" * 80)
    
    print("\n[测试] 完成！")


if __name__ == "__main__":
    main()


__all__ = ["Page", "PageStore"]
