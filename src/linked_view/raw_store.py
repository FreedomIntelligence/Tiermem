"""
RawStore：append-only 原始对话存储。

满足规范：
- 单一写入流：每个 history chunk 只写一次 RawStore
- 通过 raw_log_id 作为指针，被 SummaryIndex metadata 引用
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path


class RawLogRecord:
    """单条原始对话记录。"""

    def __init__(self, raw_log_id: str, user_id: str, timestamp: str, session_id: Optional[str], text: str, meta: Dict):
        self.raw_log_id = raw_log_id
        self.user_id = user_id
        self.timestamp = timestamp  # ISO string
        self.session_id = session_id
        self.text = text
        self.meta = meta  # 例如 turn range, speakers 等

    def to_dict(self) -> Dict:
        """转换为字典以便序列化。"""
        return {
            "raw_log_id": self.raw_log_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "text": self.text,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RawLogRecord":
        """从字典反序列化。"""
        return cls(
            raw_log_id=data["raw_log_id"],
            user_id=data["user_id"],
            timestamp=data["timestamp"],
            session_id=data.get("session_id"),
            text=data["text"],
            meta=data.get("meta", {}),
        )


class RawStore:
    """
    持久化 RawStore 实现，存储到本地文件系统。

    - 每个 session_id 对应一个 JSON 文件
    - 文件路径：{storage_dir}/{session_id}.json
    - 支持按 session_id 删除数据
    """

    def __init__(self, storage_dir: str = "./tmp/raw_store") -> None:
        """
        初始化 RawStore。

        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # 内存缓存：按 session_id 组织的记录
        self._records_cache: Dict[str, Dict[str, RawLogRecord]] = {}

    def _get_session_file(self, session_id: str) -> Path:
        """获取 session 对应的文件路径。"""
        # 使用安全的文件名（移除特殊字符）
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_session_id}.json"

    def _load_session(self, session_id: str) -> Dict[str, RawLogRecord]:
        """从文件加载指定 session 的所有记录。"""
        if session_id in self._records_cache:
            return self._records_cache[session_id]

        session_file = self._get_session_file(session_id)
        records: Dict[str, RawLogRecord] = {}

        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for record_data in data.get("records", []):
                        record = RawLogRecord.from_dict(record_data)
                        records[record.raw_log_id] = record
            except Exception as e:
                # 文件损坏时返回空字典，避免中断流程
                print(f"[RawStore] Failed to load session {session_id}: {e}")

        self._records_cache[session_id] = records
        return records

    def _save_session(self, session_id: str, records: Dict[str, RawLogRecord]) -> None:
        """将指定 session 的所有记录保存到文件。"""
        session_file = self._get_session_file(session_id)
        try:
            data = {
                "session_id": session_id,
                "records": [record.to_dict() for record in records.values()],
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[RawStore] Failed to save session {session_id}: {e}")

    def _get_all_records(self) -> Dict[str, RawLogRecord]:
        """获取所有 session 的记录（用于兼容旧接口）。"""
        all_records: Dict[str, RawLogRecord] = {}
        # 遍历存储目录中的所有 JSON 文件
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for record_data in data.get("records", []):
                        record = RawLogRecord.from_dict(record_data)
                        all_records[record.raw_log_id] = record
            except Exception:
                continue
        return all_records

    # === 写入 ===
    def put(self, record: RawLogRecord) -> None:
        """
        追加写入一条 RawLogRecord。

        若 same raw_log_id 已存在，则覆盖视为 bug；但这里仍然直接覆盖，以免中断主流程。
        """
        if not record.session_id:
            # 如果没有 session_id，使用 "default" 作为默认值
            session_id = "default"
        else:
            session_id = record.session_id

        records = self._load_session(session_id)
        records[record.raw_log_id] = record
        self._records_cache[session_id] = records
        self._save_session(session_id, records)

    # === 读取 ===
    def get(self, raw_log_id: str) -> Optional[RawLogRecord]:
        """按 raw_log_id 读取单条记录；不存在时返回 None。"""
        # 需要遍历所有 session 查找
        all_records = self._get_all_records()
        return all_records.get(raw_log_id)

    def batch_get(self, ids: List[str]) -> List[RawLogRecord]:
        """按给定 id 列表批量读取，自动跳过不存在的 id。"""
        all_records = self._get_all_records()
        results: List[RawLogRecord] = []
        for _id in ids:
            rec = all_records.get(_id)
            if rec is not None:
                results.append(rec)
        return results

    def count(self) -> int:
        """获取所有记录的总数。"""
        all_records = self._get_all_records()
        return len(all_records)

    # === 删除 ===
    def delete_by_session(self, session_id: str) -> None:
        """
        删除指定 session_id 的所有记录。

        Args:
            session_id: 要删除的 session ID
        """
        # 删除内存缓存
        if session_id in self._records_cache:
            del self._records_cache[session_id]

        # 删除文件
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            try:
                session_file.unlink()
            except Exception as e:
                print(f"[RawStore] Failed to delete session file {session_file}: {e}")


__all__ = ["RawLogRecord", "RawStore"]


