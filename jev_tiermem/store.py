"""Local raw pages and plain Markdown notes; no vector database or ML imports."""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


def tokens(text):
    # Index CJK characters separately so Chinese queries do not need a tokenizer model.
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def atomic_write(path, text):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memory-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Store:
    NOTE = re.compile(r"<!-- jev_tiermem (\{[^\n]*\}) -->\n(.*?)\n<!-- /jev_tiermem -->", re.S)
    HEADER = "# Memory\n\n"

    def __init__(self, directory, session, chunk_chars=3000):
        if not isinstance(session, str) or not session.strip():
            raise ValueError("session must be a nonempty string")
        self.directory = Path(directory) / digest(session)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.directory / "MEMORY.md"
        self.db_path = self.directory / "raw.sqlite3"
        self.chunk_chars = chunk_chars
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('session', ?)", (session,))
            db.execute("""CREATE TABLE IF NOT EXISTS raw (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                observation_id TEXT NOT NULL, part INTEGER NOT NULL, speaker TEXT NOT NULL,
                timestamp TEXT, dia_id TEXT, text TEXT NOT NULL)""")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS raw_fts USING fts5(id UNINDEXED, terms)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def observe(self, text, speaker="user", timestamp=None, dia_id=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("observation text must be nonempty")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker must be nonempty")
        for name, value in (("timestamp", timestamp), ("dia_id", dia_id)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        observation_id = digest(json.dumps([speaker, timestamp, dia_id, text], ensure_ascii=False))
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT id FROM raw WHERE observation_id=? ORDER BY part", (observation_id,)
            ).fetchall()
            if existing:
                return [row["id"] for row in existing]
            ids = []
            for part, start in enumerate(range(0, len(text), self.chunk_chars)):
                chunk = text[start:start + self.chunk_chars]
                raw_id = "r-" + digest(f"{observation_id}:{part}")
                db.execute(
                    "INSERT INTO raw (id, observation_id, part, speaker, timestamp, dia_id, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (raw_id, observation_id, part, speaker, timestamp, dia_id, chunk),
                )
                db.execute("INSERT INTO raw_fts VALUES (?, ?)",
                           (raw_id, " ".join(tokens(speaker + " " + (timestamp or "") + " " + chunk))))
                ids.append(raw_id)
        return ids

    def get(self, ids):
        with self.connection() as db:
            return [dict(row) for raw_id in dict.fromkeys(ids)
                    if (row := db.execute("SELECT * FROM raw WHERE id=?", (raw_id,)).fetchone())]

    def search(self, query, limit=4, exclude=()):
        terms = list(dict.fromkeys(tokens(query)))[:64]
        if not terms or limit <= 0:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        excluded = set(exclude)
        with self.connection() as db:
            rows = db.execute(
                "SELECT raw.* FROM raw_fts JOIN raw ON raw.id=raw_fts.id "
                "WHERE raw_fts MATCH ? ORDER BY bm25(raw_fts), raw.seq LIMIT ?",
                (expression, limit + len(excluded)),
            ).fetchall()
        return [dict(row) for row in rows if row["id"] not in excluded][:limit]

    def _read_notes(self):
        text = self.memory_path.read_text(encoding="utf-8") if self.memory_path.exists() else self.HEADER
        notes = []
        for match in self.NOTE.finditer(text):
            metadata = json.loads(match.group(1))
            if (not isinstance(metadata.get("id"), str)
                    or not isinstance(metadata.get("raw_ids"), list)
                    or not all(isinstance(value, str) for value in metadata["raw_ids"])):
                raise ValueError("Invalid MEMORY.md source metadata")
            notes.append({**metadata, "text": match.group(2).strip()})
        # Do not silently discard hand edits outside a managed block.
        remainder = self.NOTE.sub("", text).replace(self.HEADER, "", 1).strip()
        if remainder:
            raise ValueError("Edit note bodies inside the MEMORY.md markers; use add_summary for new notes")
        return text, notes

    def notes(self):
        return self._read_notes()[1]

    def summary_hits(self, query, limit):
        terms = Counter(tokens(query))
        notes = self.notes()

        def rank(item):
            counts = Counter(tokens(item[1]["text"]))
            return sum(min(count, counts[term]) for term, count in terms.items()), item[0]

        ranked = sorted(enumerate(notes), key=rank, reverse=True)
        return [note for _, note in ranked[:limit]]

    def append_note(self, text, raw_ids, kind, max_chars):
        if not isinstance(text, str) or not text.strip():
            return "empty"
        if "<!--" in text or "-->" in text:
            return "invalid_markers"
        raw_ids = list(dict.fromkeys(raw_ids))
        if not raw_ids or len(self.get(raw_ids)) != len(raw_ids):
            return "invalid_sources"
        # Serialize writers in different processes with SQLite's write lock.
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current, notes = self._read_notes()
            normalized = " ".join(text.lower().split())
            duplicate = next((note for note in notes
                              if " ".join(note["text"].lower().split()) == normalized), None)
            if duplicate is not None:
                # Identical summaries of later records still need source coverage; otherwise
                # compact() would keep retrying the same pending batch forever.
                if kind != "summary":
                    return "duplicate"
                combined = list(dict.fromkeys(duplicate["raw_ids"] + raw_ids))
                if combined == duplicate["raw_ids"] and duplicate.get("kind") == "summary":
                    return "duplicate"
                metadata = {"id": duplicate["id"], "raw_ids": combined, "kind": "summary"}

                def update(match):
                    if json.loads(match.group(1))["id"] != duplicate["id"]:
                        return match.group(0)
                    return ("<!-- jev_tiermem " + json.dumps(metadata, ensure_ascii=False) + " -->\n"
                            + match.group(2) + "\n<!-- /jev_tiermem -->")

                revised = self.NOTE.sub(update, current)
                if len(revised) > max_chars:
                    return "summary_budget"
                atomic_write(self.memory_path, revised)
                return "merged"
            metadata = {"id": "s-" + digest(text + json.dumps(raw_ids)), "raw_ids": raw_ids, "kind": kind}
            block = ("<!-- jev_tiermem " + json.dumps(metadata, ensure_ascii=False) + " -->\n"
                     + text.strip() + "\n<!-- /jev_tiermem -->\n\n")
            if len(current) + len(block) > max_chars:
                return "summary_budget"
            atomic_write(self.memory_path, current + block)
        return "written"

    def pending_batch(self, max_chars):
        covered = {raw_id for note in self.notes() if note.get("kind") == "summary"
                   for raw_id in note["raw_ids"]}
        batch, size = [], 0
        with self.connection() as db:
            for row in db.execute("SELECT * FROM raw ORDER BY seq"):
                if row["id"] in covered:
                    continue
                if size + len(row["text"]) > max_chars:
                    break
                batch.append(dict(row))
                size += len(row["text"])
        return batch
