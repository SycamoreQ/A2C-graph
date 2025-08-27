import sqlite3
import os
import re
from datetime import datetime
from typing import Any, Iterator
from pipeline_storage import PipelineStorage  

class SQLStore(PipelineStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._ensure_table()

    def _ensure_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS storage (
            id TEXT PRIMARY KEY,
            value TEXT,
            created_at TEXT
        )
        """)
        self.conn.commit()

    def find(self, file_pattern: re.Pattern[str], base_dir: str = None, file_filter: dict[str, Any] = None, max_count=-1) -> Iterator[tuple[str, dict[str, Any]]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, value FROM storage")
        count = 0
        for row in cursor.fetchall():
            match = file_pattern.search(row)
            if match:
                if file_filter is None or all(str(match.groupdict().get(k, "")) == v for k,v in (file_filter or {}).items()):
                    yield (row, match.groupdict())
                    count += 1
                    if max_count > 0 and count >= max_count:
                        break

    async def get(self, key: str, as_bytes: bool = None, encoding: str = None) -> Any:
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM storage WHERE id = ?", (key,))
        row = cursor.fetchone()
        if row:
            return row.encode(encoding or "utf-8") if as_bytes else row
        return None

    async def set(self, key: str, value: Any, encoding: str = None) -> None:
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        if isinstance(value, bytes):
            value = value.decode(encoding or "utf-8")
        cursor.execute("INSERT OR REPLACE INTO storage (id, value, created_at) VALUES (?, ?, ?)", (key, value, now))
        self.conn.commit()

    async def has(self, key: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM storage WHERE id = ?", (key,))
        return bool(cursor.fetchone())

    async def delete(self, key: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM storage WHERE id = ?", (key,))
        self.conn.commit()

    async def clear(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM storage")
        self.conn.commit()

    def keys(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM storage")
        return [row for row in cursor.fetchall()]
        
    def child(self, name: str = None) -> 'SQLStore':
        # For SQLite, you could use a prefixed table or schema, or just return self
        return self

    async def get_creation_date(self, key: str) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT created_at FROM storage WHERE id = ?", (key,))
        row = cursor.fetchone()
        if row:
            return row
        return ""
    
    