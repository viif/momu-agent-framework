"""文档存储实现

支持的数据库后端：
- SQLite: 轻量级关系型数据库
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import aiosqlite

from ..core.exceptions import StorageException
from ..utils.logger import get_logger


class DocumentStore(ABC):
    """文档存储抽象基类"""

    @abstractmethod
    async def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """添加或替换一条记录，返回 memory_id。"""

    @abstractmethod
    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """按 ID 获取单条记录，不存在时返回 None。"""

    @abstractmethod
    async def search_memories(
        self,
        user_id: str | None = None,
        memory_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        importance_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按条件过滤记录，结果按 importance DESC、timestamp DESC 排序。"""

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """更新记录的指定字段，返回是否命中记录。三个字段均为 None 时直接返回 False。"""

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """删除指定记录，返回是否命中记录。"""

    @abstractmethod
    async def get_database_stats(self) -> dict[str, Any]:
        """返回数据库统计信息，包括各表记录数、类型分布和活跃用户 Top 10。"""

    @abstractmethod
    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """将内容作为 document 类型记录写入，返回自动生成的 UUID。"""

    @abstractmethod
    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """按 ID 获取文档记录，不存在时返回 None。"""

    @abstractmethod
    async def close(self) -> None:
        """关闭文档存储连接并释放资源。"""


class SQLiteDocumentStore(DocumentStore):
    """SQLite 文档存储实现（基于 aiosqlite）"""

    _DEFAULT_DB_PATH = os.path.join(os.getcwd(), ".storage", "document.db")
    _instances: dict[str, "SQLiteDocumentStore"] = {}

    def __new__(cls, db_path: str | None = None) -> "SQLiteDocumentStore":
        abs_path = os.path.abspath(db_path or cls._DEFAULT_DB_PATH)
        if abs_path not in cls._instances:
            cls._instances[abs_path] = super().__new__(cls)
        return cls._instances[abs_path]

    def __init__(self, db_path: str | None = None) -> None:
        if hasattr(self, "_initialized"):
            return
        self.db_path = db_path or self._DEFAULT_DB_PATH
        self._conn: aiosqlite.Connection | None = None
        self._db_ready = False
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._initialized = True

        self.logger = get_logger(__name__)
        self.logger.debug(
            f"🧠 SQLiteDocumentStore 初始化完成 (db_path: {self.db_path})"
        )

    async def _ensure_db(self) -> None:
        if self._db_ready:
            return
        async with self._init_lock:
            if self._db_ready:
                return
            db = await aiosqlite.connect(self.db_path)
            db.row_factory = aiosqlite.Row
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id         TEXT PRIMARY KEY,
                    name       TEXT,
                    properties TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    memory_type TEXT    NOT NULL,
                    timestamp   INTEGER NOT NULL,
                    importance  REAL    NOT NULL,
                    properties  TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            for sql in [
                "CREATE INDEX IF NOT EXISTS idx_memories_user_id    ON memories (user_id)",
                "CREATE INDEX IF NOT EXISTS idx_memories_type       ON memories (memory_type)",
                "CREATE INDEX IF NOT EXISTS idx_memories_timestamp  ON memories (timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            ]:
                await db.execute(sql)
            await db.commit()
            self._conn = db
            self._db_ready = True
            self.logger.info(f"🧠 SQLite 数据库初始化完成: {self.db_path}")

    async def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """添加或替换一条记录，返回 memory_id。"""
        try:
            await self._ensure_db()
            assert self._conn is not None
            async with self._write_lock:
                await self._conn.execute(
                    "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
                    (user_id, user_id),
                )
                await self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                        (id, user_id, content, memory_type, timestamp, importance,
                         properties, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        memory_id,
                        user_id,
                        content,
                        memory_type,
                        timestamp,
                        importance,
                        json.dumps(properties) if properties else None,
                    ),
                )
                await self._conn.commit()
            self.logger.debug(
                f"🧠 写入记录 [{memory_id}] (user: '{user_id}', type: '{memory_type}', "
                f"importance: {importance})"
            )
            return memory_id
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 add_memory 失败: {e}")
            raise StorageException(f"add_memory failed: {e}") from e

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """按 ID 获取单条记录，不存在时返回 None。"""
        try:
            await self._ensure_db()
            assert self._conn is not None
            async with self._conn.execute(
                """
                SELECT id, user_id, content, memory_type, timestamp,
                       importance, properties, created_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                self.logger.debug(f"🧠 未找到记录 [{memory_id}]")
                return None
            return {
                "memory_id": row["id"],
                "user_id": row["user_id"],
                "content": row["content"],
                "memory_type": row["memory_type"],
                "timestamp": row["timestamp"],
                "importance": row["importance"],
                "properties": json.loads(row["properties"])
                if row["properties"]
                else {},
                "created_at": row["created_at"],
            }
        except StorageException:
            raise
        except Exception as e:
            raise StorageException(f"get_memory failed: {e}") from e

    async def search_memories(
        self,
        user_id: str | None = None,
        memory_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        importance_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按条件过滤记录，结果按 importance DESC、timestamp DESC 排序。"""
        try:
            await self._ensure_db()
            assert self._conn is not None
            conditions: list[str] = []
            params: list[Any] = []
            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(user_id)
            if memory_type is not None:
                conditions.append("memory_type = ?")
                params.append(memory_type)
            if start_time is not None:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time is not None:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            if importance_threshold is not None:
                conditions.append("importance >= ?")
                params.append(importance_threshold)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            async with self._conn.execute(
                f"""
                SELECT id, user_id, content, memory_type, timestamp,
                       importance, properties, created_at
                FROM memories
                {where}
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                params + [limit],
            ) as cursor:
                rows = await cursor.fetchall()
            results = [
                {
                    "memory_id": row["id"],
                    "user_id": row["user_id"],
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "timestamp": row["timestamp"],
                    "importance": row["importance"],
                    "properties": json.loads(row["properties"])
                    if row["properties"]
                    else {},
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
            self.logger.debug(
                f"🧠 搜索记录，命中 {len(results)} 条 "
                f"(user: {user_id!r}, type: {memory_type!r}, limit: {limit})"
            )
            return results
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 search_memories 失败: {e}")
            raise StorageException(f"search_memories failed: {e}") from e

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """更新记录的指定字段，返回是否命中记录。三个字段均为 None 时直接返回 False。"""
        if content is None and importance is None and properties is None:
            return False
        try:
            await self._ensure_db()
            assert self._conn is not None
            fields: list[str] = []
            params: list[Any] = []
            if content is not None:
                fields.append("content = ?")
                params.append(content)
            if importance is not None:
                fields.append("importance = ?")
                params.append(importance)
            if properties is not None:
                fields.append("properties = ?")
                params.append(json.dumps(properties))
            fields.append("updated_at = CURRENT_TIMESTAMP")
            params.append(memory_id)
            async with self._write_lock:
                cursor = await self._conn.execute(
                    f"UPDATE memories SET {', '.join(fields)} WHERE id = ?",
                    params,
                )
                await self._conn.commit()
                hit = cursor.rowcount > 0
            if hit:
                self.logger.debug(f"🧠 更新记录 [{memory_id}] 成功")
            else:
                self.logger.warning(f"🧠 更新记录失败，未找到 [{memory_id}]")
            return hit
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 update_memory 失败: {e}")
            raise StorageException(f"update_memory failed: {e}") from e

    async def delete_memory(self, memory_id: str) -> bool:
        """删除指定记录，返回是否命中记录。"""
        try:
            await self._ensure_db()
            assert self._conn is not None
            async with self._write_lock:
                cursor = await self._conn.execute(
                    "DELETE FROM memories WHERE id = ?", (memory_id,)
                )
                await self._conn.commit()
                hit = cursor.rowcount > 0
            if hit:
                self.logger.debug(f"🧠 删除记录 [{memory_id}] 成功")
            else:
                self.logger.warning(f"🧠 删除记录失败，未找到 [{memory_id}]")
            return hit
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 delete_memory 失败: {e}")
            raise StorageException(f"delete_memory failed: {e}") from e

    async def get_database_stats(self) -> dict[str, Any]:
        """返回数据库统计信息，包括各表记录数、类型分布和活跃用户 Top 10。"""
        try:
            await self._ensure_db()
            assert self._conn is not None
            async with self._conn.execute("SELECT COUNT(*) as count FROM users") as cur:
                row = await cur.fetchone()
                users_count = row["count"] if row else 0
            async with self._conn.execute(
                "SELECT COUNT(*) as count FROM memories"
            ) as cur:
                row = await cur.fetchone()
                memories_count = row["count"] if row else 0
            async with self._conn.execute(
                "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
            ) as cur:
                memory_types = {
                    row["memory_type"]: row["count"] for row in await cur.fetchall()
                }
            async with self._conn.execute(
                """
                SELECT user_id, COUNT(*) as count
                FROM memories
                GROUP BY user_id
                ORDER BY count DESC
                LIMIT 10
                """
            ) as cur:
                top_users = {
                    row["user_id"]: row["count"] for row in await cur.fetchall()
                }
            return {
                "users_count": users_count,
                "memories_count": memories_count,
                "memory_types": memory_types,
                "top_users": top_users,
                "store_type": "sqlite",
                "db_path": self.db_path,
            }
        except StorageException:
            raise
        except Exception as e:
            raise StorageException(f"get_database_stats failed: {e}") from e

    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """将内容作为 document 类型记录写入，返回自动生成的 UUID。"""
        import time

        doc_id = str(uuid4())
        user_id = (metadata or {}).get("user_id", "system")
        return await self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {},
        )

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """按 ID 获取文档记录，不存在时返回 None。"""
        return await self.get_memory(document_id)

    async def close(self) -> None:
        """关闭持久连接并重置初始化状态。"""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._db_ready = False
            self.logger.info(f"🧠 SQLite 连接已关闭: {self.db_path}")
