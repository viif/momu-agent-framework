"""图数据库存储实现"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import kuzu

from ..core.exceptions import MemoryException
from ..utils.logger import get_logger


class GraphStore(ABC):
    """图数据库存储抽象基类"""

    @abstractmethod
    async def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """添加或更新实体节点。"""

    @abstractmethod
    async def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """添加或更新实体关系。"""

    @abstractmethod
    async def find_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按路径深度查找相关实体。"""

    @abstractmethod
    async def search_entities_by_name(
        self,
        name_pattern: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """按名称模式搜索实体。"""

    @abstractmethod
    async def get_entity_relationships(self, entity_id: str) -> list[dict[str, Any]]:
        """获取实体的所有关系。"""

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """删除实体及其关系。"""

    @abstractmethod
    async def clear_all(self) -> bool:
        """清空图数据。"""

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """获取图数据库统计信息。"""

    @abstractmethod
    async def close(self) -> None:
        """关闭连接并释放资源。"""


class KuzuGraphStore(GraphStore):
    """Kuzu 图数据库存储实现"""

    _DEFAULT_DB_PATH = str(Path(__file__).parent / "graph.kuzu")
    _instances: dict[str, "KuzuGraphStore"] = {}

    def __new__(cls, db_path: str | None = None) -> "KuzuGraphStore":
        path = db_path or cls._DEFAULT_DB_PATH
        abs_path = path if path == ":memory:" else os.path.abspath(path)
        if abs_path not in cls._instances:
            cls._instances[abs_path] = super().__new__(cls)
        return cls._instances[abs_path]

    def __init__(self, db_path: str | None = None, num_threads: int = 0) -> None:
        if hasattr(self, "_initialized"):
            return
        self.db_path = db_path or self._DEFAULT_DB_PATH
        self.num_threads = num_threads
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._ready = False
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._initialized = True

        self.logger = get_logger(__name__)
        self.logger.debug(f"🧠 KuzuGraphStore 初始化完成 (db_path: {self.db_path})")

    async def _run_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_db()
        conn = self._conn
        assert conn is not None

        def _execute() -> list[dict[str, Any]]:
            result = conn.execute(query, parameters or {})
            if isinstance(result, list):
                rows: list[dict[str, Any]] = []
                for item in result:
                    for row in item.rows_as_dict().get_all():
                        if isinstance(row, dict):
                            rows.append(row)
                return rows

            rows: list[dict[str, Any]] = []
            for row in result.rows_as_dict().get_all():
                if isinstance(row, dict):
                    rows.append(row)
            return rows

        return await asyncio.to_thread(_execute)

    async def _run_query_no_init(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        conn = self._conn
        assert conn is not None

        def _execute() -> list[dict[str, Any]]:
            result = conn.execute(query, parameters or {})
            if isinstance(result, list):
                rows: list[dict[str, Any]] = []
                for item in result:
                    for row in item.rows_as_dict().get_all():
                        if isinstance(row, dict):
                            rows.append(row)
                return rows

            rows: list[dict[str, Any]] = []
            for row in result.rows_as_dict().get_all():
                if isinstance(row, dict):
                    rows.append(row)
            return rows

        return await asyncio.to_thread(_execute)

    async def _ensure_db(self) -> None:
        if self._ready:
            return
        async with self._init_lock:
            if self._ready:
                return
            db_path = self.db_path
            if db_path != ":memory:":
                db_path = os.path.abspath(db_path)
            self._db = await asyncio.to_thread(kuzu.Database, db_path)
            self._conn = await asyncio.to_thread(
                kuzu.Connection, self._db, self.num_threads
            )
            await self._ensure_schema()
            self._ready = True
            self.logger.info(f"🧠 Kuzu 数据库初始化完成: {self.db_path}")

    async def _ensure_schema(self) -> None:
        await self._run_query_no_init(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING,
                name STRING,
                type STRING,
                properties STRING,
                created_at INT64,
                updated_at INT64,
                PRIMARY KEY(id)
            )
            """
        )
        await self._run_query_no_init(
            """
            CREATE REL TABLE IF NOT EXISTS RELATES(
                FROM Entity TO Entity,
                rel_type STRING,
                properties STRING,
                created_at INT64,
                updated_at INT64
            )
            """
        )

    @staticmethod
    def _serialize_properties(properties: dict[str, Any] | None) -> str:
        return json.dumps(properties or {}, ensure_ascii=False)

    @staticmethod
    def _deserialize_properties(raw: Any) -> dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _extract_entity(entity_value: Any) -> dict[str, Any]:
        if not isinstance(entity_value, dict):
            return {}
        return {
            "id": entity_value.get("id", ""),
            "name": entity_value.get("name", ""),
            "type": entity_value.get("type", ""),
            "properties": KuzuGraphStore._deserialize_properties(
                entity_value.get("properties")
            ),
            "created_at": entity_value.get("created_at"),
            "updated_at": entity_value.get("updated_at"),
        }

    async def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        try:
            now = int(time.time())
            await self._run_query(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.properties = $properties,
                    e.created_at = COALESCE(e.created_at, $now),
                    e.updated_at = $now
                """,
                {
                    "entity_id": entity_id,
                    "name": name,
                    "entity_type": entity_type,
                    "properties": self._serialize_properties(properties),
                    "now": now,
                },
            )
            return True
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 add_entity 失败: {e}")
            raise MemoryException(f"add_entity failed: {e}") from e

    async def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        try:
            now = int(time.time())
            async with self._write_lock:
                rows = await self._run_query(
                    """
                    MATCH (a:Entity {id: $from_id}), (b:Entity {id: $to_id})
                    RETURN COUNT(a) AS from_count, COUNT(b) AS to_count
                    """,
                    {"from_id": from_entity_id, "to_id": to_entity_id},
                )
                if not rows:
                    return False
                from_count = int(rows[0].get("from_count", 0))
                to_count = int(rows[0].get("to_count", 0))
                if from_count == 0 or to_count == 0:
                    return False

                await self._run_query(
                    """
                    MATCH (a:Entity {id: $from_id})
                        -[r:RELATES {rel_type: $rel_type}]->
                        (b:Entity {id: $to_id})
                    DELETE r
                    """,
                    {
                        "from_id": from_entity_id,
                        "to_id": to_entity_id,
                        "rel_type": relationship_type,
                    },
                )
                await self._run_query(
                    """
                    MATCH (a:Entity {id: $from_id}), (b:Entity {id: $to_id})
                    CREATE (a)-[:RELATES {
                        rel_type: $rel_type,
                        properties: $properties,
                        created_at: $now,
                        updated_at: $now
                    }]->(b)
                    """,
                    {
                        "from_id": from_entity_id,
                        "to_id": to_entity_id,
                        "rel_type": relationship_type,
                        "properties": self._serialize_properties(properties),
                        "now": now,
                    },
                )
            return True
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 add_relationship 失败: {e}")
            raise MemoryException(f"add_relationship failed: {e}") from e

    async def find_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        try:
            if max_depth <= 0 or limit <= 0:
                return []

            rows = await self._run_query(
                f"""
                MATCH p=(start:Entity {{id: $entity_id}})-[r:RELATES*1..{max_depth}]-(related:Entity)
                WHERE related.id <> $entity_id
                RETURN related AS related,
                       relationships(p) AS relationship_path,
                       length(p) AS distance
                ORDER BY distance, related.name
                LIMIT $limit
                """,
                {"entity_id": entity_id, "limit": limit},
            )

            allowed_types = set(relationship_types or [])
            deduped: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

            for row in rows:
                related = self._extract_entity(row.get("related"))
                if not related.get("id"):
                    continue
                path = row.get("relationship_path") or []
                rel_types: list[str] = []
                for rel in path:
                    if isinstance(rel, dict):
                        rel_type = str(rel.get("rel_type", ""))
                        if rel_type:
                            rel_types.append(rel_type)
                if allowed_types and any(rel not in allowed_types for rel in rel_types):
                    continue

                result = dict(related)
                result["distance"] = int(row.get("distance", 0))
                result["relationship_path"] = rel_types
                key = (result["id"], tuple(rel_types))
                current = deduped.get(key)
                if current is None or result["distance"] < current["distance"]:
                    deduped[key] = result

            items = list(deduped.values())
            items.sort(key=lambda x: (int(x["distance"]), str(x.get("name", ""))))
            return items[:limit]
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 find_related_entities 失败: {e}")
            raise MemoryException(f"find_related_entities failed: {e}") from e

    async def search_entities_by_name(
        self,
        name_pattern: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        try:
            if limit <= 0:
                return []
            rows = await self._run_query(
                """
                MATCH (e:Entity)
                WHERE e.name =~ $pattern
                RETURN e AS entity
                ORDER BY e.name
                LIMIT $limit
                """,
                {"pattern": f".*{name_pattern}.*", "limit": limit},
            )

            allowed_types = set(entity_types or [])
            entities: list[dict[str, Any]] = []
            for row in rows:
                entity = self._extract_entity(row.get("entity"))
                if not entity.get("id"):
                    continue
                if allowed_types and entity.get("type") not in allowed_types:
                    continue
                entities.append(entity)
            return entities
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 search_entities_by_name 失败: {e}")
            raise MemoryException(f"search_entities_by_name failed: {e}") from e

    async def get_entity_relationships(self, entity_id: str) -> list[dict[str, Any]]:
        try:
            outgoing_rows = await self._run_query(
                """
                MATCH (e:Entity {id: $entity_id})-[r:RELATES]->(other:Entity)
                RETURN r AS rel, other AS other
                """,
                {"entity_id": entity_id},
            )
            incoming_rows = await self._run_query(
                """
                MATCH (e:Entity {id: $entity_id})<-[r:RELATES]-(other:Entity)
                RETURN r AS rel, other AS other
                """,
                {"entity_id": entity_id},
            )

            relationships: list[dict[str, Any]] = []
            for row in outgoing_rows:
                rel_obj = row.get("rel")
                rel: dict[str, Any] = rel_obj if isinstance(rel_obj, dict) else {}
                relationships.append(
                    {
                        "relationship": {
                            "type": rel.get("rel_type", ""),
                            "properties": self._deserialize_properties(
                                rel.get("properties")
                            ),
                            "created_at": rel.get("created_at"),
                            "updated_at": rel.get("updated_at"),
                        },
                        "other_entity": self._extract_entity(row.get("other")),
                        "direction": "outgoing",
                    }
                )

            for row in incoming_rows:
                rel_obj = row.get("rel")
                rel: dict[str, Any] = rel_obj if isinstance(rel_obj, dict) else {}
                relationships.append(
                    {
                        "relationship": {
                            "type": rel.get("rel_type", ""),
                            "properties": self._deserialize_properties(
                                rel.get("properties")
                            ),
                            "created_at": rel.get("created_at"),
                            "updated_at": rel.get("updated_at"),
                        },
                        "other_entity": self._extract_entity(row.get("other")),
                        "direction": "incoming",
                    }
                )

            return relationships
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 get_entity_relationships 失败: {e}")
            raise MemoryException(f"get_entity_relationships failed: {e}") from e

    async def delete_entity(self, entity_id: str) -> bool:
        try:
            async with self._write_lock:
                exists_rows = await self._run_query(
                    """
                    MATCH (e:Entity {id: $entity_id})
                    RETURN COUNT(e) AS count
                    """,
                    {"entity_id": entity_id},
                )
                if not exists_rows or int(exists_rows[0].get("count", 0)) == 0:
                    return False

                await self._run_query(
                    """
                    MATCH (e:Entity {id: $entity_id})-[r:RELATES]->(:Entity)
                    DELETE r
                    """,
                    {"entity_id": entity_id},
                )
                await self._run_query(
                    """
                    MATCH (e:Entity {id: $entity_id})<-[r:RELATES]-(:Entity)
                    DELETE r
                    """,
                    {"entity_id": entity_id},
                )
                await self._run_query(
                    """
                    MATCH (e:Entity {id: $entity_id})
                    DELETE e
                    """,
                    {"entity_id": entity_id},
                )
            return True
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 delete_entity 失败: {e}")
            raise MemoryException(f"delete_entity failed: {e}") from e

    async def clear_all(self) -> bool:
        try:
            async with self._write_lock:
                await self._run_query(
                    """
                    MATCH (:Entity)-[r:RELATES]->(:Entity)
                    DELETE r
                    """
                )
                await self._run_query("MATCH (e:Entity) DELETE e")
            return True
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 clear_all 失败: {e}")
            raise MemoryException(f"clear_all failed: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        try:
            node_rows = await self._run_query("MATCH (n) RETURN COUNT(n) AS count")
            rel_rows = await self._run_query(
                "MATCH ()-[r:RELATES]->() RETURN COUNT(r) AS count"
            )
            entity_rows = await self._run_query(
                "MATCH (n:Entity) RETURN COUNT(n) AS count"
            )

            total_nodes = int(node_rows[0].get("count", 0)) if node_rows else 0
            total_relationships = int(rel_rows[0].get("count", 0)) if rel_rows else 0
            entity_nodes = int(entity_rows[0].get("count", 0)) if entity_rows else 0

            return {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "entity_nodes": entity_nodes,
                "store_type": "kuzu",
                "db_path": self.db_path,
            }
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 get_stats 失败: {e}")
            raise MemoryException(f"get_stats failed: {e}") from e

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
        if self._db is not None:
            await asyncio.to_thread(self._db.close)
            self._db = None
        self._ready = False
        self.logger.info(f"🧠 Kuzu 连接已关闭: {self.db_path}")
