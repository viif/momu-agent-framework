"""向量存储实现

支持的后端：
- Chroma: 本地向量数据库
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, cast
from uuid import uuid4

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embeddings, Metadata, Where

from ..core.exceptions import StorageException
from ..utils.logger import get_logger


class VectorStore(ABC):
    """向量存储抽象基类（qdrant 风格接口）"""

    @abstractmethod
    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:
        """批量写入向量。"""

    @abstractmethod
    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按向量相似度检索。"""

    @abstractmethod
    async def delete_vectors(self, ids: list[str]) -> bool:
        """按向量点 ID 批量删除。"""

    @abstractmethod
    async def delete_memories(self, memory_ids: list[str]) -> bool:
        """按业务 memory_id 批量删除。"""

    @abstractmethod
    async def clear_collection(self) -> bool:
        """清空集合。"""

    @abstractmethod
    async def get_collection_info(self) -> dict[str, Any]:
        """获取集合详情。"""

    @abstractmethod
    async def get_collection_stats(self) -> dict[str, Any]:
        """获取集合统计。"""


class ChromaVectorStore(VectorStore):
    """Chroma 向量存储实现"""

    _DEFAULT_PATH = os.path.join(os.getcwd(), ".storage", "chroma_data")
    _DEFAULT_COLLECTION = "memories"
    _instances: dict[tuple[str, str], "ChromaVectorStore"] = {}

    def __new__(
        cls,
        collection_path: str | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_function: Any = None,
    ) -> "ChromaVectorStore":
        _ = embedding_function
        abs_path = os.path.abspath(collection_path or cls._DEFAULT_PATH)
        key = (abs_path, collection_name)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        collection_path: str | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_function: Any = None,
    ) -> None:
        if hasattr(self, "_initialized"):
            return
        self.collection_path = os.path.abspath(collection_path or self._DEFAULT_PATH)
        self.collection_name = collection_name
        self._embedding_function = embedding_function
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None
        self._ready = False
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        os.makedirs(self.collection_path, exist_ok=True)
        self._initialized = True

        self.logger = get_logger(__name__)
        self.logger.debug(
            f"🧠 ChromaVectorStore 初始化完成 "
            f"(path: {self.collection_path}, collection: {self.collection_name})"
        )

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        async with self._init_lock:
            if self._ready:
                return
            client = await asyncio.to_thread(
                chromadb.PersistentClient, path=self.collection_path
            )
            collection = await asyncio.to_thread(
                client.get_or_create_collection,
                name=self.collection_name,
                embedding_function=self._embedding_function,
            )
            self._client = client
            self._collection = collection
            self._ready = True
            self.logger.info(
                f"🧠 Chroma 集合初始化完成: {self.collection_path}/{self.collection_name}"
            )

    def _build_where(self, where: dict[str, Any] | None) -> Where | None:
        if not where:
            return None
        conditions: list[dict[str, Any]] = []
        for key, value in where.items():
            if isinstance(value, dict):
                conditions.append({key: value})
            else:
                conditions.append({key: {"$eq": value}})
        if len(conditions) == 1:
            return cast(Where, conditions[0])
        return cast(Where, {"$and": conditions})

    def _row_to_dict(
        self, point_id: str, distance: float | None, metadata: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        score = 0.0 if distance is None else 1.0 / (1.0 + float(distance))
        return {
            "id": point_id,
            "score": score,
            "metadata": dict(metadata) if metadata is not None else {},
        }

    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:
        """批量写入向量。"""
        try:
            if not vectors:
                self.logger.warning("🧠 向量列表为空")
                return False
            if len(vectors) != len(metadata):
                self.logger.warning("🧠 vectors 与 metadata 长度不一致")
                return False
            if ids is not None and len(ids) != len(vectors):
                self.logger.warning("🧠 ids 与 vectors 长度不一致")
                return False

            expected_dim = len(vectors[0])
            if expected_dim == 0:
                self.logger.warning("🧠 向量维度不能为 0")
                return False
            if any(len(v) != expected_dim for v in vectors):
                self.logger.warning("🧠 向量维度不一致")
                return False

            await self._ensure_collection()
            assert self._collection is not None

            point_ids = ids or [str(uuid4()) for _ in vectors]
            now = int(time.time())
            docs: list[str] = []
            metas: list[Metadata] = []

            for point_id, meta in zip(point_ids, metadata):
                meta_copy = dict(meta)
                meta_copy.setdefault(
                    "memory_id", str(meta_copy.get("memory_id", point_id))
                )
                meta_copy.setdefault("timestamp", now)
                meta_copy.setdefault("added_at", now)
                docs.append(str(meta_copy.get("content", "")))
                metas.append(cast(Metadata, meta_copy))

            embeddings_arr: Embeddings = [
                np.array(v, dtype=np.float32) for v in vectors
            ]
            async with self._write_lock:
                await asyncio.to_thread(
                    self._collection.upsert,
                    ids=[str(i) for i in point_ids],
                    embeddings=embeddings_arr,
                    documents=docs,
                    metadatas=metas,
                )
            self.logger.debug(f"🧠 批量写入向量成功: {len(vectors)}")
            return True
        except Exception as e:
            self.logger.error(f"🧠 add_vectors 失败: {e}")
            raise StorageException(f"add_vectors failed: {e}") from e

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按向量相似度检索。"""
        try:
            if limit <= 0:
                return []
            await self._ensure_collection()
            assert self._collection is not None
            count = await asyncio.to_thread(self._collection.count)
            if count == 0:
                return []

            kwargs: dict[str, Any] = {
                "query_embeddings": [np.array(query_vector, dtype=np.float32)],
                "n_results": min(limit, count),
                "include": ["metadatas", "distances"],
            }
            chroma_where = self._build_where(where)
            if chroma_where is not None:
                kwargs["where"] = chroma_where

            result = await asyncio.to_thread(self._collection.query, **kwargs)
            ids_list = result.get("ids") or [[]]
            distances_list = result.get("distances") or [[]]
            metas_list = result.get("metadatas") or [[]]
            ids0 = ids_list[0] if len(ids_list) > 0 else []
            distances0 = distances_list[0] if len(distances_list) > 0 else []
            metas0 = metas_list[0] if len(metas_list) > 0 else []
            items = [
                self._row_to_dict(point_id, distance, meta)
                for point_id, distance, meta in zip(ids0, distances0, metas0)
            ]
            if score_threshold is not None:
                items = [i for i in items if i["score"] >= score_threshold]

            self.logger.debug(
                f"🧠 向量搜索命中 {len(items)} 条 (where: {where!r}, limit: {limit})"
            )
            return items
        except Exception as e:
            self.logger.error(f"🧠 search_similar 失败: {e}")
            raise StorageException(f"search_similar failed: {e}") from e

    async def delete_vectors(self, ids: list[str]) -> bool:
        """按向量点 ID 批量删除。"""
        try:
            if not ids:
                return True
            await self._ensure_collection()
            assert self._collection is not None
            async with self._write_lock:
                await asyncio.to_thread(
                    self._collection.delete, ids=[str(i) for i in ids]
                )
            self.logger.debug(f"🧠 按点 ID 删除向量成功: {len(ids)}")
            return True
        except Exception as e:
            self.logger.error(f"🧠 delete_vectors 失败: {e}")
            raise StorageException(f"delete_vectors failed: {e}") from e

    async def delete_memories(self, memory_ids: list[str]) -> bool:
        """按业务 memory_id 批量删除。"""
        try:
            if not memory_ids:
                return True
            await self._ensure_collection()
            assert self._collection is not None

            existing = await asyncio.to_thread(
                self._collection.get,
                where=cast(Where, {"memory_id": {"$in": memory_ids}}),
                include=[],
            )
            target_ids = existing.get("ids", [])
            if not target_ids:
                return False

            async with self._write_lock:
                await asyncio.to_thread(self._collection.delete, ids=target_ids)
            self.logger.debug(f"🧠 按 memory_id 删除向量成功: {len(target_ids)}")
            return True
        except Exception as e:
            self.logger.error(f"🧠 delete_memories 失败: {e}")
            raise StorageException(f"delete_memories failed: {e}") from e

    async def clear_collection(self) -> bool:
        """清空集合。"""
        try:
            await self._ensure_collection()
            assert self._client is not None
            async with self._write_lock:
                await asyncio.to_thread(
                    self._client.delete_collection,
                    self.collection_name,
                )
                self._collection = await asyncio.to_thread(
                    self._client.get_or_create_collection,
                    name=self.collection_name,
                    embedding_function=self._embedding_function,
                )
            self.logger.info(
                f"🧠 Chroma 集合已清空: {self.collection_path}/{self.collection_name}"
            )
            return True
        except Exception as e:
            self.logger.error(f"🧠 clear_collection 失败: {e}")
            raise StorageException(f"clear_collection failed: {e}") from e

    async def get_collection_info(self) -> dict[str, Any]:
        """获取集合详情。"""
        try:
            await self._ensure_collection()
            assert self._collection is not None
            count = await asyncio.to_thread(self._collection.count)

            vector_size: int | None = None
            if count > 0:
                sample = await asyncio.to_thread(
                    self._collection.get,
                    limit=1,
                    include=["embeddings"],
                )
                embeddings = sample.get("embeddings")
                if (
                    embeddings is not None
                    and len(embeddings) > 0
                    and embeddings[0] is not None
                ):
                    vector_size = len(embeddings[0])

            return {
                "name": self.collection_name,
                "vectors_count": count,
                "indexed_vectors_count": count,
                "points_count": count,
                "segments_count": 1 if count > 0 else 0,
                "config": {
                    "vector_size": vector_size,
                    "distance": "unknown",
                },
            }
        except Exception as e:
            self.logger.error(f"🧠 get_collection_info 失败: {e}")
            raise StorageException(f"get_collection_info failed: {e}") from e

    async def get_collection_stats(self) -> dict[str, Any]:
        """获取集合统计。"""
        try:
            info = await self.get_collection_info()
            info["store_type"] = "chroma"
            info["collection_path"] = self.collection_path
            return info
        except Exception as e:
            self.logger.error(f"🧠 get_collection_stats 失败: {e}")
            raise StorageException(f"get_collection_stats failed: {e}") from e

    async def close(self) -> None:
        """关闭连接并重置初始化状态。"""
        if self._collection is not None or self._client is not None:
            self._collection = None
            self._client = None
            self._ready = False
            self.logger.info(
                f"🧠 Chroma 连接已关闭: {self.collection_path}/{self.collection_name}"
            )
