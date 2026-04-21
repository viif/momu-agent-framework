"""情景记忆实现

- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- 模式识别能力
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..core.exceptions import MemoryException
from ..storage.document import DocumentStore, SQLiteDocumentStore
from ..storage.vector import ChromaVectorStore, VectorStore
from ..utils.embedding import get_text_embedder
from ..utils.logger import get_logger
from .base import Memory, MemoryConfig, MemoryItem


class EpisodicMemory(Memory):
    """情景记忆实现。"""

    def __init__(
        self,
        config: MemoryConfig,
        storage_backend: Any = None,
        document_store: DocumentStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        super().__init__(config, storage_backend)
        self.document_store = document_store or SQLiteDocumentStore()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = get_text_embedder()

        self.logger = get_logger(__name__)
        self.logger.debug("🧠 EpisodicMemory 初始化完成")

    async def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆。"""
        try:
            properties = dict(memory_item.metadata)
            timestamp = int(memory_item.timestamp.timestamp())

            # 先写入文档存储，保证向量写入失败时记忆仍可用。
            await self.document_store.add_memory(
                memory_id=memory_item.id,
                user_id=memory_item.user_id,
                content=memory_item.content,
                memory_type="episodic",
                timestamp=timestamp,
                importance=memory_item.importance,
                properties=properties,
            )

            vector = self._encode_text(memory_item.content)
            if vector is None:
                return memory_item.id

            vector_metadata = {
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "memory_type": "episodic",
                "timestamp": timestamp,
                "importance": memory_item.importance,
                "content": memory_item.content,
            }
            if "session_id" in properties:
                vector_metadata["session_id"] = properties["session_id"]

            try:
                await self.vector_store.add_vectors(
                    vectors=[vector],
                    metadata=[vector_metadata],
                    ids=[self._vector_point_id(memory_item.id)],
                )
            except Exception as e:
                self.logger.warning(f"🧠 写入向量索引失败，已保留文档记忆: {e}")

            return memory_item.id
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 添加情景记忆失败: {e}")
            raise MemoryException(f"添加情景记忆失败: {e}") from e

    async def retrieve(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> list[MemoryItem]:
        """检索情景记忆。"""
        try:
            user_id = kwargs.get("user_id")
            session_id = kwargs.get("session_id")
            importance_threshold = kwargs.get("importance_threshold")
            score_threshold = kwargs.get("score_threshold")
            start_time, end_time = self._resolve_time_bounds(kwargs)

            seen_ids: set[str] = set()
            ranked_items: list[tuple[float, MemoryItem]] = []

            query_vector = self._encode_text(query)
            vector_hits: list[dict[str, Any]] = []
            if query_vector is not None:
                where: dict[str, Any] = {"memory_type": "episodic"}
                if user_id:
                    where["user_id"] = user_id
                if session_id:
                    where["session_id"] = session_id

                try:
                    # 优先向量召回，扩大候选集后再进行本地重排。
                    vector_hits = await self.vector_store.search_similar(
                        query_vector=query_vector,
                        limit=max(limit * 4, 20),
                        score_threshold=score_threshold,
                        where=where,
                    )
                except Exception as e:
                    self.logger.warning(f"🧠 向量检索失败，回退文档检索: {e}")

            for hit in vector_hits:
                metadata = hit.get("metadata") or {}
                memory_id = str(metadata.get("memory_id") or "")
                if not memory_id or memory_id in seen_ids:
                    continue

                doc = await self.document_store.get_memory(memory_id)
                if doc is None or doc.get("memory_type") != "episodic":
                    continue
                if not self._match_filters(
                    doc,
                    user_id=user_id,
                    session_id=session_id,
                    start_time=start_time,
                    end_time=end_time,
                    importance_threshold=importance_threshold,
                ):
                    continue

                vector_score = float(hit.get("score") or 0.0)
                keyword_score = self._keyword_score(
                    query, str(doc.get("content") or "")
                )
                recency_score = self._recency_score(int(doc.get("timestamp") or 0))
                importance = float(doc.get("importance") or 0.5)
                # 混合向量、关键词、时效性与重要性进行重排打分。
                relevance_score = (
                    vector_score * 0.6
                    + keyword_score * 0.2
                    + recency_score * 0.1
                    + importance * 0.1
                )

                ranked_items.append(
                    (
                        relevance_score,
                        self._to_memory_item(
                            doc,
                            relevance_score=relevance_score,
                            vector_score=vector_score,
                            keyword_score=keyword_score,
                            recency_score=recency_score,
                        ),
                    )
                )
                seen_ids.add(memory_id)

            if len(ranked_items) < limit:
                # 向量结果不足时，使用文档检索补全候选。
                docs = await self.document_store.search_memories(
                    user_id=user_id,
                    memory_type="episodic",
                    start_time=start_time,
                    end_time=end_time,
                    importance_threshold=importance_threshold,
                    limit=max(limit * 8, 50),
                )

                for doc in docs:
                    memory_id = str(doc.get("memory_id") or "")
                    if not memory_id or memory_id in seen_ids:
                        continue
                    if not self._match_filters(
                        doc,
                        user_id=user_id,
                        session_id=session_id,
                        start_time=start_time,
                        end_time=end_time,
                        importance_threshold=importance_threshold,
                    ):
                        continue

                    keyword_score = self._keyword_score(
                        query, str(doc.get("content") or "")
                    )
                    recency_score = self._recency_score(int(doc.get("timestamp") or 0))
                    importance = float(doc.get("importance") or 0.5)
                    relevance_score = (
                        keyword_score * 0.7 + recency_score * 0.2 + importance * 0.1
                    )

                    ranked_items.append(
                        (
                            relevance_score,
                            self._to_memory_item(
                                doc,
                                relevance_score=relevance_score,
                                vector_score=0.0,
                                keyword_score=keyword_score,
                                recency_score=recency_score,
                            ),
                        )
                    )
                    seen_ids.add(memory_id)

            ranked_items.sort(key=lambda item: item[0], reverse=True)
            return [memory_item for _, memory_item in ranked_items[:limit]]
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 检索情景记忆失败: {e}")
            raise MemoryException(f"检索情景记忆失败: {e}") from e

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新情景记忆。"""
        try:
            existing = await self.document_store.get_memory(memory_id)
            if existing is None:
                return False

            merged_properties = dict(existing.get("properties") or {})
            if metadata is not None:
                merged_properties.update(metadata)

            updated = await self.document_store.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                properties=merged_properties if metadata is not None else None,
            )
            if not updated:
                return False

            new_content = (
                content if content is not None else str(existing.get("content") or "")
            )
            new_importance = float(
                importance
                if importance is not None
                else (existing.get("importance") or 0.5)
            )
            new_timestamp = int(
                existing.get("timestamp") or int(datetime.now().timestamp())
            )
            new_user_id = str(existing.get("user_id") or "")
            new_properties = (
                merged_properties
                if metadata is not None
                else dict(existing.get("properties") or {})
            )

            vector = self._encode_text(new_content)
            if vector is not None:
                # 先删除旧向量点，再写入新向量，避免同 ID 下的脏索引。
                try:
                    await self.vector_store.delete_vectors(
                        [self._vector_point_id(memory_id)]
                    )
                except Exception:
                    pass

                vector_metadata = {
                    "memory_id": memory_id,
                    "user_id": new_user_id,
                    "memory_type": "episodic",
                    "timestamp": new_timestamp,
                    "importance": new_importance,
                    "content": new_content,
                }
                if "session_id" in new_properties:
                    vector_metadata["session_id"] = new_properties["session_id"]

                try:
                    await self.vector_store.add_vectors(
                        vectors=[vector],
                        metadata=[vector_metadata],
                        ids=[self._vector_point_id(memory_id)],
                    )
                except Exception as e:
                    self.logger.warning(f"🧠 更新向量索引失败，文档已更新: {e}")

            self.logger.debug(f"🧠 更新情景记忆 [{memory_id}] 成功")
            return True
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 更新情景记忆失败: {e}")
            raise MemoryException(f"更新情景记忆失败: {e}") from e

    async def remove(self, memory_id: str) -> bool:
        """删除情景记忆。"""
        try:
            doc_deleted = await self.document_store.delete_memory(memory_id)
            vector_deleted = False

            try:
                vector_deleted = await self.vector_store.delete_vectors(
                    [self._vector_point_id(memory_id)]
                )
            except Exception:
                try:
                    # 兼容不支持点 ID 删除的向量后端。
                    vector_deleted = await self.vector_store.delete_memories(
                        [memory_id]
                    )
                except Exception as e:
                    self.logger.warning(f"🧠 删除向量索引失败: {e}")

            return bool(doc_deleted or vector_deleted)
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 删除情景记忆失败: {e}")
            raise MemoryException(f"删除情景记忆失败: {e}") from e

    async def has_memory(self, memory_id: str) -> bool:
        """检查情景记忆是否存在。"""
        memory = await self.document_store.get_memory(memory_id)
        return memory is not None and memory.get("memory_type") == "episodic"

    async def clear(self) -> None:
        """清空所有情景记忆（仅 episodic）。"""
        removed_count = 0

        while True:
            docs = await self.document_store.search_memories(
                memory_type="episodic", limit=500
            )
            if not docs:
                break

            deleted_any = False
            ids: list[str] = []
            for doc in docs:
                memory_id = str(doc.get("memory_id") or "")
                if not memory_id:
                    continue
                if await self.document_store.delete_memory(memory_id):
                    deleted_any = True
                    removed_count += 1
                    ids.append(memory_id)

            if ids:
                point_ids = [self._vector_point_id(memory_id) for memory_id in ids]
                try:
                    await self.vector_store.delete_vectors(point_ids)
                except Exception:
                    try:
                        # 批量清理时同样保留按 memory_id 的兜底路径。
                        await self.vector_store.delete_memories(ids)
                    except Exception as e:
                        self.logger.warning(f"🧠 清理向量索引失败: {e}")

            if not deleted_any:
                break

        self.logger.info(f"🧠 清空情景记忆完成，共移除 {removed_count} 条")

    async def get_stats(self) -> dict[str, Any]:
        """获取情景记忆统计信息。"""
        docs = await self.document_store.search_memories(
            memory_type="episodic", limit=10000
        )
        count = len(docs)

        avg_importance = (
            sum(float(doc.get("importance") or 0.5) for doc in docs) / count
            if count
            else 0.0
        )
        timestamps = [
            int(doc.get("timestamp") or 0)
            for doc in docs
            if int(doc.get("timestamp") or 0) > 0
        ]
        time_span_days = 0.0
        if timestamps:
            time_span_days = (max(timestamps) - min(timestamps)) / 86400.0

        session_ids = {
            str((doc.get("properties") or {}).get("session_id"))
            for doc in docs
            if (doc.get("properties") or {}).get("session_id") is not None
        }

        db_stats = await self.document_store.get_database_stats()
        try:
            vector_stats = await self.vector_store.get_collection_stats()
        except Exception:
            vector_stats = {"store_type": "unknown"}

        return {
            "count": count,
            "forgotten_count": 0,
            "total_count": count,
            "sessions_count": len(session_ids),
            "avg_importance": avg_importance,
            "time_span_days": time_span_days,
            "memory_type": "episodic",
            "vector_store": vector_stats,
            "document_store": {
                "store_type": db_stats.get("store_type"),
                "db_path": db_stats.get("db_path"),
                "memories_count": db_stats.get("memories_count"),
            },
        }

    async def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """情景记忆遗忘机制。"""
        docs = await self.document_store.search_memories(
            memory_type="episodic", limit=10000
        )
        if not docs:
            return 0

        now = datetime.now()
        to_remove: list[str] = []

        if strategy == "importance_based":
            for doc in docs:
                if float(doc.get("importance") or 0.0) < threshold:
                    to_remove.append(str(doc.get("memory_id")))
        elif strategy == "time_based":
            cutoff = now - timedelta(days=max_age_days)
            for doc in docs:
                timestamp = datetime.fromtimestamp(int(doc.get("timestamp") or 0))
                if timestamp < cutoff:
                    to_remove.append(str(doc.get("memory_id")))
        elif strategy == "capacity_based" and len(docs) > self.config.max_capacity:
            ordered = sorted(
                docs,
                key=lambda item: (
                    float(item.get("importance") or 0.0),
                    int(item.get("timestamp") or 0),
                ),
            )
            excess = len(docs) - self.config.max_capacity
            to_remove.extend(str(doc.get("memory_id")) for doc in ordered[:excess])

        forgotten = 0
        for memory_id in dict.fromkeys(to_remove):
            if await self.remove(memory_id):
                forgotten += 1

        return forgotten

    def _vector_point_id(self, memory_id: str) -> str:
        return f"episodic:{memory_id}"

    def _encode_text(self, text: str) -> list[float] | None:
        try:
            vector = self.embedder.encode(text)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                vector = vector[0]
            if not isinstance(vector, list):
                return None
            return [float(v) for v in vector]
        except Exception as e:
            self.logger.warning(f"🧠 计算文本向量失败: {e}")
            return None

    def _resolve_time_bounds(
        self, kwargs: dict[str, Any]
    ) -> tuple[int | None, int | None]:
        time_range = kwargs.get("time_range")
        if time_range is not None:
            return self._to_epoch(time_range[0]), self._to_epoch(time_range[1])

        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        return self._to_epoch(start_time), self._to_epoch(end_time)

    def _match_filters(
        self,
        doc: dict[str, Any],
        *,
        user_id: str | None,
        session_id: str | None,
        start_time: int | None,
        end_time: int | None,
        importance_threshold: float | None,
    ) -> bool:
        if user_id and doc.get("user_id") != user_id:
            return False

        doc_timestamp = int(doc.get("timestamp") or 0)
        if start_time is not None and doc_timestamp < start_time:
            return False
        if end_time is not None and doc_timestamp > end_time:
            return False

        if importance_threshold is not None and float(
            doc.get("importance") or 0.0
        ) < float(importance_threshold):
            return False

        if session_id:
            session_value = (doc.get("properties") or {}).get("session_id")
            if session_value != session_id:
                return False

        return True

    def _keyword_score(self, query: str, content: str) -> float:
        if not query.strip() or not content.strip():
            return 0.0

        query_lower = query.lower()
        content_lower = content.lower()
        if query_lower in content_lower:
            return 1.0

        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        union = query_words.union(content_words)
        if not union:
            return 0.0
        return len(query_words.intersection(content_words)) / len(union)

    def _recency_score(self, timestamp: int) -> float:
        if timestamp <= 0:
            return 0.1
        age_hours = max(0.0, (datetime.now().timestamp() - timestamp) / 3600)
        decay_factor = self.config.decay_factor ** (age_hours / 6)
        return max(0.1, float(decay_factor))

    def _to_memory_item(
        self,
        doc: dict[str, Any],
        *,
        relevance_score: float,
        vector_score: float,
        keyword_score: float,
        recency_score: float,
    ) -> MemoryItem:
        metadata = dict(doc.get("properties") or {})
        metadata.update(
            {
                "relevance_score": relevance_score,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "recency_score": recency_score,
            }
        )

        return MemoryItem(
            id=str(doc.get("memory_id")),
            content=str(doc.get("content") or ""),
            memory_type="episodic",
            user_id=str(doc.get("user_id") or ""),
            timestamp=datetime.fromtimestamp(int(doc.get("timestamp") or 0)),
            importance=float(doc.get("importance") or 0.5),
            metadata=metadata,
        )

    def _to_epoch(self, value: datetime | int | float | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return int(value.timestamp())
        return int(value)
