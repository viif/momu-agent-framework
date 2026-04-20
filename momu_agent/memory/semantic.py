"""语义记忆实现。"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ..core.exceptions import MemoryException
from ..storage.graph import GraphStore, KuzuGraphStore
from ..storage.vector import ChromaVectorStore, VectorStore
from ..utils.embedding import get_text_embedder
from ..utils.logger import get_logger
from .base import Memory, MemoryConfig, MemoryItem


class SemanticMemory(Memory):
    """基于图存储与向量检索的语义记忆实现。"""

    _CONCEPT_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "我们",
        "你们",
        "他们",
        "以及",
        "但是",
        "如果",
        "因为",
        "所以",
        "这个",
        "那个",
        "一个",
        "一种",
        "已经",
        "可以",
        "需要",
        "进行",
        "相关",
        "以及",
        "然后",
        "就是",
        "还是",
        "其中",
    }

    def __init__(
        self,
        config: MemoryConfig,
        storage_backend: Any = None,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        super().__init__(config, storage_backend)
        self.graph_store = graph_store or KuzuGraphStore()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = get_text_embedder()

        self.logger = get_logger(__name__)
        self.logger.debug("🧠 SemanticMemory 初始化完成")

    async def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆。"""
        try:
            timestamp = int(memory_item.timestamp.timestamp())
            stored_metadata = dict(memory_item.metadata)
            concepts = self._extract_concepts(memory_item.content, stored_metadata)
            stored_metadata["concepts"] = concepts

            memory_entity_id = self._memory_entity_id(memory_item.id)
            user_entity_id = self._user_entity_id(memory_item.user_id)
            vector_point_id = self._vector_point_id(memory_item.id)

            await self.graph_store.add_entity(
                entity_id=user_entity_id,
                name=memory_item.user_id,
                entity_type="semantic_user",
                properties={"user_id": memory_item.user_id},
            )
            await self.graph_store.add_entity(
                entity_id=memory_entity_id,
                name=memory_item.id,
                entity_type="semantic_memory",
                properties={
                    "memory_id": memory_item.id,
                    "content": memory_item.content,
                    "memory_type": "semantic",
                    "user_id": memory_item.user_id,
                    "timestamp": timestamp,
                    "importance": memory_item.importance,
                    "metadata": stored_metadata,
                    "concepts": concepts,
                    "vector_point_id": vector_point_id,
                },
            )

            await self.graph_store.add_relationship(
                from_entity_id=user_entity_id,
                to_entity_id=memory_entity_id,
                relationship_type="OWNS",
                properties={
                    "memory_id": memory_item.id,
                    "created_at": timestamp,
                },
            )

            for concept in concepts:
                concept_entity_id = self._concept_entity_id(concept)
                await self.graph_store.add_entity(
                    entity_id=concept_entity_id,
                    name=concept,
                    entity_type="semantic_concept",
                    properties={"normalized": concept, "source": "rule"},
                )
                await self.graph_store.add_relationship(
                    from_entity_id=memory_entity_id,
                    to_entity_id=concept_entity_id,
                    relationship_type="MENTIONS",
                    properties={
                        "memory_id": memory_item.id,
                        "weight": 1.0,
                        "count": 1,
                        "source": "rule",
                    },
                )

            vector = self._encode_text(memory_item.content)
            if vector is None:
                return memory_item.id

            vector_metadata = {
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "memory_type": "semantic",
                "timestamp": timestamp,
                "importance": memory_item.importance,
                "content": memory_item.content,
                "concepts": concepts,
            }

            try:
                await self.vector_store.add_vectors(
                    vectors=[vector],
                    metadata=[vector_metadata],
                    ids=[vector_point_id],
                )
            except Exception as e:
                self.logger.warning(f"🧠 写入语义向量索引失败，已保留图记忆: {e}")

            return memory_item.id
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 添加语义记忆失败: {e}")
            raise MemoryException(f"添加语义记忆失败: {e}") from e

    async def retrieve(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> list[MemoryItem]:
        """检索语义记忆。"""
        try:
            user_id = kwargs.get("user_id")
            importance_threshold = kwargs.get("importance_threshold")
            score_threshold = kwargs.get("score_threshold")
            start_time, end_time = self._resolve_time_bounds(kwargs)

            query_concepts = self._extract_concepts(query)
            seen_ids: set[str] = set()
            ranked_items: list[tuple[float, MemoryItem]] = []

            query_vector = self._encode_text(query)
            vector_hits: list[dict[str, Any]] = []
            if query_vector is not None:
                where: dict[str, Any] = {"memory_type": "semantic"}
                if user_id:
                    where["user_id"] = user_id

                try:
                    vector_hits = await self.vector_store.search_similar(
                        query_vector=query_vector,
                        limit=max(limit * 4, 20),
                        score_threshold=score_threshold,
                        where=where,
                    )
                except Exception as e:
                    self.logger.warning(f"🧠 向量检索失败，回退图检索: {e}")

            for hit in vector_hits:
                metadata = hit.get("metadata") or {}
                memory_id = str(metadata.get("memory_id") or "")
                if not memory_id or memory_id in seen_ids:
                    continue

                entity = await self._get_memory_entity(memory_id)
                if entity is None:
                    continue

                properties = dict(entity.get("properties") or {})
                if not self._match_filters(
                    properties,
                    user_id=user_id,
                    start_time=start_time,
                    end_time=end_time,
                    importance_threshold=importance_threshold,
                ):
                    continue

                vector_score = float(hit.get("score") or 0.0)
                keyword_score = self._keyword_score(
                    query, str(properties.get("content") or "")
                )
                recency_score = self._recency_score(
                    int(properties.get("timestamp") or 0)
                )
                importance = float(properties.get("importance") or 0.5)
                graph_bonus = self._graph_bonus(
                    query_concepts, properties.get("concepts")
                )
                relevance_score = (
                    vector_score * 0.6
                    + keyword_score * 0.15
                    + recency_score * 0.1
                    + importance * 0.1
                    + graph_bonus * 0.05
                )

                ranked_items.append(
                    (
                        relevance_score,
                        self._entity_to_memory_item(
                            entity,
                            relevance_score=relevance_score,
                            vector_score=vector_score,
                            keyword_score=keyword_score,
                            recency_score=recency_score,
                            graph_bonus=graph_bonus,
                        ),
                    )
                )
                seen_ids.add(memory_id)

            if len(ranked_items) < limit:
                graph_candidates = await self._graph_fallback_candidates(
                    query_concepts,
                    limit=max(limit * 6, 30),
                )

                for entity in graph_candidates:
                    properties = dict(entity.get("properties") or {})
                    memory_id = str(
                        properties.get("memory_id") or entity.get("name") or ""
                    )
                    if not memory_id or memory_id in seen_ids:
                        continue
                    if not self._match_filters(
                        properties,
                        user_id=user_id,
                        start_time=start_time,
                        end_time=end_time,
                        importance_threshold=importance_threshold,
                    ):
                        continue

                    keyword_score = self._keyword_score(
                        query, str(properties.get("content") or "")
                    )
                    recency_score = self._recency_score(
                        int(properties.get("timestamp") or 0)
                    )
                    importance = float(properties.get("importance") or 0.5)
                    graph_bonus = self._graph_bonus(
                        query_concepts, properties.get("concepts")
                    )
                    relevance_score = (
                        keyword_score * 0.55
                        + recency_score * 0.15
                        + importance * 0.2
                        + graph_bonus * 0.1
                    )

                    ranked_items.append(
                        (
                            relevance_score,
                            self._entity_to_memory_item(
                                entity,
                                relevance_score=relevance_score,
                                vector_score=0.0,
                                keyword_score=keyword_score,
                                recency_score=recency_score,
                                graph_bonus=graph_bonus,
                            ),
                        )
                    )
                    seen_ids.add(memory_id)

            ranked_items.sort(key=lambda item: item[0], reverse=True)
            return [memory_item for _, memory_item in ranked_items[:limit]]
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 检索语义记忆失败: {e}")
            raise MemoryException(f"检索语义记忆失败: {e}") from e

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新语义记忆。"""
        try:
            existing = await self._get_memory_entity(memory_id)
            if existing is None:
                return False

            snapshot = self._entity_to_memory_item(existing)
            merged_metadata = dict(snapshot.metadata)
            if metadata is not None:
                merged_metadata.update(metadata)

            updated_item = MemoryItem(
                id=memory_id,
                content=content if content is not None else snapshot.content,
                memory_type="semantic",
                user_id=snapshot.user_id,
                timestamp=snapshot.timestamp,
                importance=(
                    importance if importance is not None else snapshot.importance
                ),
                metadata=merged_metadata,
            )

            graph_deleted = await self._delete_memory_graph(memory_id)
            if not graph_deleted:
                return False
            await self._delete_memory_vector(memory_id)

            try:
                await self.add(updated_item)
            except Exception as e:
                try:
                    await self.add(snapshot)
                except Exception as restore_error:
                    self.logger.error(f"🧠 语义记忆回滚失败: {restore_error}")
                raise MemoryException(f"更新语义记忆失败: {e}") from e

            self.logger.debug(f"🧠 更新语义记忆 [{memory_id}] 成功")
            return True
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 更新语义记忆失败: {e}")
            raise MemoryException(f"更新语义记忆失败: {e}") from e

    async def remove(self, memory_id: str) -> bool:
        """删除语义记忆。"""
        try:
            graph_deleted = await self._delete_memory_graph(memory_id)
            vector_deleted = await self._delete_memory_vector(memory_id)
            return bool(graph_deleted or vector_deleted)
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 删除语义记忆失败: {e}")
            raise MemoryException(f"删除语义记忆失败: {e}") from e

    async def has_memory(self, memory_id: str) -> bool:
        """检查语义记忆是否存在。"""
        return await self._get_memory_entity(memory_id) is not None

    async def clear(self) -> None:
        """清空所有语义记忆。"""
        memories = await self._list_entities("semantic_memory")
        removed_count = 0
        for entity in memories:
            properties = dict(entity.get("properties") or {})
            memory_id = str(properties.get("memory_id") or entity.get("name") or "")
            if not memory_id:
                continue
            if await self.remove(memory_id):
                removed_count += 1

        self.logger.info(f"🧠 清空语义记忆完成，共移除 {removed_count} 条")

    async def get_stats(self) -> dict[str, Any]:
        """获取语义记忆统计信息。"""
        memories = await self._list_entities("semantic_memory")
        concepts = await self._list_entities("semantic_concept")
        users = await self._list_entities("semantic_user")

        count = len(memories)
        avg_importance = 0.0
        timestamps: list[int] = []
        if memories:
            importances = []
            for entity in memories:
                properties = dict(entity.get("properties") or {})
                importances.append(float(properties.get("importance") or 0.5))
                timestamp = int(properties.get("timestamp") or 0)
                if timestamp > 0:
                    timestamps.append(timestamp)
            avg_importance = sum(importances) / len(importances)

        time_span_days = 0.0
        if timestamps:
            time_span_days = (max(timestamps) - min(timestamps)) / 86400.0

        try:
            vector_stats = await self.vector_store.get_collection_stats()
        except Exception:
            vector_stats = {"store_type": "unknown"}

        graph_stats = await self.graph_store.get_stats()
        return {
            "count": count,
            "forgotten_count": 0,
            "total_count": count,
            "concepts_count": len(concepts),
            "users_count": len(users),
            "avg_importance": avg_importance,
            "time_span_days": time_span_days,
            "memory_type": "semantic",
            "graph_store": graph_stats,
            "vector_store": vector_stats,
        }

    async def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """遗忘语义记忆。"""
        memories = await self._list_entities("semantic_memory")
        if not memories:
            return 0

        now = datetime.now()
        to_remove: list[str] = []

        if strategy == "importance_based":
            for entity in memories:
                properties = dict(entity.get("properties") or {})
                if float(properties.get("importance") or 0.0) < threshold:
                    to_remove.append(
                        str(properties.get("memory_id") or entity.get("name"))
                    )
        elif strategy == "time_based":
            cutoff = now - timedelta(days=max_age_days)
            for entity in memories:
                properties = dict(entity.get("properties") or {})
                timestamp = datetime.fromtimestamp(
                    int(properties.get("timestamp") or 0)
                )
                if timestamp < cutoff:
                    to_remove.append(
                        str(properties.get("memory_id") or entity.get("name"))
                    )
        elif strategy == "capacity_based" and len(memories) > self.config.max_capacity:
            ordered = sorted(
                memories,
                key=lambda entity: (
                    float((entity.get("properties") or {}).get("importance") or 0.0),
                    int((entity.get("properties") or {}).get("timestamp") or 0),
                ),
            )
            excess = len(memories) - self.config.max_capacity
            to_remove.extend(
                str(
                    (entity.get("properties") or {}).get("memory_id")
                    or entity.get("name")
                )
                for entity in ordered[:excess]
            )

        forgotten = 0
        for memory_id in dict.fromkeys(to_remove):
            if memory_id and await self.remove(memory_id):
                forgotten += 1

        return forgotten

    async def _get_memory_entity(self, memory_id: str) -> dict[str, Any] | None:
        escaped_memory_id = re.escape(memory_id)
        rows = await self.graph_store.search_entities_by_name(
            escaped_memory_id,
            entity_types=["semantic_memory"],
            limit=20,
        )
        expected_entity_id = self._memory_entity_id(memory_id)
        for row in rows:
            if row.get("id") == expected_entity_id:
                return row
        return None

    async def _graph_fallback_candidates(
        self, query_concepts: list[str], limit: int
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}

        for concept in query_concepts:
            concept_rows = await self.graph_store.search_entities_by_name(
                re.escape(concept),
                entity_types=["semantic_concept"],
                limit=10,
            )
            expected_concept_id = self._concept_entity_id(concept)
            for concept_row in concept_rows:
                if concept_row.get("id") != expected_concept_id:
                    continue
                related = await self.graph_store.find_related_entities(
                    concept_row["id"],
                    relationship_types=["MENTIONS"],
                    max_depth=1,
                    limit=limit,
                )
                for entity in related:
                    if entity.get("type") != "semantic_memory":
                        continue
                    properties = dict(entity.get("properties") or {})
                    memory_id = str(
                        properties.get("memory_id") or entity.get("name") or ""
                    )
                    if memory_id and memory_id not in candidates:
                        candidates[memory_id] = entity

        if len(candidates) < limit:
            for entity in await self._list_entities("semantic_memory"):
                properties = dict(entity.get("properties") or {})
                memory_id = str(properties.get("memory_id") or entity.get("name") or "")
                if memory_id and memory_id not in candidates:
                    candidates[memory_id] = entity
                    if len(candidates) >= limit:
                        break

        return list(candidates.values())

    async def _delete_memory_graph(self, memory_id: str) -> bool:
        entity = await self._get_memory_entity(memory_id)
        if entity is None:
            return False

        memory_entity_id = str(entity.get("id"))
        related_nodes = await self.graph_store.get_entity_relationships(
            memory_entity_id
        )
        deleted = await self.graph_store.delete_entity(memory_entity_id)
        if not deleted:
            return False

        for relationship in related_nodes:
            other_entity = relationship.get("other_entity") or {}
            other_entity_id = str(other_entity.get("id") or "")
            other_entity_type = str(other_entity.get("type") or "")
            if not other_entity_id or not other_entity_type.startswith("semantic_"):
                continue
            try:
                if not await self.graph_store.get_entity_relationships(other_entity_id):
                    await self.graph_store.delete_entity(other_entity_id)
            except Exception as e:
                self.logger.warning(f"🧠 清理孤立图节点失败: {e}")

        return True

    async def _delete_memory_vector(self, memory_id: str) -> bool:
        try:
            return await self.vector_store.delete_vectors(
                [self._vector_point_id(memory_id)]
            )
        except Exception:
            try:
                return await self.vector_store.delete_memories([memory_id])
            except Exception as e:
                self.logger.warning(f"🧠 删除语义向量索引失败: {e}")
                return False

    async def _list_entities(self, entity_type: str) -> list[dict[str, Any]]:
        return await self.graph_store.search_entities_by_name(
            "",
            entity_types=[entity_type],
            limit=10000,
        )

    def _memory_entity_id(self, memory_id: str) -> str:
        return f"semantic:memory:{memory_id}"

    def _user_entity_id(self, user_id: str) -> str:
        return f"semantic:user:{user_id}"

    def _concept_entity_id(self, concept: str) -> str:
        return f"semantic:concept:{concept}"

    def _vector_point_id(self, memory_id: str) -> str:
        return f"semantic:{memory_id}"

    def _extract_concepts(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[str]:
        if metadata is not None:
            raw_concepts = metadata.get("concepts")
            if isinstance(raw_concepts, str):
                raw_concepts = [raw_concepts]
            if isinstance(raw_concepts, (list, tuple, set)):
                normalized = []
                seen: set[str] = set()
                for concept in raw_concepts:
                    value = self._normalize_concept(str(concept))
                    if value and value not in seen:
                        seen.add(value)
                        normalized.append(value)
                if normalized:
                    return normalized[:8]

        tokens = [
            self._normalize_concept(token)
            for token in self._CONCEPT_PATTERN.findall(text or "")
        ]
        tokens = [
            token
            for token in tokens
            if token
            and token not in self._STOPWORDS
            and len(token) >= 2
            and not token.isdigit()
        ]
        if not tokens:
            return []

        counts = Counter(tokens)
        first_index = {token: index for index, token in enumerate(tokens)}
        ranked = sorted(
            counts,
            key=lambda token: (-counts[token], first_index[token], token),
        )
        return ranked[:8]

    def _normalize_concept(self, token: str) -> str:
        return token.strip().lower().replace(" ", "")

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
        properties: dict[str, Any],
        *,
        user_id: str | None,
        start_time: int | None,
        end_time: int | None,
        importance_threshold: float | None,
    ) -> bool:
        if user_id and properties.get("user_id") != user_id:
            return False

        timestamp = int(properties.get("timestamp") or 0)
        if start_time is not None and timestamp < start_time:
            return False
        if end_time is not None and timestamp > end_time:
            return False

        if importance_threshold is not None and float(
            properties.get("importance") or 0.0
        ) < float(importance_threshold):
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

    def _graph_bonus(self, query_concepts: list[str], memory_concepts: Any) -> float:
        if not query_concepts or not isinstance(memory_concepts, list):
            return 0.0

        query_set = set(query_concepts)
        memory_set = {str(concept) for concept in memory_concepts}
        if not query_set or not memory_set:
            return 0.0
        return len(query_set.intersection(memory_set)) / len(query_set)

    def _entity_to_memory_item(
        self,
        entity: dict[str, Any],
        *,
        relevance_score: float = 0.0,
        vector_score: float = 0.0,
        keyword_score: float = 0.0,
        recency_score: float = 0.0,
        graph_bonus: float = 0.0,
    ) -> MemoryItem:
        properties = dict(entity.get("properties") or {})
        metadata = dict(properties.get("metadata") or {})
        concepts = properties.get("concepts")
        if isinstance(concepts, list):
            metadata.setdefault("concepts", concepts)
        metadata.update(
            {
                "relevance_score": relevance_score,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "recency_score": recency_score,
                "graph_bonus": graph_bonus,
            }
        )

        timestamp = int(properties.get("timestamp") or 0)
        return MemoryItem(
            id=str(properties.get("memory_id") or entity.get("name") or ""),
            content=str(properties.get("content") or ""),
            memory_type="semantic",
            user_id=str(properties.get("user_id") or ""),
            timestamp=(
                datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
            ),
            importance=float(properties.get("importance") or 0.5),
            metadata=metadata,
        )

    def _to_epoch(self, value: datetime | int | float | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return int(value.timestamp())
        return int(value)
