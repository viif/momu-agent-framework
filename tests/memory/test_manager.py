from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from momu_agent.memory import MemoryConfig, MemoryManager
from momu_agent.memory.base import MemoryItem
from momu_agent.memory.episodic import EpisodicMemory
from momu_agent.memory.semantic import SemanticMemory
from momu_agent.memory.working import WorkingMemory
from momu_agent.storage.document import DocumentStore
from momu_agent.storage.graph import GraphStore
from momu_agent.storage.vector import VectorStore


class DummyEmbedder:
    def encode(self, text: str):
        lowered = text.lower()
        return [
            float(lowered.count("python")),
            float(lowered.count("async")),
            float(lowered.count("memory")),
        ]


class FakeDocumentStore(DocumentStore):
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: dict | None = None,
    ) -> str:
        self.rows[memory_id] = {
            "memory_id": memory_id,
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "timestamp": timestamp,
            "importance": importance,
            "properties": dict(properties or {}),
        }
        return memory_id

    async def get_memory(self, memory_id: str) -> dict | None:
        return self.rows.get(memory_id)

    async def search_memories(
        self,
        user_id: str | None = None,
        memory_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        importance_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict]:
        items = list(self.rows.values())
        if user_id is not None:
            items = [item for item in items if item["user_id"] == user_id]
        if memory_type is not None:
            items = [item for item in items if item["memory_type"] == memory_type]
        if start_time is not None:
            items = [item for item in items if item["timestamp"] >= start_time]
        if end_time is not None:
            items = [item for item in items if item["timestamp"] <= end_time]
        if importance_threshold is not None:
            items = [
                item for item in items if item["importance"] >= importance_threshold
            ]
        items.sort(
            key=lambda item: (item["importance"], item["timestamp"]), reverse=True
        )
        return items[:limit]

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        properties: dict | None = None,
    ) -> bool:
        row = self.rows.get(memory_id)
        if row is None:
            return False
        if content is not None:
            row["content"] = content
        if importance is not None:
            row["importance"] = importance
        if properties is not None:
            row["properties"] = dict(properties)
        return True

    async def delete_memory(self, memory_id: str) -> bool:
        return self.rows.pop(memory_id, None) is not None

    async def get_database_stats(self) -> dict:
        return {
            "users_count": 1,
            "memories_count": len(self.rows),
            "memory_types": {"episodic": len(self.rows)},
            "top_users": {},
            "store_type": "fake-doc",
            "db_path": "",
        }

    async def add_document(self, content: str, metadata: dict | None = None) -> str:
        memory_id = f"doc-{len(self.rows) + 1}"
        return await self.add_memory(
            memory_id=memory_id,
            user_id=(metadata or {}).get("user_id", "system"),
            content=content,
            memory_type="document",
            timestamp=int(datetime.now().timestamp()),
            importance=0.5,
            properties=metadata,
        )

    async def get_document(self, document_id: str) -> dict | None:
        return await self.get_memory(document_id)

    async def close(self) -> None:
        return None


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.points: dict[str, dict] = {}

    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict],
        ids: list[str] | None = None,
    ) -> bool:
        point_ids = ids or [f"p-{index}" for index in range(len(vectors))]
        for point_id, vector, meta in zip(point_ids, vectors, metadata):
            self.points[point_id] = {
                "id": point_id,
                "vector": list(vector),
                "metadata": dict(meta),
            }
        return True

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        items = list(self.points.values())
        if where:
            for key, value in where.items():
                items = [item for item in items if item["metadata"].get(key) == value]

        results = []
        for item in items:
            distance = sum(abs(a - b) for a, b in zip(query_vector, item["vector"]))
            score = 1.0 / (1.0 + distance)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                {
                    "id": item["id"],
                    "score": score,
                    "metadata": dict(item["metadata"]),
                }
            )
        results.sort(key=lambda row: row["score"], reverse=True)
        return results[:limit]

    async def delete_vectors(self, ids: list[str]) -> bool:
        hit = False
        for point_id in ids:
            if self.points.pop(point_id, None) is not None:
                hit = True
        return hit

    async def delete_memories(self, memory_ids: list[str]) -> bool:
        hit = False
        to_delete = [
            point_id
            for point_id, payload in self.points.items()
            if payload["metadata"].get("memory_id") in memory_ids
        ]
        for point_id in to_delete:
            self.points.pop(point_id, None)
            hit = True
        return hit

    async def clear_collection(self) -> bool:
        self.points.clear()
        return True

    async def get_collection_info(self) -> dict:
        return {"vectors_count": len(self.points)}

    async def get_collection_stats(self) -> dict:
        return {"store_type": "fake-vector", "vectors_count": len(self.points)}

    async def close(self) -> None:
        return None


class FakeGraphStore(GraphStore):
    def __init__(self) -> None:
        self.entities: dict[str, dict] = {}
        self.relationships: dict[tuple[str, str, str], dict] = {}

    async def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> bool:
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "properties": dict(properties or {}),
        }
        return True

    async def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: dict | None = None,
    ) -> bool:
        self.relationships[(from_entity_id, to_entity_id, relationship_type)] = {
            "properties": dict(properties or {})
        }
        return True

    async def find_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[dict]:
        return []

    async def search_entities_by_name(
        self,
        name_pattern: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        pattern = name_pattern.replace("\\", "")
        allowed_types = set(entity_types or [])
        matches = [
            dict(entity)
            for entity in self.entities.values()
            if (not allowed_types or entity["type"] in allowed_types)
            and (not pattern or pattern in entity["name"])
        ]
        return matches[:limit]

    async def get_entity_relationships(self, entity_id: str) -> list[dict]:
        return []

    async def delete_entity(self, entity_id: str) -> bool:
        return self.entities.pop(entity_id, None) is not None

    async def clear_all(self) -> bool:
        self.entities.clear()
        self.relationships.clear()
        return True

    async def get_stats(self) -> dict:
        return {
            "total_nodes": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_nodes": len(self.entities),
            "store_type": "fake-graph",
            "db_path": "",
        }

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def mock_embedders(monkeypatch):
    monkeypatch.setattr(
        "momu_agent.memory.episodic.get_text_embedder", lambda: DummyEmbedder()
    )
    monkeypatch.setattr(
        "momu_agent.memory.semantic.get_text_embedder", lambda: DummyEmbedder()
    )


@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig(
        storage_path="./memory_data",
        working_memory_capacity=10,
        working_memory_tokens=100,
        working_memory_ttl_minutes=120,
        decay_factor=0.95,
    )


@pytest.fixture
def manager(config: MemoryConfig) -> MemoryManager:
    return MemoryManager(
        config=config,
        user_id="u1",
        working_memory=WorkingMemory(config),
        episodic_memory=EpisodicMemory(
            config,
            document_store=FakeDocumentStore(),
            vector_store=FakeVectorStore(),
        ),
        semantic_memory=SemanticMemory(
            config,
            graph_store=FakeGraphStore(),
            vector_store=FakeVectorStore(),
        ),
    )


def make_item(
    memory_id: str,
    content: str,
    memory_type: str,
    *,
    importance: float = 0.5,
    user_id: str = "u1",
    timestamp: datetime | None = None,
    metadata: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_add_memory_routes_to_target_type(manager: MemoryManager):
    working_id = await manager.add_memory(
        "working memory", memory_type="working", auto_classify=False
    )
    episodic_id = await manager.add_memory(
        "meeting recap",
        memory_type="episodic",
        metadata={"session_id": "s-1"},
        auto_classify=False,
    )
    semantic_id = await manager.add_memory(
        "python async 概念",
        memory_type="semantic",
        metadata={"concepts": ["python", "async"]},
        auto_classify=False,
    )

    assert await manager.memory_types["working"].has_memory(working_id) is True
    assert await manager.memory_types["episodic"].has_memory(episodic_id) is True
    assert await manager.memory_types["semantic"].has_memory(semantic_id) is True


@pytest.mark.asyncio
async def test_add_memory_auto_classify_and_importance_defaults(manager: MemoryManager):
    semantic_id = await manager.add_memory(
        "这是一个重要的概念定义",
        metadata={"priority": "high", "concepts": ["definition"]},
    )

    results = await manager.retrieve_memories(
        "概念",
        memory_types=["semantic"],
        limit=5,
    )

    assert results[0].id == semantic_id
    assert results[0].memory_type == "semantic"
    assert results[0].importance >= 0.8


@pytest.mark.asyncio
async def test_retrieve_memories_across_types_and_sorts_results(manager: MemoryManager):
    await manager.memory_types["working"].add(
        make_item(
            "w1",
            "python async local notes",
            "working",
            importance=0.4,
            metadata={"relevance_score": 0.4},
        )
    )
    await manager.memory_types["episodic"].add(
        make_item(
            "e1",
            "python async tutorial happened today",
            "episodic",
            importance=0.6,
            metadata={"session_id": "s-1"},
        )
    )
    await manager.memory_types["semantic"].add(
        make_item(
            "s1",
            "python async memory knowledge",
            "semantic",
            importance=0.7,
            metadata={"concepts": ["python", "async"]},
        )
    )

    results = await manager.retrieve_memories("python async", limit=3)
    scores = [
        float(memory.metadata.get("relevance_score", memory.importance))
        for memory in results
    ]

    assert {memory.id for memory in results} == {"w1", "e1", "s1"}
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_retrieve_memories_passes_session_and_threshold_filters(
    manager: MemoryManager,
):
    await manager.add_memory(
        "session one event",
        memory_type="episodic",
        importance=0.8,
        metadata={"session_id": "s-1"},
        auto_classify=False,
    )
    await manager.add_memory(
        "session two event",
        memory_type="episodic",
        importance=0.3,
        metadata={"session_id": "s-2"},
        auto_classify=False,
    )

    results = await manager.retrieve_memories(
        "event",
        memory_types=["episodic"],
        session_id="s-1",
        importance_threshold=0.5,
    )

    assert len(results) == 1
    assert results[0].metadata["session_id"] == "s-1"


@pytest.mark.asyncio
async def test_update_and_remove_memory_find_target_across_types(
    manager: MemoryManager,
):
    memory_id = await manager.add_memory(
        "old semantic knowledge",
        memory_type="semantic",
        metadata={"concepts": ["knowledge"]},
        auto_classify=False,
    )

    updated = await manager.update_memory(
        memory_id,
        content="updated semantic knowledge",
        importance=0.9,
        metadata={"concepts": ["updated"]},
    )
    removed = await manager.remove_memory(memory_id)

    assert updated is True
    assert removed is True
    assert await manager.memory_types["semantic"].has_memory(memory_id) is False


@pytest.mark.asyncio
async def test_forget_memories_and_stats_and_clear(manager: MemoryManager):
    stale_time = datetime.now() - timedelta(days=40)
    await manager.memory_types["working"].add(
        make_item("w-old", "old work", "working", importance=0.05, timestamp=stale_time)
    )
    await manager.memory_types["episodic"].add(
        make_item(
            "e-old",
            "old episode",
            "episodic",
            importance=0.05,
            timestamp=stale_time,
        )
    )
    await manager.memory_types["semantic"].add(
        make_item(
            "s-new",
            "important knowledge",
            "semantic",
            importance=0.9,
            metadata={"concepts": ["knowledge"]},
        )
    )

    forgotten = await manager.forget_memories(
        strategy="importance_based", threshold=0.1
    )
    stats = await manager.get_memory_stats()

    assert forgotten >= 2
    assert stats["enabled_types"] == ["working", "episodic", "semantic"]
    assert stats["total_memories"] >= 1
    assert stats["memories_by_type"]["semantic"]["count"] >= 1

    await manager.clear_all_memories()
    cleared_stats = await manager.get_memory_stats()

    assert cleared_stats["total_memories"] == 0


@pytest.mark.asyncio
async def test_consolidate_memories_moves_high_importance_working_memories(
    manager: MemoryManager,
):
    await manager.memory_types["working"].add(
        make_item("w1", "very important note", "working", importance=0.9)
    )
    await manager.memory_types["working"].add(
        make_item("w2", "minor note", "working", importance=0.2)
    )

    moved = await manager.consolidate_memories(
        from_type="working",
        to_type="episodic",
        importance_threshold=0.7,
    )

    episodic_results = await manager.retrieve_memories(
        "important",
        memory_types=["episodic"],
        limit=5,
    )
    working_memory = manager.memory_types["working"]
    assert isinstance(working_memory, WorkingMemory)
    working_results = await working_memory.get_all()

    assert moved == 1
    assert [memory.id for memory in episodic_results] == ["w1"]
    assert [memory.id for memory in working_results] == ["w2"]


@pytest.mark.asyncio
async def test_close_calls_all_memory_backends(manager: MemoryManager):
    close_mocks: list[AsyncMock] = []
    for backend in manager.memory_types.values():
        close_mock = AsyncMock(return_value=None)
        backend.close = close_mock
        close_mocks.append(close_mock)

    await manager.close()

    for close_mock in close_mocks:
        close_mock.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_close_continues_when_backend_fails(manager: MemoryManager):
    backends = list(manager.memory_types.values())
    close_mocks: list[AsyncMock] = []

    failing_close = AsyncMock(side_effect=RuntimeError("boom"))
    backends[0].close = failing_close
    close_mocks.append(failing_close)

    for backend in backends[1:]:
        close_mock = AsyncMock(return_value=None)
        backend.close = close_mock
        close_mocks.append(close_mock)

    await manager.close()

    for close_mock in close_mocks:
        close_mock.assert_awaited_once_with()
