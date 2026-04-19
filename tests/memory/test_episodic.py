from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from momu_agent.memory.base import MemoryConfig, MemoryItem
from momu_agent.memory.episodic import EpisodicMemory
from momu_agent.storage.document import DocumentStore, SQLiteDocumentStore
from momu_agent.storage.vector import ChromaVectorStore, VectorStore


class DummyEmbedder:
    def encode(self, text: str):
        lowered = text.lower()
        return [
            float(lowered.count("python")),
            float(lowered.count("async")),
            float(lowered.count("tutorial")),
        ]


class FakeDocumentStore(DocumentStore):
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

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
        self.rows[memory_id] = {
            "memory_id": memory_id,
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "timestamp": timestamp,
            "importance": importance,
            "properties": dict(properties or {}),
            "created_at": "",
        }
        return memory_id

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self.rows.get(memory_id)

    async def search_memories(
        self,
        user_id: str | None = None,
        memory_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        importance_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
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
        items.sort(key=lambda x: (x["importance"], x["timestamp"]), reverse=True)
        return items[:limit]

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        properties: dict[str, Any] | None = None,
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

    async def get_database_stats(self) -> dict[str, Any]:
        memory_types: dict[str, int] = {}
        top_users: dict[str, int] = {}
        for row in self.rows.values():
            memory_types[row["memory_type"]] = (
                memory_types.get(row["memory_type"], 0) + 1
            )
            top_users[row["user_id"]] = top_users.get(row["user_id"], 0) + 1
        return {
            "users_count": len(top_users),
            "memories_count": len(self.rows),
            "memory_types": memory_types,
            "top_users": top_users,
            "store_type": "fake",
            "db_path": "",
        }

    async def add_document(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> str:
        memory_id = f"doc-{len(self.rows) + 1}"
        await self.add_memory(
            memory_id=memory_id,
            user_id=(metadata or {}).get("user_id", "system"),
            content=content,
            memory_type="document",
            timestamp=int(datetime.now().timestamp()),
            importance=0.5,
            properties=metadata or {},
        )
        return memory_id

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        return await self.get_memory(document_id)


class FakeVectorStore(VectorStore):
    def __init__(self, fail_search: bool = False) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.fail_search = fail_search

    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
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
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.fail_search:
            raise RuntimeError("vector search failed")

        items = list(self.points.values())
        if where:
            for key, value in where.items():
                items = [item for item in items if item["metadata"].get(key) == value]

        results: list[dict[str, Any]] = []
        for item in items:
            vector = item["vector"]
            distance = sum(abs(a - b) for a, b in zip(query_vector, vector))
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

        results.sort(key=lambda x: x["score"], reverse=True)
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

    async def get_collection_info(self) -> dict[str, Any]:
        return {"vectors_count": len(self.points)}

    async def get_collection_stats(self) -> dict[str, Any]:
        return {
            "store_type": "fake-vector",
            "vectors_count": len(self.points),
        }


def make_item(
    memory_id: str,
    content: str,
    *,
    user_id: str = "u1",
    importance: float = 0.5,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        memory_type="episodic",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
        metadata=metadata or {},
    )


@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig(storage_path="./memory_data", decay_factor=0.95)


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    monkeypatch.setattr(
        "momu_agent.memory.episodic.get_text_embedder", lambda: DummyEmbedder()
    )


@pytest.mark.asyncio
async def test_constructor_uses_default_stores(config):
    memory = EpisodicMemory(config=config)

    assert isinstance(memory.document_store, SQLiteDocumentStore)
    assert isinstance(memory.vector_store, ChromaVectorStore)


@pytest.mark.asyncio
async def test_constructor_uses_injected_stores(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    assert memory.document_store is doc_store
    assert memory.vector_store is vector_store


@pytest.mark.asyncio
async def test_add_persists_to_document_and_vector(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    item = make_item("m1", "python async memory", metadata={"session_id": "s1"})
    memory_id = await memory.add(item)

    assert memory_id == "m1"
    assert "m1" in doc_store.rows
    assert "episodic:m1" in vector_store.points
    assert vector_store.points["episodic:m1"]["metadata"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_retrieve_with_user_filter_and_rerank(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await memory.add(
        make_item(
            "m1",
            "python async tutorial",
            user_id="u1",
            importance=0.7,
            timestamp=datetime.now() - timedelta(hours=2),
            metadata={"session_id": "s1"},
        )
    )
    await memory.add(
        make_item(
            "m2",
            "python async tutorial advanced",
            user_id="u1",
            importance=0.9,
            timestamp=datetime.now() - timedelta(minutes=10),
            metadata={"session_id": "s1"},
        )
    )
    await memory.add(
        make_item(
            "m3",
            "python async tutorial",
            user_id="u2",
            importance=1.0,
            metadata={"session_id": "s1"},
        )
    )

    results = await memory.retrieve(
        "python async tutorial",
        user_id="u1",
        session_id="s1",
        limit=2,
    )

    assert [item.id for item in results] == ["m2", "m1"]
    assert all(item.user_id == "u1" for item in results)


@pytest.mark.asyncio
async def test_retrieve_fallback_when_vector_fails(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore(fail_search=True)
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await memory.add(
        make_item("m1", "hello episodic memory", user_id="u1", importance=0.8)
    )
    await memory.add(make_item("m2", "irrelevant text", user_id="u1", importance=0.2))

    results = await memory.retrieve("episodic memory", user_id="u1", limit=1)

    assert [item.id for item in results] == ["m1"]


@pytest.mark.asyncio
async def test_update_syncs_document_and_vector(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await memory.add(make_item("m1", "old content", metadata={"session_id": "s1"}))
    updated = await memory.update(
        "m1",
        content="new content",
        importance=0.9,
        metadata={"source": "updated", "session_id": "s2"},
    )

    assert updated is True
    assert doc_store.rows["m1"]["content"] == "new content"
    assert doc_store.rows["m1"]["importance"] == 0.9
    assert doc_store.rows["m1"]["properties"]["source"] == "updated"
    assert vector_store.points["episodic:m1"]["metadata"]["content"] == "new content"
    assert vector_store.points["episodic:m1"]["metadata"]["session_id"] == "s2"


@pytest.mark.asyncio
async def test_remove_deletes_both_stores(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await memory.add(make_item("m1", "to remove"))

    removed = await memory.remove("m1")

    assert removed is True
    assert "m1" not in doc_store.rows
    assert "episodic:m1" not in vector_store.points


@pytest.mark.asyncio
async def test_has_memory_checks_document_store(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await memory.add(make_item("m1", "exists"))

    assert await memory.has_memory("m1") is True
    assert await memory.has_memory("m2") is False


@pytest.mark.asyncio
async def test_clear_only_removes_episodic(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await doc_store.add_memory(
        memory_id="d1",
        user_id="u1",
        content="document memory",
        memory_type="document",
        timestamp=int(datetime.now().timestamp()),
        importance=0.5,
        properties={},
    )
    await memory.add(make_item("m1", "episodic one"))
    await memory.add(make_item("m2", "episodic two"))

    await memory.clear()

    assert "m1" not in doc_store.rows
    assert "m2" not in doc_store.rows
    assert "d1" in doc_store.rows


@pytest.mark.asyncio
async def test_get_stats_returns_merged_stats(config):
    doc_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    memory = EpisodicMemory(
        config=config,
        document_store=doc_store,
        vector_store=vector_store,
    )

    await memory.add(
        make_item("m1", "alpha", metadata={"session_id": "s1"}, importance=0.6)
    )
    await memory.add(
        make_item("m2", "beta", metadata={"session_id": "s2"}, importance=0.8)
    )

    stats = await memory.get_stats()

    assert stats["count"] == 2
    assert stats["memory_type"] == "episodic"
    assert stats["sessions_count"] == 2
    assert stats["vector_store"]["store_type"] == "fake-vector"
    assert "document_store" in stats
