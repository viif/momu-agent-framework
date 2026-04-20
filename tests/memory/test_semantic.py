from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from momu_agent.memory.base import MemoryConfig, MemoryItem
from momu_agent.memory.semantic import SemanticMemory
from momu_agent.storage.graph import GraphStore, KuzuGraphStore
from momu_agent.storage.vector import ChromaVectorStore, VectorStore


class DummyEmbedder:
    def encode(self, text: str):
        lowered = text.lower()
        return [
            float(lowered.count("python")),
            float(lowered.count("graph")),
            float(lowered.count("memory")),
        ]


class FakeGraphStore(GraphStore):
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: dict[tuple[str, str, str], dict[str, Any]] = {}

    async def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "properties": dict(properties or {}),
            "created_at": self.entities.get(entity_id, {}).get("created_at", 0),
            "updated_at": 0,
        }
        return True

    async def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        if from_entity_id not in self.entities or to_entity_id not in self.entities:
            return False
        self.relationships[(from_entity_id, to_entity_id, relationship_type)] = {
            "from": from_entity_id,
            "to": to_entity_id,
            "relationship": {
                "type": relationship_type,
                "properties": dict(properties or {}),
                "created_at": 0,
                "updated_at": 0,
            },
        }
        return True

    async def find_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if max_depth <= 0 or limit <= 0 or entity_id not in self.entities:
            return []

        allowed = set(relationship_types or [])
        visited = {entity_id}
        queue: list[tuple[str, int, list[str]]] = [(entity_id, 0, [])]
        results: list[dict[str, Any]] = []

        while queue and len(results) < limit:
            current, depth, path = queue.pop(0)
            if depth >= max_depth:
                continue

            neighbors: list[tuple[str, str]] = []
            for (src, dst, rel_type), rel in self.relationships.items():
                if src == current:
                    neighbors.append((dst, rel_type))
                elif dst == current:
                    neighbors.append((src, rel_type))

            for neighbor, rel_type in neighbors:
                new_path = [*path, rel_type]
                if allowed and any(part not in allowed for part in new_path):
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                entity = dict(self.entities[neighbor])
                entity["distance"] = depth + 1
                entity["relationship_path"] = new_path
                results.append(entity)
                queue.append((neighbor, depth + 1, new_path))
                if len(results) >= limit:
                    break

        results.sort(key=lambda item: (item["distance"], item["name"]))
        return results[:limit]

    async def search_entities_by_name(
        self,
        name_pattern: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        allowed = set(entity_types or [])
        pattern = name_pattern.replace("\\", "")
        items = []
        for entity in self.entities.values():
            if allowed and entity["type"] not in allowed:
                continue
            if pattern and pattern not in entity["name"]:
                continue
            items.append(dict(entity))
        items.sort(key=lambda item: item["name"])
        return items[:limit]

    async def get_entity_relationships(self, entity_id: str) -> list[dict[str, Any]]:
        rows = []
        for (src, dst, rel_type), rel in self.relationships.items():
            if src == entity_id:
                rows.append(
                    {
                        "relationship": rel["relationship"],
                        "other_entity": dict(self.entities[dst]),
                        "direction": "outgoing",
                    }
                )
            elif dst == entity_id:
                rows.append(
                    {
                        "relationship": rel["relationship"],
                        "other_entity": dict(self.entities[src]),
                        "direction": "incoming",
                    }
                )
        return rows

    async def delete_entity(self, entity_id: str) -> bool:
        if entity_id not in self.entities:
            return False
        self.entities.pop(entity_id, None)
        self.relationships = {
            key: value
            for key, value in self.relationships.items()
            if key[0] != entity_id and key[1] != entity_id
        }
        return True

    async def clear_all(self) -> bool:
        self.entities.clear()
        self.relationships.clear()
        return True

    async def get_stats(self) -> dict[str, Any]:
        return {
            "total_nodes": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_nodes": len(self.entities),
            "store_type": "fake-graph",
            "db_path": "",
        }

    async def close(self) -> None:
        return None


class FakeVectorStore(VectorStore):
    def __init__(self, fail_search: bool = False, fail_add: bool = False) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.fail_search = fail_search
        self.fail_add = fail_add

    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:
        if self.fail_add:
            raise RuntimeError("vector add failed")
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

        results = []
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

    async def get_collection_info(self) -> dict[str, Any]:
        return {"vectors_count": len(self.points)}

    async def get_collection_stats(self) -> dict[str, Any]:
        return {"store_type": "fake-vector", "vectors_count": len(self.points)}


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
        memory_type="semantic",
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
        "momu_agent.memory.semantic.get_text_embedder", lambda: DummyEmbedder()
    )


@pytest.mark.asyncio
async def test_constructor_uses_default_stores(config):
    memory = SemanticMemory(config=config)

    assert isinstance(memory.graph_store, KuzuGraphStore)
    assert isinstance(memory.vector_store, ChromaVectorStore)


@pytest.mark.asyncio
async def test_constructor_uses_injected_stores(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    assert memory.graph_store is graph_store
    assert memory.vector_store is vector_store


@pytest.mark.asyncio
async def test_add_persists_graph_entities_relationships_and_vector(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    item = make_item("m1", "python graph memory", metadata={"concepts": ["python"]})
    memory_id = await memory.add(item)

    assert memory_id == "m1"
    assert "semantic:memory:m1" in graph_store.entities
    assert "semantic:user:u1" in graph_store.entities
    assert "semantic:concept:python" in graph_store.entities
    assert (
        "semantic:user:u1",
        "semantic:memory:m1",
        "OWNS",
    ) in graph_store.relationships
    assert (
        "semantic:memory:m1",
        "semantic:concept:python",
        "MENTIONS",
    ) in graph_store.relationships
    assert "semantic:m1" in vector_store.points


@pytest.mark.asyncio
async def test_retrieve_prefers_vector_and_applies_user_filter(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await memory.add(
        make_item(
            "m1",
            "python graph memory basics",
            user_id="u1",
            importance=0.6,
            timestamp=datetime.now() - timedelta(hours=2),
            metadata={"concepts": ["python", "graph"]},
        )
    )
    await memory.add(
        make_item(
            "m2",
            "python graph memory advanced",
            user_id="u1",
            importance=0.9,
            timestamp=datetime.now() - timedelta(minutes=5),
            metadata={"concepts": ["python", "graph", "memory"]},
        )
    )
    await memory.add(
        make_item(
            "m3",
            "python graph memory advanced",
            user_id="u2",
            importance=1.0,
            metadata={"concepts": ["python", "graph", "memory"]},
        )
    )

    results = await memory.retrieve("python graph memory", user_id="u1", limit=2)

    assert [item.id for item in results] == ["m2", "m1"]
    assert all(item.user_id == "u1" for item in results)


@pytest.mark.asyncio
async def test_retrieve_fallback_to_graph_when_vector_fails(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore(fail_search=True)
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await memory.add(
        make_item(
            "m1",
            "python graph memory",
            user_id="u1",
            importance=0.8,
            metadata={"concepts": ["python", "graph"]},
        )
    )
    await memory.add(
        make_item(
            "m2",
            "irrelevant text",
            user_id="u1",
            importance=0.2,
            metadata={"concepts": ["other"]},
        )
    )

    results = await memory.retrieve("python graph", user_id="u1", limit=1)

    assert [item.id for item in results] == ["m1"]


@pytest.mark.asyncio
async def test_update_rebuilds_graph_links_and_vector_payload(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await memory.add(make_item("m1", "old memory", metadata={"concepts": ["old"]}))
    updated = await memory.update(
        "m1",
        content="new python graph memory",
        importance=0.9,
        metadata={"concepts": ["python", "graph"], "source": "updated"},
    )

    assert updated is True
    props = graph_store.entities["semantic:memory:m1"]["properties"]
    assert props["content"] == "new python graph memory"
    assert props["importance"] == 0.9
    assert props["metadata"]["source"] == "updated"
    assert "semantic:concept:python" in graph_store.entities
    assert (
        vector_store.points["semantic:m1"]["metadata"]["content"]
        == "new python graph memory"
    )


@pytest.mark.asyncio
async def test_remove_deletes_graph_and_vector(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await memory.add(make_item("m1", "to remove", metadata={"concepts": ["python"]}))

    removed = await memory.remove("m1")

    assert removed is True
    assert "semantic:memory:m1" not in graph_store.entities
    assert "semantic:m1" not in vector_store.points


@pytest.mark.asyncio
async def test_has_memory_checks_graph_entity(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await memory.add(make_item("m1", "exists"))

    assert await memory.has_memory("m1") is True
    assert await memory.has_memory("m2") is False


@pytest.mark.asyncio
async def test_clear_only_removes_semantic_scope(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await graph_store.add_entity("other:1", "other", "other_type", {"value": 1})
    await memory.add(make_item("m1", "semantic one", metadata={"concepts": ["one"]}))
    await memory.add(make_item("m2", "semantic two", metadata={"concepts": ["two"]}))

    await memory.clear()

    assert "semantic:memory:m1" not in graph_store.entities
    assert "semantic:memory:m2" not in graph_store.entities
    assert "other:1" in graph_store.entities


@pytest.mark.asyncio
async def test_get_stats_returns_semantic_aggregates(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    await memory.add(
        make_item("m1", "python", metadata={"concepts": ["python"]}, importance=0.6)
    )
    await memory.add(
        make_item(
            "m2",
            "graph memory",
            metadata={"concepts": ["graph", "memory"]},
            importance=0.8,
        )
    )

    stats = await memory.get_stats()

    assert stats["count"] == 2
    assert stats["memory_type"] == "semantic"
    assert stats["concepts_count"] == 3
    assert stats["graph_store"]["store_type"] == "fake-graph"
    assert stats["vector_store"]["store_type"] == "fake-vector"


@pytest.mark.asyncio
async def test_add_keeps_graph_when_vector_write_fails(config):
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore(fail_add=True)
    memory = SemanticMemory(
        config=config,
        graph_store=graph_store,
        vector_store=vector_store,
    )

    memory_id = await memory.add(
        make_item("m1", "python graph memory", metadata={"concepts": ["python"]})
    )

    assert memory_id == "m1"
    assert "semantic:memory:m1" in graph_store.entities
    assert "semantic:m1" not in vector_store.points
