from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from momu_agent.rag.pipeline import (
    create_rag_pipeline,
    embed_query,
    index_chunks,
    load_and_chunk_texts,
    merge_snippets,
    rank,
    search_vectors,
)


class DummyEmbedder:
    def encode(self, texts):
        def to_vec(text: str) -> list[float]:
            lowered = text.lower()
            return [
                float(lowered.count("python")),
                float(lowered.count("agent")),
                float(len(lowered) % 7),
            ]

        if isinstance(texts, str):
            return to_vec(texts)
        return [to_vec(text) for text in texts]


class FakeDocumentStore:
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
            "created_at": datetime.now().isoformat(),
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
        return {
            "users_count": 1,
            "memories_count": len(self.rows),
            "memory_types": {"rag_chunk": len(self.rows)},
            "top_users": {"rag_user": len(self.rows)},
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


class FakeVectorStore:
    def __init__(self, fail_search: bool = False) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_search = fail_search

    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:
        point_ids = ids or [f"p-{index}" for index in range(len(vectors))]
        for point_id, vec, meta in zip(point_ids, vectors, metadata):
            self.rows[str(point_id)] = {
                "id": str(point_id),
                "vector": list(vec),
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

        items = list(self.rows.values())
        if where:
            for key, value in where.items():
                items = [item for item in items if item["metadata"].get(key) == value]

        results: list[dict[str, Any]] = []
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

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def delete_vectors(self, ids: list[str]) -> bool:
        for point_id in ids:
            self.rows.pop(str(point_id), None)
        return True

    async def delete_memories(self, memory_ids: list[str]) -> bool:
        to_remove = [
            point_id
            for point_id, row in self.rows.items()
            if row["metadata"].get("memory_id") in memory_ids
        ]
        for point_id in to_remove:
            self.rows.pop(point_id, None)
        return bool(to_remove)

    async def clear_collection(self) -> bool:
        self.rows.clear()
        return True

    async def get_collection_info(self) -> dict[str, Any]:
        return {
            "name": "fake",
            "vectors_count": len(self.rows),
            "indexed_vectors_count": len(self.rows),
            "points_count": len(self.rows),
            "segments_count": 1,
            "config": {"vector_size": 3},
        }

    async def get_collection_stats(self) -> dict[str, Any]:
        info = await self.get_collection_info()
        info["store_type"] = "fake-vector"
        return info


@pytest.fixture
def embedder() -> DummyEmbedder:
    return DummyEmbedder()


@pytest.fixture
def chunk_items() -> list[dict[str, Any]]:
    return [
        {
            "id": "c1",
            "content": "Python agent framework intro",
            "metadata": {"rag_namespace": "ns1", "source": "doc1"},
        },
        {
            "id": "c2",
            "content": "Graph retrieval for agent",
            "metadata": {"rag_namespace": "ns1", "source": "doc2"},
        },
    ]


def test_load_and_chunk_texts(tmp_path: Path):
    file_path = tmp_path / "input.txt"
    file_path.write_text("A" * 120 + "\n\n" + "B" * 120, encoding="utf-8")

    chunks = load_and_chunk_texts([str(file_path)], chunk_size=80, chunk_overlap=20)

    assert len(chunks) >= 2
    assert all("id" in chunk for chunk in chunks)
    assert all("content" in chunk for chunk in chunks)
    assert all(chunk["metadata"]["rag_namespace"] == "default" for chunk in chunks)


def test_embed_query_returns_dimension(embedder, monkeypatch):
    monkeypatch.setattr("momu_agent.rag.pipeline.get_dimension", lambda: 3)

    vec = embed_query("python agent", embedder=embedder)

    assert len(vec) == 3


@pytest.mark.asyncio
async def test_index_chunks_persists_to_stores(chunk_items, embedder, monkeypatch):
    monkeypatch.setattr("momu_agent.rag.pipeline.get_dimension", lambda: 3)
    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()

    count = await index_chunks(
        chunk_items,
        store=vector_store,
        document_store=document_store,
        embedder=embedder,
        namespace="ns1",
    )

    assert count == 2
    assert len(vector_store.rows) == 2
    assert len(document_store.rows) == 2
    assert all(
        row["memory_type"] == "rag_chunk" for row in document_store.rows.values()
    )


@pytest.mark.asyncio
async def test_search_vectors_with_where_and_threshold(
    chunk_items, embedder, monkeypatch
):
    monkeypatch.setattr("momu_agent.rag.pipeline.get_dimension", lambda: 3)
    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()
    await index_chunks(
        chunk_items,
        store=vector_store,
        document_store=document_store,
        embedder=embedder,
        namespace="ns1",
    )

    hits = await search_vectors(
        "python agent",
        top_k=5,
        score_threshold=0.1,
        namespace="ns1",
        store=vector_store,
        document_store=document_store,
        embedder=embedder,
    )

    assert len(hits) >= 1
    assert all(hit["metadata"]["rag_namespace"] == "ns1" for hit in hits)


def test_rank_and_merge_snippets():
    ranked = rank(
        [
            {
                "id": "1",
                "score": 0.5,
                "metadata": {"memory_id": "1", "content": "python agent"},
            },
            {
                "id": "2",
                "score": 0.9,
                "metadata": {"memory_id": "2", "content": "agent memory"},
            },
        ],
        "python agent",
    )

    assert ranked[0]["memory_id"] in {"1", "2"}
    merged = merge_snippets(ranked, max_chars=20)
    assert merged
    assert len(merged) <= 20


@pytest.mark.asyncio
async def test_search_vectors_fallback_to_document_store(embedder, monkeypatch):
    monkeypatch.setattr("momu_agent.rag.pipeline.get_dimension", lambda: 3)
    vector_store = FakeVectorStore(fail_search=True)
    document_store = FakeDocumentStore()

    await document_store.add_memory(
        memory_id="c1",
        user_id="rag_user",
        content="python agent fallback",
        memory_type="rag_chunk",
        timestamp=int(datetime.now().timestamp()),
        importance=0.5,
        properties={"rag_namespace": "ns1", "memory_id": "c1"},
    )

    hits = await search_vectors(
        "python",
        top_k=3,
        namespace="ns1",
        store=vector_store,
        document_store=document_store,
        embedder=embedder,
    )

    assert len(hits) == 1
    assert hits[0]["metadata"]["memory_id"] == "c1"


@pytest.mark.asyncio
async def test_create_rag_pipeline_end_to_end(tmp_path: Path, embedder, monkeypatch):
    monkeypatch.setattr("momu_agent.rag.pipeline.get_dimension", lambda: 3)

    file_path = tmp_path / "rag.txt"
    file_path.write_text(
        "Python agent memory system.\n\nRAG retrieval returns context.",
        encoding="utf-8",
    )

    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()
    pipeline = create_rag_pipeline(
        chunk_size=50,
        chunk_overlap=10,
        top_k=5,
        namespace="ns1",
        vector_store=vector_store,
        document_store=document_store,
        embedder=embedder,
    )

    indexed = await pipeline["add_documents"]([str(file_path)])
    assert indexed >= 1

    results = await pipeline["search"]("python agent")
    assert len(results) >= 1

    context = merge_snippets(results, max_chars=300)
    assert context

    stats = await pipeline["get_stats"]()
    assert stats["namespace"] == "ns1"
    assert "vector" in stats
    assert "document" in stats
