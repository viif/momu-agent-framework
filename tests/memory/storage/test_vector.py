from collections.abc import AsyncGenerator
from typing import cast

import chromadb
import pytest

from momu_agent.memory.storage.vector import ChromaVectorStore, VectorStore


class _SimpleEF(chromadb.EmbeddingFunction):
    """确定性三维 embedding，仅用于测试。"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "simple-test-ef"

    @staticmethod
    def build_from_config(config: dict) -> "_SimpleEF":
        _ = config
        return _SimpleEF()

    def get_config(self) -> dict:
        return {}

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return cast(
            chromadb.Embeddings,
            [
                [float(sum(t.encode()) % 100) / 100.0, len(t) / 200.0, 0.5]
                for t in input
            ],
        )


@pytest.fixture
def collection_path(tmp_path) -> str:
    return str(tmp_path / "chroma")


@pytest.fixture
def ef() -> _SimpleEF:
    return _SimpleEF()


@pytest.fixture
async def store(collection_path, ef) -> AsyncGenerator[ChromaVectorStore, None]:
    s = ChromaVectorStore(collection_path, embedding_function=ef)
    yield s
    await s.close()


def make_item(idx: int = 1, **overrides) -> tuple[list[float], dict, str]:
    base_vector = [0.1 * idx, 0.2, 0.5]
    base_meta = {
        "memory_id": f"m{idx}",
        "user_id": "u1",
        "memory_type": "episodic",
        "importance": 0.5,
        "content": f"text-{idx}",
    }
    base_id = f"p{idx}"
    vector = overrides.pop("vector", base_vector)
    meta = {**base_meta, **overrides.pop("meta", {})}
    point_id = overrides.pop("point_id", base_id)
    return vector, meta, point_id


def test_same_path_and_name_returns_same_instance(collection_path, ef):
    a = ChromaVectorStore(collection_path, embedding_function=ef)
    b = ChromaVectorStore(collection_path, embedding_function=ef)
    assert a is b


def test_different_paths_return_different_instances(tmp_path, ef):
    a = ChromaVectorStore(str(tmp_path / "a"), embedding_function=ef)
    b = ChromaVectorStore(str(tmp_path / "b"), embedding_function=ef)
    assert a is not b


async def test_add_vectors_success(store):
    v1, m1, id1 = make_item(1)
    v2, m2, id2 = make_item(2)
    ok = await store.add_vectors([v1, v2], [m1, m2], [id1, id2])
    assert ok

    info = await store.get_collection_info()
    assert info["points_count"] == 2


async def test_add_vectors_empty_returns_false(store):
    assert not await store.add_vectors([], [])


async def test_add_vectors_length_mismatch_returns_false(store):
    v1, m1, id1 = make_item(1)
    assert not await store.add_vectors([v1], [m1, m1], [id1])


async def test_search_similar_returns_id_score_metadata(store):
    v1, m1, id1 = make_item(1)
    await store.add_vectors([v1], [m1], [id1])

    results = await store.search_similar(v1)
    assert len(results) == 1
    assert results[0]["id"] == id1
    assert "score" in results[0]
    assert results[0]["metadata"]["memory_id"] == "m1"


async def test_search_similar_where_filter(store):
    v1, m1, id1 = make_item(1, meta={"user_id": "u1"})
    v2, m2, id2 = make_item(2, meta={"user_id": "u2"})
    await store.add_vectors([v1, v2], [m1, m2], [id1, id2])

    results = await store.search_similar(v1, where={"user_id": "u1"})
    assert len(results) == 1
    assert results[0]["metadata"]["user_id"] == "u1"


async def test_search_similar_score_threshold(store):
    v1, m1, id1 = make_item(1)
    await store.add_vectors([v1], [m1], [id1])

    low = await store.search_similar(v1, score_threshold=0.1)
    high = await store.search_similar(v1, score_threshold=0.99)
    assert len(low) == 1
    assert len(high) == 1


async def test_search_similar_limit_respected(store):
    vectors, metas, ids = [], [], []
    for i in range(1, 6):
        v, m, pid = make_item(i)
        vectors.append(v)
        metas.append(m)
        ids.append(pid)
    await store.add_vectors(vectors, metas, ids)

    results = await store.search_similar([0.1, 0.2, 0.5], limit=2)
    assert len(results) <= 2


async def test_delete_vectors_success(store):
    v1, m1, id1 = make_item(1)
    await store.add_vectors([v1], [m1], [id1])

    assert await store.delete_vectors([id1])
    info = await store.get_collection_info()
    assert info["points_count"] == 0


async def test_delete_vectors_empty_is_true(store):
    assert await store.delete_vectors([])


async def test_delete_memories_by_memory_id(store):
    v1, m1, id1 = make_item(1)
    v2, m2, id2 = make_item(2)
    await store.add_vectors([v1, v2], [m1, m2], [id1, id2])

    assert await store.delete_memories(["m1"])
    results = await store.search_similar(v2, limit=10)
    memory_ids = {r["metadata"]["memory_id"] for r in results}
    assert "m1" not in memory_ids


async def test_delete_memories_missing_returns_false(store):
    assert not await store.delete_memories(["missing"])


async def test_clear_collection(store):
    v1, m1, id1 = make_item(1)
    await store.add_vectors([v1], [m1], [id1])

    assert await store.clear_collection()
    info = await store.get_collection_info()
    assert info["points_count"] == 0


async def test_get_collection_info_and_stats(store):
    info = await store.get_collection_info()
    for key in (
        "name",
        "vectors_count",
        "indexed_vectors_count",
        "points_count",
        "segments_count",
        "config",
    ):
        assert key in info

    stats = await store.get_collection_stats()
    assert stats["store_type"] == "chroma"
    assert "collection_path" in stats


async def test_close_keeps_persistent_data(collection_path, ef):
    store1 = ChromaVectorStore(collection_path, embedding_function=ef)
    v1, m1, id1 = make_item(1)
    await store1.add_vectors([v1], [m1], [id1])
    await store1.close()

    store2 = ChromaVectorStore(collection_path, embedding_function=ef)
    results = await store2.search_similar(v1)
    assert any(r["metadata"]["memory_id"] == "m1" for r in results)


def test_vector_store_is_abstract():
    assert issubclass(ChromaVectorStore, VectorStore)
