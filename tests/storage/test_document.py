from collections.abc import AsyncGenerator

import pytest

from momu_agent.storage.document import DocumentStore, SQLiteDocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
async def store(db_path) -> AsyncGenerator[SQLiteDocumentStore, None]:
    s = SQLiteDocumentStore(db_path)
    yield s
    await s.close()


def make_kwargs(**overrides) -> dict:
    base = dict(
        memory_id="m1",
        user_id="u1",
        content="test content",
        memory_type="episodic",
        timestamp=1_000_000,
        importance=0.5,
    )
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_same_db_path_returns_same_instance(db_path):
    a = SQLiteDocumentStore(db_path)
    b = SQLiteDocumentStore(db_path)
    assert a is b


def test_different_db_paths_return_different_instances(tmp_path):
    a = SQLiteDocumentStore(str(tmp_path / "a.db"))
    b = SQLiteDocumentStore(str(tmp_path / "b.db"))
    assert a is not b


# ---------------------------------------------------------------------------
# add_memory / get_memory
# ---------------------------------------------------------------------------


async def test_add_and_get_memory_round_trips(store):
    kwargs = make_kwargs(properties={"tag": "v1"})
    await store.add_memory(**kwargs)
    result = await store.get_memory("m1")
    assert result is not None
    assert result["memory_id"] == "m1"
    assert result["user_id"] == "u1"
    assert result["content"] == "test content"
    assert result["memory_type"] == "episodic"
    assert result["timestamp"] == 1_000_000
    assert result["importance"] == 0.5
    assert result["properties"] == {"tag": "v1"}


async def test_none_properties_returns_empty_dict(store):
    await store.add_memory(**make_kwargs())
    result = await store.get_memory("m1")
    assert result["properties"] == {}


async def test_get_memory_returns_none_for_missing_id(store):
    assert await store.get_memory("nonexistent") is None


async def test_add_memory_upserts_on_duplicate_id(store):
    await store.add_memory(**make_kwargs())
    await store.add_memory(**make_kwargs(content="updated"))
    result = await store.get_memory("m1")
    assert result["content"] == "updated"


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------


async def test_search_by_user_id(store):
    await store.add_memory(**make_kwargs(memory_id="m1", user_id="u1"))
    await store.add_memory(**make_kwargs(memory_id="m2", user_id="u2"))
    results = await store.search_memories(user_id="u1")
    assert len(results) == 1
    assert results[0]["user_id"] == "u1"


async def test_search_by_memory_type(store):
    await store.add_memory(**make_kwargs(memory_id="m1", memory_type="episodic"))
    await store.add_memory(**make_kwargs(memory_id="m2", memory_type="semantic"))
    results = await store.search_memories(memory_type="semantic")
    assert len(results) == 1
    assert results[0]["memory_type"] == "semantic"


async def test_search_by_time_range(store):
    await store.add_memory(**make_kwargs(memory_id="m1", timestamp=100))
    await store.add_memory(**make_kwargs(memory_id="m2", timestamp=200))
    await store.add_memory(**make_kwargs(memory_id="m3", timestamp=300))
    results = await store.search_memories(start_time=150, end_time=250)
    assert len(results) == 1
    assert results[0]["memory_id"] == "m2"


async def test_search_by_importance_threshold(store):
    await store.add_memory(**make_kwargs(memory_id="m1", importance=0.3))
    await store.add_memory(**make_kwargs(memory_id="m2", importance=0.7))
    results = await store.search_memories(importance_threshold=0.5)
    assert len(results) == 1
    assert results[0]["memory_id"] == "m2"


async def test_search_ordered_by_importance_desc_then_timestamp_desc(store):
    await store.add_memory(**make_kwargs(memory_id="m1", importance=0.5, timestamp=100))
    await store.add_memory(**make_kwargs(memory_id="m2", importance=0.9, timestamp=50))
    await store.add_memory(**make_kwargs(memory_id="m3", importance=0.5, timestamp=200))
    results = await store.search_memories()
    assert results[0]["memory_id"] == "m2"
    assert results[1]["memory_id"] == "m3"
    assert results[2]["memory_id"] == "m1"


async def test_search_limit_respected(store):
    for i in range(5):
        await store.add_memory(**make_kwargs(memory_id=f"m{i}"))
    results = await store.search_memories(limit=3)
    assert len(results) == 3


async def test_search_returns_empty_list_when_no_matches(store):
    results = await store.search_memories(user_id="nobody")
    assert results == []


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------


async def test_update_content(store):
    await store.add_memory(**make_kwargs())
    assert await store.update_memory("m1", content="new content")
    result = await store.get_memory("m1")
    assert result["content"] == "new content"


async def test_update_importance(store):
    await store.add_memory(**make_kwargs())
    assert await store.update_memory("m1", importance=0.9)
    result = await store.get_memory("m1")
    assert result["importance"] == 0.9


async def test_update_properties(store):
    await store.add_memory(**make_kwargs())
    assert await store.update_memory("m1", properties={"k": "v"})
    result = await store.get_memory("m1")
    assert result["properties"] == {"k": "v"}


async def test_update_all_none_returns_false(store):
    await store.add_memory(**make_kwargs())
    assert not await store.update_memory("m1")


async def test_update_nonexistent_returns_false(store):
    assert not await store.update_memory("ghost", content="x")


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------


async def test_delete_existing_returns_true_and_removes(store):
    await store.add_memory(**make_kwargs())
    assert await store.delete_memory("m1")
    assert await store.get_memory("m1") is None


async def test_delete_nonexistent_returns_false(store):
    assert not await store.delete_memory("ghost")


# ---------------------------------------------------------------------------
# get_database_stats
# ---------------------------------------------------------------------------


async def test_database_stats_keys(store):
    stats = await store.get_database_stats()
    for key in (
        "users_count",
        "memories_count",
        "memory_types",
        "top_users",
        "store_type",
        "db_path",
    ):
        assert key in stats


async def test_database_stats_counts(store):
    await store.add_memory(**make_kwargs(memory_id="m1", user_id="u1"))
    await store.add_memory(**make_kwargs(memory_id="m2", user_id="u2"))
    stats = await store.get_database_stats()
    assert stats["memories_count"] == 2
    assert stats["users_count"] == 2


async def test_database_stats_memory_types_distribution(store):
    await store.add_memory(**make_kwargs(memory_id="m1", memory_type="episodic"))
    await store.add_memory(**make_kwargs(memory_id="m2", memory_type="semantic"))
    await store.add_memory(**make_kwargs(memory_id="m3", memory_type="episodic"))
    stats = await store.get_database_stats()
    assert stats["memory_types"]["episodic"] == 2
    assert stats["memory_types"]["semantic"] == 1


# ---------------------------------------------------------------------------
# add_document / get_document
# ---------------------------------------------------------------------------


async def test_add_document_returns_uuid(store):
    doc_id = await store.add_document("hello")
    assert len(doc_id) == 36  # UUID4 format


async def test_add_document_stores_as_document_type(store):
    doc_id = await store.add_document("hello")
    result = await store.get_document(doc_id)
    assert result["memory_type"] == "document"


async def test_add_document_uses_metadata_user_id(store):
    doc_id = await store.add_document("hello", metadata={"user_id": "u99"})
    result = await store.get_document(doc_id)
    assert result["user_id"] == "u99"


async def test_add_document_without_metadata_defaults_to_system(store):
    doc_id = await store.add_document("hello")
    result = await store.get_document(doc_id)
    assert result["user_id"] == "system"


async def test_get_document_returns_none_for_missing_id(store):
    assert await store.get_document("no-such-doc") is None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


def test_document_store_is_abstract():
    assert issubclass(SQLiteDocumentStore, DocumentStore)
