from collections.abc import AsyncGenerator

import pytest

from momu_agent.storage.graph import GraphStore, KuzuGraphStore


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "graph.kuzu")


@pytest.fixture
async def store(db_path) -> AsyncGenerator[KuzuGraphStore, None]:
    s = KuzuGraphStore(db_path)
    yield s
    await s.close()


async def seed_entity(
    store: KuzuGraphStore,
    entity_id: str,
    name: str,
    entity_type: str = "person",
    properties: dict | None = None,
) -> None:
    ok = await store.add_entity(
        entity_id=entity_id,
        name=name,
        entity_type=entity_type,
        properties=properties,
    )
    assert ok


def test_same_db_path_returns_same_instance(db_path):
    a = KuzuGraphStore(db_path)
    b = KuzuGraphStore(db_path)
    assert a is b


def test_different_db_paths_return_different_instances(tmp_path):
    a = KuzuGraphStore(str(tmp_path / "a.kuzu"))
    b = KuzuGraphStore(str(tmp_path / "b.kuzu"))
    assert a is not b


async def test_add_and_search_entity(store):
    assert await store.add_entity("e1", "Alice", "person", {"role": "dev"})
    rows = await store.search_entities_by_name("Ali")
    assert len(rows) == 1
    assert rows[0]["id"] == "e1"
    assert rows[0]["properties"] == {"role": "dev"}


async def test_add_entity_upserts_on_duplicate_id(store):
    assert await store.add_entity("e1", "Alice", "person", {"v": 1})
    assert await store.add_entity("e1", "Alice-2", "person", {"v": 2})
    rows = await store.search_entities_by_name("Alice")
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice-2"
    assert rows[0]["properties"] == {"v": 2}


async def test_none_properties_returns_empty_dict(store):
    assert await store.add_entity("e1", "Alice", "person")
    rows = await store.search_entities_by_name("Alice")
    assert rows[0]["properties"] == {}


async def test_add_relationship_success(store):
    await seed_entity(store, "e1", "Alice")
    await seed_entity(store, "e2", "Bob")
    ok = await store.add_relationship("e1", "e2", "KNOWS", {"since": 2024})
    assert ok

    rels = await store.get_entity_relationships("e1")
    assert len(rels) == 1
    assert rels[0]["direction"] == "outgoing"
    assert rels[0]["relationship"]["type"] == "KNOWS"
    assert rels[0]["relationship"]["properties"] == {"since": 2024}
    assert rels[0]["other_entity"]["id"] == "e2"


async def test_add_relationship_returns_false_when_endpoint_missing(store):
    await seed_entity(store, "e1", "Alice")
    assert not await store.add_relationship("e1", "missing", "KNOWS", None)


async def test_add_relationship_deduplicates_by_triplet(store):
    await seed_entity(store, "e1", "Alice")
    await seed_entity(store, "e2", "Bob")
    assert await store.add_relationship("e1", "e2", "KNOWS", {"v": 1})
    assert await store.add_relationship("e1", "e2", "KNOWS", {"v": 2})

    rels = await store.get_entity_relationships("e1")
    assert len(rels) == 1
    assert rels[0]["relationship"]["properties"] == {"v": 2}


async def test_find_related_entities_respects_depth_and_limit(store):
    await seed_entity(store, "e1", "A")
    await seed_entity(store, "e2", "B")
    await seed_entity(store, "e3", "C")
    await seed_entity(store, "e4", "D")

    assert await store.add_relationship("e1", "e2", "R1")
    assert await store.add_relationship("e2", "e3", "R2")
    assert await store.add_relationship("e3", "e4", "R3")

    depth1 = await store.find_related_entities("e1", max_depth=1, limit=10)
    assert [item["id"] for item in depth1] == ["e2"]

    depth2 = await store.find_related_entities("e1", max_depth=2, limit=10)
    assert {item["id"] for item in depth2} == {"e2", "e3"}

    limited = await store.find_related_entities("e1", max_depth=3, limit=1)
    assert len(limited) == 1


async def test_find_related_entities_filters_relationship_types(store):
    await seed_entity(store, "e1", "A")
    await seed_entity(store, "e2", "B")
    await seed_entity(store, "e3", "C")

    assert await store.add_relationship("e1", "e2", "R1")
    assert await store.add_relationship("e2", "e3", "R2")

    only_r1 = await store.find_related_entities(
        "e1", relationship_types=["R1"], max_depth=2, limit=10
    )
    assert [item["id"] for item in only_r1] == ["e2"]

    both = await store.find_related_entities(
        "e1", relationship_types=["R1", "R2"], max_depth=2, limit=10
    )
    assert {item["id"] for item in both} == {"e2", "e3"}


async def test_search_entities_by_name_and_type_filter(store):
    await seed_entity(store, "e1", "Alice", "person")
    await seed_entity(store, "e2", "AliceOrg", "organization")

    rows = await store.search_entities_by_name("Alice")
    assert len(rows) == 2

    people = await store.search_entities_by_name("Alice", entity_types=["person"])
    assert len(people) == 1
    assert people[0]["id"] == "e1"


async def test_get_entity_relationships_incoming_and_outgoing(store):
    await seed_entity(store, "e1", "A")
    await seed_entity(store, "e2", "B")
    await seed_entity(store, "e3", "C")

    assert await store.add_relationship("e1", "e2", "OUT")
    assert await store.add_relationship("e3", "e1", "IN")

    rels = await store.get_entity_relationships("e1")
    directions = {item["direction"] for item in rels}
    assert directions == {"incoming", "outgoing"}


async def test_delete_entity_existing_returns_true_and_removes_edges(store):
    await seed_entity(store, "e1", "A")
    await seed_entity(store, "e2", "B")
    assert await store.add_relationship("e1", "e2", "R")

    assert await store.delete_entity("e1")

    rels_e2 = await store.get_entity_relationships("e2")
    assert rels_e2 == []


async def test_delete_entity_missing_returns_false(store):
    assert not await store.delete_entity("missing")


async def test_clear_all_resets_stats(store):
    await seed_entity(store, "e1", "A")
    await seed_entity(store, "e2", "B")
    assert await store.add_relationship("e1", "e2", "R")

    assert await store.clear_all()
    stats = await store.get_stats()
    assert stats["total_nodes"] == 0
    assert stats["total_relationships"] == 0
    assert stats["entity_nodes"] == 0


async def test_get_stats_keys(store):
    stats = await store.get_stats()
    for key in (
        "total_nodes",
        "total_relationships",
        "entity_nodes",
        "store_type",
        "db_path",
    ):
        assert key in stats


async def test_close_allows_reinitialization(db_path):
    store1 = KuzuGraphStore(db_path)
    assert await store1.add_entity("e1", "Alice", "person")
    await store1.close()

    store2 = KuzuGraphStore(db_path)
    rows = await store2.search_entities_by_name("Alice")
    assert len(rows) == 1


def test_graph_store_is_abstract():
    assert issubclass(KuzuGraphStore, GraphStore)
