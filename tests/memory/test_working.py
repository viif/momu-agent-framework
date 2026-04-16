from __future__ import annotations

import builtins
from datetime import datetime, timedelta

import pytest

from momu_agent.memory.base import MemoryConfig, MemoryItem
from momu_agent.memory.working import WorkingMemory


def make_memory(
    memory_id: str,
    content: str,
    *,
    importance: float = 0.5,
    user_id: str = "user-1",
    timestamp: datetime | None = None,
    metadata: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        memory_type="working",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
        metadata=metadata or {},
    )


@pytest.fixture
def memory_config() -> MemoryConfig:
    return MemoryConfig(
        working_memory_capacity=3,
        working_memory_tokens=20,
        working_memory_ttl_minutes=120,
        decay_factor=0.95,
    )


@pytest.fixture
def working_memory(memory_config: MemoryConfig) -> WorkingMemory:
    return WorkingMemory(memory_config)


def test_add_and_retrieve_uses_keyword_fallback(working_memory, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("sklearn unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    older = datetime.now() - timedelta(minutes=10)
    newer = datetime.now() - timedelta(minutes=5)

    matching = make_memory(
        "m1",
        "python asyncio testing notes",
        importance=0.9,
        user_id="u1",
        timestamp=newer,
    )
    forgotten = make_memory(
        "m2",
        "python hidden memory",
        importance=1.0,
        user_id="u1",
        timestamp=older,
        metadata={"forgotten": True},
    )
    other_user = make_memory(
        "m3",
        "python for another user",
        importance=0.8,
        user_id="u2",
        timestamp=older,
    )

    working_memory.add(matching)
    working_memory.add(forgotten)
    working_memory.add(other_user)

    results = working_memory.retrieve("python", user_id="u1")

    assert [memory.id for memory in results] == ["m1"]


def test_update_changes_content_importance_metadata_and_tokens(working_memory):
    memory = make_memory("m1", "alpha beta", importance=0.2)
    working_memory.add(memory)

    updated = working_memory.update(
        "m1",
        content="alpha beta gamma delta",
        importance=0.9,
        metadata={"source": "updated"},
    )

    assert updated is True
    assert memory.content == "alpha beta gamma delta"
    assert memory.importance == 0.9
    assert memory.metadata == {"source": "updated"}
    assert working_memory.current_tokens == 4


def test_remove_and_clear_update_internal_state(working_memory):
    first = make_memory("m1", "one two")
    second = make_memory("m2", "three four five")
    working_memory.add(first)
    working_memory.add(second)

    removed = working_memory.remove("m1")

    assert removed is True
    assert working_memory.has_memory("m1") is False
    assert working_memory.current_tokens == 3

    working_memory.clear()

    assert working_memory.get_all() == []
    assert working_memory.memory_heap == []
    assert working_memory.current_tokens == 0


def test_capacity_limit_removes_lowest_priority_memory(memory_config: MemoryConfig):
    memory_config.working_memory_capacity = 2
    working_memory = WorkingMemory(memory_config)

    low_priority = make_memory("m1", "low", importance=0.1)
    high_priority = make_memory("m2", "high value", importance=0.9)
    medium_priority = make_memory("m3", "medium value", importance=0.5)

    working_memory.add(low_priority)
    working_memory.add(high_priority)
    working_memory.add(medium_priority)

    remaining_ids = {memory.id for memory in working_memory.get_all()}

    assert remaining_ids == {"m2", "m3"}
    assert working_memory.has_memory("m1") is False


def test_token_limit_removes_lowest_priority_memory(memory_config: MemoryConfig):
    memory_config.working_memory_tokens = 5
    working_memory = WorkingMemory(memory_config)

    high_priority = make_memory("m1", "one two", importance=1.0)
    low_priority = make_memory("m2", "three four five six", importance=0.1)

    working_memory.add(high_priority)
    working_memory.add(low_priority)

    remaining_ids = [memory.id for memory in working_memory.get_all()]

    assert remaining_ids == ["m1"]
    assert working_memory.current_tokens == 2


def test_expire_old_memories_removes_stale_entries_and_updates_tokens(memory_config):
    memory_config.working_memory_ttl_minutes = 30
    working_memory = WorkingMemory(memory_config)

    expired = make_memory(
        "m1",
        "stale memory",
        timestamp=datetime.now() - timedelta(minutes=31),
    )
    fresh = make_memory(
        "m2",
        "fresh memory data",
        timestamp=datetime.now() - timedelta(minutes=5),
    )

    working_memory.add(expired)
    working_memory.add(fresh)

    stats = working_memory.get_stats()

    assert stats["count"] == 1
    assert [memory.id for memory in working_memory.get_all()] == ["m2"]
    assert working_memory.current_tokens == 3


def test_recent_important_and_context_summary_are_sorted(memory_config):
    working_memory = WorkingMemory(memory_config)
    oldest = datetime.now() - timedelta(minutes=20)
    middle = datetime.now() - timedelta(minutes=10)
    newest = datetime.now() - timedelta(minutes=1)

    low = make_memory("m1", "low priority", importance=0.2, timestamp=middle)
    high = make_memory("m2", "H" * 60, importance=0.9, timestamp=oldest)
    latest = make_memory(
        "m3",
        "recent memory " + "x" * 70,
        importance=0.5,
        timestamp=newest,
    )

    working_memory.add(low)
    working_memory.add(high)
    working_memory.add(latest)

    assert [memory.id for memory in working_memory.get_recent(2)] == ["m3", "m1"]
    assert [memory.id for memory in working_memory.get_important(2)] == ["m2", "m3"]

    full_summary = working_memory.get_context_summary(max_length=160)

    assert full_summary.startswith("Working Memory Context:\n")
    assert "H" * 60 in full_summary
    assert "recent memory" in full_summary

    truncated_summary = working_memory.get_context_summary(max_length=130)

    assert "recent memory" in truncated_summary
    assert "..." in truncated_summary


def test_forget_supports_importance_time_and_capacity_strategies(memory_config):
    memory_config.working_memory_capacity = 5
    memory_config.working_memory_ttl_minutes = 10_000
    working_memory = WorkingMemory(memory_config)

    working_memory.add(make_memory("m1", "low importance", importance=0.05))
    working_memory.add(
        make_memory(
            "m2",
            "old memory",
            importance=0.7,
            timestamp=datetime.now() - timedelta(days=2),
        )
    )
    working_memory.add(make_memory("m3", "high importance", importance=0.9))

    forgotten = working_memory.forget(strategy="importance_based", threshold=0.1)
    assert forgotten == 1
    assert working_memory.has_memory("m1") is False

    forgotten = working_memory.forget(strategy="time_based", max_age_days=1)
    assert forgotten == 1
    assert working_memory.has_memory("m2") is False

    working_memory.add(make_memory("m4", "mid 1", importance=0.4))
    working_memory.add(make_memory("m5", "mid 2", importance=0.3))
    working_memory.max_capacity = 2

    forgotten = working_memory.forget(strategy="capacity_based")

    assert forgotten == 1
    assert {memory.id for memory in working_memory.get_all()} == {"m3", "m4"}
