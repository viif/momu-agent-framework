from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from momu_agent.context.builder import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    count_tokens,
)
from momu_agent.core.message import Message


@pytest.mark.asyncio
async def test_build_runs_gssc_pipeline_in_order():
    llm = AsyncMock()
    builder = ContextBuilder(llm=llm)
    builder._gather = AsyncMock(return_value=[ContextPacket(content="g")])
    builder._select = Mock(return_value=[ContextPacket(content="s")])
    builder._structure = Mock(return_value="structured")
    builder._compress = AsyncMock(return_value="compressed")

    result = await builder.build(
        user_query="query",
        conversation_history=[],
        system_instructions="sys",
        additional_packets=[],
    )

    assert result == "compressed"
    builder._gather.assert_awaited_once()
    builder._select.assert_called_once()
    builder._structure.assert_called_once()
    builder._compress.assert_awaited_once_with("structured")


@pytest.mark.asyncio
async def test_gather_includes_system_history_and_additional_packets():
    llm = AsyncMock()
    history = [Message(content=f"msg-{i}", role="user") for i in range(12)]
    extra = ContextPacket(content="extra", metadata={"type": "tool_result"})
    builder = ContextBuilder(llm=llm)

    packets = await builder._gather(
        user_query="hello",
        conversation_history=history,
        system_instructions="system prompt",
        additional_packets=[extra],
    )

    instruction_packets = [
        p for p in packets if p.metadata.get("type") == "instructions"
    ]
    history_packets = [p for p in packets if p.metadata.get("type") == "history"]

    assert len(instruction_packets) == 1
    assert instruction_packets[0].content == "system prompt"
    assert len(history_packets) == 1
    assert history_packets[0].metadata["count"] == 6
    assert "msg-0" not in history_packets[0].content
    assert "msg-5" not in history_packets[0].content
    assert "msg-6" in history_packets[0].content
    assert "msg-11" in history_packets[0].content
    assert any(packet.content == "extra" for packet in packets)


@pytest.mark.asyncio
async def test_gather_extracts_only_user_query_from_structured_user_history():
    llm = AsyncMock()
    builder = ContextBuilder(llm=llm)
    history = [
        Message(
            content=("[Task]\n用户问题：之前的问题\n\n[Output]\n1. 结论\n2. 依据"),
            role="user",
        ),
        Message(content="之前的回答", role="assistant"),
    ]

    packets = await builder._gather(
        user_query="hello",
        conversation_history=history,
        system_instructions=None,
        additional_packets=[],
    )

    history_packet = next(p for p in packets if p.metadata.get("type") == "history")
    assert "[user] 之前的问题" in history_packet.content
    assert "[Task]" not in history_packet.content
    assert "[Output]" not in history_packet.content
    assert "[assistant] 之前的回答" in history_packet.content


@pytest.mark.asyncio
async def test_gather_falls_back_to_original_user_history_when_extraction_is_empty():
    llm = AsyncMock()
    builder = ContextBuilder(llm=llm)
    history = [
        Message(content="[Task]\n用户问题：\n[Output]\n1. 结论", role="user"),
    ]

    packets = await builder._gather(
        user_query="hello",
        conversation_history=history,
        system_instructions=None,
        additional_packets=[],
    )

    history_packet = next(p for p in packets if p.metadata.get("type") == "history")
    assert history_packet.content == "[user] [Task]\n用户问题：\n[Output]\n1. 结论"


def test_extract_user_query_from_history_returns_none_for_plain_text():
    assert ContextBuilder._extract_user_query_from_history("普通问题") is None


def test_extract_user_query_from_history_returns_query_only_for_structured_text():
    content = "[Task]\n用户问题：如何修复\n\n[Evidence]\n引用"
    assert ContextBuilder._extract_user_query_from_history(content) == "如何修复"


@pytest.mark.asyncio
async def test_gather_calls_memory_and_rag_with_async_payloads():
    llm = AsyncMock()
    memory_tool = AsyncMock()
    memory_tool.run = AsyncMock(side_effect=["state memo", "related memo"])
    rag_tool = AsyncMock()
    rag_tool.run = AsyncMock(return_value="rag facts")
    builder = ContextBuilder(llm=llm, memory_tool=memory_tool, rag_tool=rag_tool)

    packets = await builder._gather(
        user_query="python agent",
        conversation_history=[],
        system_instructions=None,
        additional_packets=[],
    )

    assert memory_tool.run.await_count == 2
    memory_tool.run.assert_any_await(
        {
            "action": "search",
            "query": "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
            "importance_threshold": 0.7,
            "limit": 5,
        }
    )
    memory_tool.run.assert_any_await(
        {
            "action": "search",
            "query": "python agent",
            "limit": 5,
        }
    )
    rag_tool.run.assert_awaited_once_with(
        {
            "action": "search",
            "query": "python agent",
            "limit": 5,
        }
    )

    types = {packet.metadata.get("type") for packet in packets}
    assert {"task_state", "related_memory", "knowledge_base"}.issubset(types)


@pytest.mark.asyncio
async def test_gather_tool_failures_do_not_raise(caplog):
    llm = AsyncMock()
    memory_tool = AsyncMock()
    memory_tool.run = AsyncMock(side_effect=RuntimeError("boom"))
    rag_tool = AsyncMock()
    rag_tool.run = AsyncMock(side_effect=RuntimeError("boom"))

    builder = ContextBuilder(llm=llm, memory_tool=memory_tool, rag_tool=rag_tool)
    with caplog.at_level("WARNING"):
        packets = await builder._gather(
            user_query="q",
            conversation_history=[Message(content="h", role="user")],
            system_instructions="sys",
            additional_packets=[],
        )

    assert any(p.metadata.get("type") == "instructions" for p in packets)
    assert any(p.metadata.get("type") == "history" for p in packets)
    assert "记忆检索失败" in caplog.text
    assert "RAG检索失败" in caplog.text


def test_select_keeps_instructions_and_applies_relevance_and_budget():
    llm = AsyncMock()
    cfg = ContextConfig(max_tokens=12, reserve_ratio=0.0, min_relevance=0.3)
    builder = ContextBuilder(llm=llm, config=cfg)

    instruction = ContextPacket(
        content="system", metadata={"type": "instructions"}, token_count=2
    )
    relevant = ContextPacket(content="python agent", token_count=5)
    low_rel = ContextPacket(content="other words", token_count=2)
    large = ContextPacket(content="python", token_count=10)

    selected = builder._select([instruction, relevant, low_rel, large], "python agent")

    assert instruction in selected
    assert relevant in selected
    assert low_rel not in selected
    assert large not in selected
    assert sum(packet.token_count for packet in selected) <= cfg.get_available_tokens()


def test_structure_contains_expected_sections():
    llm = AsyncMock()
    builder = ContextBuilder(llm=llm)
    packets = [
        ContextPacket(content="sys", metadata={"type": "instructions"}, token_count=1),
        ContextPacket(content="state", metadata={"type": "task_state"}, token_count=1),
        ContextPacket(
            content="evidence",
            metadata={"type": "knowledge_base"},
            token_count=1,
        ),
        ContextPacket(content="history", metadata={"type": "history"}, token_count=1),
    ]

    context = builder._structure(packets, "my task", None)

    assert "[Role & Policies]" in context
    assert "[Task]" in context
    assert "[State]" in context
    assert "[Evidence]" in context
    assert "[Context]" in context
    assert "[Output]" in context
    assert "用户问题：my task" in context


@pytest.mark.asyncio
async def test_compress_returns_original_when_disabled():
    llm = AsyncMock()
    cfg = ContextConfig(max_tokens=10, reserve_ratio=0.0, enable_compression=False)
    builder = ContextBuilder(llm=llm, config=cfg)

    text = "a\n" * 100
    assert await builder._compress(text) == text


@pytest.mark.asyncio
async def test_compress_uses_llm_summary_when_available(monkeypatch):
    cfg = ContextConfig(max_tokens=6, reserve_ratio=0.0, enable_compression=True)
    llm = AsyncMock()
    llm.invoke = AsyncMock(return_value="summary")
    builder = ContextBuilder(llm=llm, config=cfg)

    token_map = {
        "[Task]\nline1\nline2\nline3": 8,
        "summary": 4,
    }

    monkeypatch.setattr(
        "momu_agent.context.builder.count_tokens", lambda text: token_map.get(text, 1)
    )

    compressed = await builder._compress("[Task]\nline1\nline2\nline3")

    assert compressed == "summary"
    llm.invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_compress_fallback_to_truncate_when_llm_summary_still_over_budget(
    monkeypatch,
):
    cfg = ContextConfig(max_tokens=6, reserve_ratio=0.0, enable_compression=True)
    llm = AsyncMock()
    llm.invoke = AsyncMock(return_value="too long summary")
    builder = ContextBuilder(llm=llm, config=cfg)

    token_map = {
        "[Task]": 2,
        "line1": 2,
        "line2": 2,
        "line3": 2,
        "[Task]\nline1\nline2\nline3": 8,
        "too long summary": 7,
    }

    monkeypatch.setattr(
        "momu_agent.context.builder.count_tokens", lambda text: token_map.get(text, 1)
    )

    compressed = await builder._compress("[Task]\nline1\nline2\nline3")

    assert compressed == "[Task]\nline1\nline2"
    llm.invoke.assert_awaited_once()
    cfg = ContextConfig(max_tokens=6, reserve_ratio=0.0, enable_compression=True)
    llm = AsyncMock()
    llm.invoke = AsyncMock(side_effect=RuntimeError("llm down"))
    builder = ContextBuilder(llm=llm, config=cfg)

    token_map = {
        "[Task]": 2,
        "line1": 2,
        "line2": 2,
        "line3": 2,
        "[Task]\nline1\nline2\nline3": 8,
    }

    monkeypatch.setattr(
        "momu_agent.context.builder.count_tokens", lambda text: token_map.get(text, 1)
    )

    compressed = await builder._compress("[Task]\nline1\nline2\nline3")

    assert compressed == "[Task]\nline1\nline2"


def test_count_tokens_fallback_on_encoder_error(monkeypatch):
    def raise_error(_name):
        raise RuntimeError("no encoder")

    monkeypatch.setattr("momu_agent.context.builder.tiktoken.get_encoding", raise_error)

    value = count_tokens("abcd" * 5)
    assert value == 5


def test_select_prefers_newer_packet_under_same_relevance():
    llm = AsyncMock()
    cfg = ContextConfig(max_tokens=20, reserve_ratio=0.0, min_relevance=0.0)
    builder = ContextBuilder(llm=llm, config=cfg)

    old = ContextPacket(
        content="python agent",
        token_count=2,
        timestamp=datetime.now() - timedelta(hours=3),
    )
    new = ContextPacket(content="python agent", token_count=2, timestamp=datetime.now())

    selected = builder._select([old, new], "python agent")

    assert selected[0] == new
