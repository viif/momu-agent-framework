from unittest.mock import AsyncMock, Mock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools import MemoryTool
from momu_agent.tools.builtin.memory import memory_add, memory_get_stats, memory_search


class TestMemoryTool:
    @pytest.fixture
    def mock_manager(self):
        manager = Mock()
        manager.add_memory = AsyncMock(return_value="memory-1")
        manager.retrieve_memories = AsyncMock(return_value=[])
        manager.update_memory = AsyncMock(return_value=True)
        manager.remove_memory = AsyncMock(return_value=True)
        manager.get_memory_stats = AsyncMock(
            return_value={
                "user_id": "u1",
                "enabled_types": ["working", "episodic", "semantic"],
                "total_memories": 3,
                "memories_by_type": {
                    "working": {"count": 1},
                    "episodic": {"count": 1},
                    "semantic": {"count": 1},
                },
            }
        )
        manager.forget_memories = AsyncMock(return_value=2)
        manager.clear_all_memories = AsyncMock(return_value=None)
        return manager

    @pytest.fixture
    def memory_tool(self, mock_manager):
        return MemoryTool(memory_manager=mock_manager)

    @pytest.mark.asyncio
    async def test_add_success(self, memory_tool, mock_manager):
        result = await memory_tool.run(
            {
                "action": "add",
                "content": "remember this",
                "memory_type": "semantic",
                "metadata": {"concepts": ["memory"]},
                "auto_classify": False,
            }
        )

        assert "已添加记忆: memory-1" in result
        assert "记忆类型: semantic" in result
        mock_manager.add_memory.assert_awaited_once_with(
            content="remember this",
            memory_type="semantic",
            importance=None,
            metadata={"concepts": ["memory"]},
            auto_classify=False,
        )

    @pytest.mark.asyncio
    async def test_add_missing_content_raises(self, memory_tool):
        with pytest.raises(ToolException, match="非空 content"):
            await memory_tool.run({"action": "add", "content": "   "})

    @pytest.mark.asyncio
    async def test_search_success_formats_results(self, memory_tool, mock_manager):
        memory = Mock(
            id="m1",
            memory_type="semantic",
            content="Python memory knowledge",
            metadata={"relevance_score": 0.91},
            importance=0.8,
        )
        mock_manager.retrieve_memories.return_value = [memory]

        result = await memory_tool.run(
            {
                "action": "search",
                "query": "python memory",
                "memory_type": "semantic",
                "limit": 3,
                "importance_threshold": 0.2,
                "score_threshold": 0.1,
            }
        )

        assert "检索到 1 条记忆" in result
        assert "[semantic] Python memory knowledge" in result
        assert "score=0.910" in result
        mock_manager.retrieve_memories.assert_awaited_once_with(
            query="python memory",
            memory_types=["semantic"],
            limit=3,
            importance_threshold=0.2,
            score_threshold=0.1,
            user_id=None,
            session_id=None,
            start_time=None,
            end_time=None,
        )

    @pytest.mark.asyncio
    async def test_search_retrieve_alias_maps_to_search(
        self, memory_tool, mock_manager
    ):
        await memory_tool.run({"action": "retrieve", "query": "python"})

        mock_manager.retrieve_memories.assert_awaited_once_with(
            query="python",
            memory_types=None,
            limit=5,
            importance_threshold=0.0,
            score_threshold=None,
            user_id=None,
            session_id=None,
            start_time=None,
            end_time=None,
        )

    @pytest.mark.asyncio
    async def test_search_type_alias_maps_to_memory_type(
        self, memory_tool, mock_manager
    ):
        await memory_tool.run(
            {"action": "search", "query": "python", "type": "episodic"}
        )

        mock_manager.retrieve_memories.assert_awaited_once_with(
            query="python",
            memory_types=["episodic"],
            limit=5,
            importance_threshold=0.0,
            score_threshold=None,
            user_id=None,
            session_id=None,
            start_time=None,
            end_time=None,
        )

    @pytest.mark.asyncio
    async def test_search_rejects_content_instead_of_query(self, memory_tool):
        with pytest.raises(ToolException, match="请使用 query 参数，不要用 content"):
            await memory_tool.run({"action": "search", "content": "python"})

    @pytest.mark.asyncio
    async def test_search_empty_query_raises(self, memory_tool):
        with pytest.raises(ToolException, match="非空 query"):
            await memory_tool.run({"action": "search", "query": ""})

    @pytest.mark.asyncio
    async def test_update_success(self, memory_tool, mock_manager):
        result = await memory_tool.run(
            {
                "action": "update",
                "memory_id": "m1",
                "content": "updated",
                "importance": 0.9,
                "metadata": {"source": "test"},
            }
        )

        assert result == "已更新记忆: m1"
        mock_manager.update_memory.assert_awaited_once_with(
            memory_id="m1",
            content="updated",
            importance=0.9,
            metadata={"source": "test"},
        )

    @pytest.mark.asyncio
    async def test_update_missing_memory_id_raises(self, memory_tool):
        with pytest.raises(ToolException, match="memory_id"):
            await memory_tool.run({"action": "update", "content": "x"})

    @pytest.mark.asyncio
    async def test_remove_success(self, memory_tool, mock_manager):
        result = await memory_tool.run({"action": "remove", "memory_id": "m1"})

        assert result == "已删除记忆: m1"
        mock_manager.remove_memory.assert_awaited_once_with("m1")

    @pytest.mark.asyncio
    async def test_stats_success(self, memory_tool, mock_manager):
        result = await memory_tool.run({"action": "stats"})

        assert "记忆系统统计" in result
        assert "用户: u1" in result
        assert "总记忆数: 3" in result
        assert "- semantic: 1" in result
        mock_manager.get_memory_stats.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_forget_success(self, memory_tool, mock_manager):
        result = await memory_tool.run(
            {
                "action": "forget",
                "strategy": "importance_based",
                "threshold": 0.2,
                "max_age_days": 15,
            }
        )

        assert result == "已遗忘 2 条记忆"
        mock_manager.forget_memories.assert_awaited_once_with(
            strategy="importance_based",
            threshold=0.2,
            max_age_days=15,
        )

    @pytest.mark.asyncio
    async def test_clear_success(self, memory_tool, mock_manager):
        result = await memory_tool.run({"action": "clear"})

        assert result == "已清空所有记忆"
        mock_manager.clear_all_memories.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, memory_tool):
        with pytest.raises(ToolException, match="如需检索记忆，请使用 action=search"):
            await memory_tool.run({"action": "summary"})


class TestMemoryConvenienceFunctions:
    @pytest.mark.asyncio
    async def test_memory_add_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.memory.MemoryTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await memory_add("hello", memory_type="working")

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with(
            {"action": "add", "content": "hello", "memory_type": "working"}
        )

    @pytest.mark.asyncio
    async def test_memory_search_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.memory.MemoryTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await memory_search("python", limit=2)

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with(
            {"action": "search", "query": "python", "limit": 2}
        )

    @pytest.mark.asyncio
    async def test_memory_get_stats_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.memory.MemoryTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await memory_get_stats()

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with({"action": "stats"})
