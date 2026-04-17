import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.executor import (
    ToolExecutor,
    run_batch_tool,
    run_parallel_tools,
)


class TestToolExecutor:
    """测试 ToolExecutor 类"""

    def setup_method(self):
        """每个测试前的设置"""
        self.mock_registry = Mock()
        self.mock_registry.execute_tool = AsyncMock(return_value="mock_result")

    def test_init(self):
        """测试初始化"""
        executor = ToolExecutor(self.mock_registry, default_timeout=10.0)
        assert executor.default_timeout == 10.0

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """测试单个工具异步执行成功"""
        executor = ToolExecutor(self.mock_registry, default_timeout=5.0)

        result = await executor.execute_tool("search", "query")

        assert result == "mock_result"
        self.mock_registry.execute_tool.assert_awaited_once_with("search", "query")

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self):
        """测试单个工具执行超时"""

        async def slow_task(*args):
            await asyncio.sleep(2)
            return "result"

        self.mock_registry.execute_tool = AsyncMock(side_effect=slow_task)
        executor = ToolExecutor(self.mock_registry, default_timeout=0.1)

        with pytest.raises(ToolException) as exc_info:
            await executor.execute_tool("slow_tool", "query")

        assert "超时" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        """测试工具执行抛出普通异常"""
        self.mock_registry.execute_tool = AsyncMock(side_effect=Exception("模拟错误"))
        executor = ToolExecutor(self.mock_registry)

        with pytest.raises(ToolException) as exc_info:
            await executor.execute_tool("error_tool", "query")

        assert "模拟错误" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self):
        """测试并行执行多个工具"""
        tasks = [
            {"tool_name": "search", "input_data": "q1"},
            {"tool_name": "calc", "input_data": "1+1"},
            {"tool_name": "weather", "input_data": "bj"},
        ]

        result_map = {
            ("search", "q1"): "res1",
            ("calc", "1+1"): "res2",
            ("weather", "bj"): "res3",
        }

        async def execute_tool(name, data):
            return result_map[(name, data)]

        self.mock_registry.execute_tool = AsyncMock(side_effect=execute_tool)

        executor = ToolExecutor(self.mock_registry)
        results = await executor.execute_tools_parallel(tasks)

        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[0]["result"] == "res1"
        assert results[1]["tool_name"] == "calc"

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_with_preparation_error(self):
        """测试任务预处理失败时短路为 error，不触发 execute_tool 调用"""
        tasks = [
            {"tool_name": "search", "input_data": "q1"},
            {"tool_name": "bad_tool", "error": "工具未注册"},
        ]

        self.mock_registry.execute_tool = AsyncMock(return_value="ok")
        executor = ToolExecutor(self.mock_registry)

        results = await executor.execute_tools_parallel(tasks)

        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert results[1]["error_type"] == "TaskPreparationError"
        assert "工具未注册" in results[1]["result"]
        self.mock_registry.execute_tool.assert_awaited_once_with("search", "q1")

    @pytest.mark.asyncio
    async def test_execute_empty_tasks(self):
        """测试空任务列表"""
        executor = ToolExecutor(self.mock_registry)
        results = await executor.execute_tools_parallel([])
        assert results == []


class TestConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.asyncio
    async def test_run_parallel_tools(self):
        """测试 run_parallel_tools 函数"""
        mock_registry = Mock()
        result_map = {
            ("search", "query_A"): "data_A",
            ("search", "query_B"): "data_B",
            ("calculator", "1+1"): "data_C",
        }

        async def execute_tool(name, data):
            return result_map[(name, data)]

        mock_registry.execute_tool = AsyncMock(side_effect=execute_tool)

        tasks = [
            {"tool_name": "search", "input_data": "query_A"},
            {"tool_name": "search", "input_data": "query_B"},
            {"tool_name": "calculator", "input_data": "1+1"},
        ]

        results = await run_parallel_tools(mock_registry, tasks, timeout=5.0)

        assert len(results) == 3
        assert results[0]["result"] == "data_A"
        assert results[0]["tool_name"] == "search"
        assert results[1]["result"] == "data_B"
        assert results[1]["tool_name"] == "search"
        assert results[2]["result"] == "data_C"
        assert results[2]["tool_name"] == "calculator"
        assert mock_registry.execute_tool.await_count == 3

    @pytest.mark.asyncio
    async def test_run_batch_tool_async(self):
        """测试异步 run_batch_tool 函数"""
        mock_registry = Mock()
        result_map = {
            ("search", "query_A"): "result_1",
            ("search", "query_B"): "result_2",
        }

        async def execute_tool(name, data):
            return result_map[(name, data)]

        mock_registry.execute_tool = AsyncMock(side_effect=execute_tool)

        results = await run_batch_tool(
            mock_registry,
            tool_name="search",
            input_list=["query_A", "query_B"],
            timeout=10.0,
        )

        assert len(results) == 2
        assert results[0]["result"] == "result_1"
        assert results[1]["result"] == "result_2"
        assert results[0]["status"] == "success"
        assert mock_registry.execute_tool.await_count == 2
