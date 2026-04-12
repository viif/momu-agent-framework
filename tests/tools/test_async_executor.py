import time
from unittest.mock import Mock

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.async_executor import (
    AsyncToolExecutor,
    run_batch_tool,
    run_batch_tool_sync,
    run_parallel_tools,
    run_parallel_tools_sync,
)


class TestAsyncToolExecutor:
    """测试 AsyncToolExecutor 类"""

    def setup_method(self):
        """每个测试前的设置"""
        self.mock_registry = Mock()
        self.mock_registry.execute_tool = Mock(return_value="mock_result")

    def test_init(self):
        """测试初始化"""
        executor = AsyncToolExecutor(
            self.mock_registry, max_workers=2, default_timeout=10.0
        )
        assert executor.default_timeout == 10.0
        executor.close()

    @pytest.mark.asyncio
    async def test_execute_tool_async_success(self):
        """测试单个工具异步执行成功"""
        executor = AsyncToolExecutor(self.mock_registry, default_timeout=5.0)

        result = await executor.execute_tool_async("search", "query")

        assert result == "mock_result"
        self.mock_registry.execute_tool.assert_called_once_with("search", "query")
        executor.close()

    @pytest.mark.asyncio
    async def test_execute_tool_async_timeout(self):
        """测试单个工具执行超时"""

        def slow_task(*args):
            time.sleep(2)  # 模拟耗时操作
            return "result"

        self.mock_registry.execute_tool = Mock(side_effect=slow_task)
        executor = AsyncToolExecutor(
            self.mock_registry, default_timeout=0.1
        )  # 设置极短的超时

        with pytest.raises(ToolException) as exc_info:
            await executor.execute_tool_async("slow_tool", "query")

        assert "超时" in str(exc_info.value)
        executor.close()

    @pytest.mark.asyncio
    async def test_execute_tool_async_exception(self):
        """测试工具执行抛出普通异常"""
        self.mock_registry.execute_tool = Mock(side_effect=Exception("模拟错误"))
        executor = AsyncToolExecutor(self.mock_registry)

        with pytest.raises(ToolException) as exc_info:
            await executor.execute_tool_async("error_tool", "query")

        assert "模拟错误" in str(exc_info.value)
        executor.close()

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self):
        """测试并行执行多个工具"""
        tasks = [
            {"tool_name": "search", "input_data": "q1"},
            {"tool_name": "calc", "input_data": "1+1"},
            {"tool_name": "weather", "input_data": "bj"},
        ]

        self.mock_registry.execute_tool.side_effect = ["res1", "res2", "res3"]

        executor = AsyncToolExecutor(self.mock_registry)
        results = await executor.execute_tools_parallel(tasks)

        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[0]["result"] == "res1"
        assert results[1]["tool_name"] == "calc"
        executor.close()

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_with_error(self):
        """测试并行执行中包含失败的任务"""
        tasks = [
            {"tool_name": "good", "input_data": "ok"},
            {"tool_name": "bad", "input_data": "err"},
        ]

        def side_effect(name, data):
            if name == "bad":
                raise Exception("工具报错")
            return "ok"

        self.mock_registry.execute_tool.side_effect = side_effect
        executor = AsyncToolExecutor(self.mock_registry)

        results = await executor.execute_tools_parallel(tasks)

        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert "工具报错" in results[1]["result"]
        executor.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """测试上下文管理器"""
        executor = AsyncToolExecutor(self.mock_registry)

        with executor:
            pass

        async with executor:
            pass

    @pytest.mark.asyncio
    async def test_execute_empty_tasks(self):
        """测试空任务列表"""
        executor = AsyncToolExecutor(self.mock_registry)
        results = await executor.execute_tools_parallel([])
        assert results == []
        executor.close()


# 便捷函数的测试
class TestConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.asyncio
    async def test_run_parallel_tools(self):
        """测试 run_parallel_tools 函数"""
        mock_registry = Mock()
        mock_registry.execute_tool.side_effect = ["data_A", "data_B", "data_C"]

        tasks = [
            {"tool_name": "search", "input_data": "query_A"},
            {"tool_name": "search", "input_data": "query_B"},
            {"tool_name": "calculator", "input_data": "1+1"},
        ]

        results = await run_parallel_tools(
            mock_registry, tasks, max_workers=2, timeout=5.0
        )

        assert len(results) == 3

        assert results[0]["result"] == "data_A"
        assert results[0]["tool_name"] == "search"

        assert results[1]["result"] == "data_B"
        assert results[1]["tool_name"] == "search"

        assert results[2]["result"] == "data_C"
        assert results[2]["tool_name"] == "calculator"

        assert mock_registry.execute_tool.call_count == 3

    def test_run_parallel_tools_sync(self):
        """测试同步包装函数"""
        mock_registry = Mock()
        mock_registry.execute_tool.side_effect = ["sync_A", "sync_B"]

        tasks = [
            {"tool_name": "test", "input_data": "input_1"},
            {"tool_name": "test", "input_data": "input_2"},
        ]

        results = run_parallel_tools_sync(mock_registry, tasks)

        assert len(results) == 2

        assert results[0]["result"] == "sync_A"
        assert results[1]["result"] == "sync_B"

        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_run_batch_tool_async(self):
        """测试异步 run_batch_tool 函数"""
        mock_registry = Mock()
        mock_registry.execute_tool.side_effect = ["result_1", "result_2"]

        results = await run_batch_tool(
            mock_registry,
            tool_name="search",
            input_list=["query_A", "query_B"],
            max_workers=2,
            timeout=10.0,
        )

        assert len(results) == 2
        assert results[0]["result"] == "result_1"
        assert results[1]["result"] == "result_2"
        assert results[0]["status"] == "success"

        assert mock_registry.execute_tool.call_count == 2

    def test_run_batch_tool_sync(self):
        """测试同步包装函数 run_batch_tool_sync"""
        mock_registry = Mock()
        mock_registry.execute_tool.return_value = "sync_batch_result"

        results = run_batch_tool_sync(
            mock_registry, tool_name="calculator", input_list=["1+1"], max_workers=4
        )

        assert len(results) == 1
        assert results[0]["result"] == "sync_batch_result"
        assert results[0]["tool_name"] == "calculator"

    def test_run_parallel_tools_sync_empty(self):
        """测试同步函数处理空任务列表"""
        mock_registry = Mock()
        tasks = []

        results = run_parallel_tools_sync(mock_registry, tasks)

        assert results == []
        mock_registry.execute_tool.assert_not_called()
