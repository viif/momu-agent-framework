from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.mcp import MCPToolAdapter
from momu_agent.tools.base import Tool, ToolParameter
from momu_agent.tools.registry import ToolRegistry


class MockTool(Tool):
    """用于测试的 Mock 工具"""

    def __init__(self, name: str = "mock_tool", should_fail: bool = False):
        super().__init__(name=name, description="Mock tool for testing")
        self.should_fail = should_fail

    async def run(self, parameters: dict[str, Any]) -> str:
        if self.should_fail:
            raise ToolException("Mock tool execution failed")
        return f"MockTool result: {parameters}"

    def get_parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="input", type="string", description="Input text")]


class TestToolRegistry:
    """测试 ToolRegistry 类"""

    @pytest.fixture
    def registry(self):
        """创建一个干净的注册表实例"""
        return ToolRegistry()

    def test_register_tool(self, registry):
        """测试注册 Tool 对象"""
        tool = MockTool()
        registry.register_tool(tool)

        assert "mock_tool" in registry.list_tools()
        assert registry.get_tool("mock_tool") == tool

    def test_register_function(self, registry):
        """测试注册函数工具"""

        def my_func(text):
            return f"Result: {text}"

        registry.register_function("my_func", "A test function", my_func)

        assert "my_func" in registry.list_tools()
        assert registry.get_function("my_func") == my_func

    def test_register_duplicate_tool(self, registry):
        """测试注册同名工具时的警告"""
        tool1 = MockTool()
        tool2 = MockTool()

        registry.register_tool(tool1)
        with patch.object(registry.logger, "warning") as mock_log:
            registry.register_tool(tool2)
            mock_log.assert_called_once()
            assert registry.get_tool("mock_tool") == tool2

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, registry):
        """测试执行 Tool 对象成功"""
        tool = MockTool()
        registry.register_tool(tool)

        result = await registry.execute_tool("mock_tool", "hello")
        assert "MockTool result" in result

    @pytest.mark.asyncio
    async def test_execute_tool_with_dict_params(self, registry):
        """测试执行 Tool 对象时传入参数字典"""
        tool = MockTool()
        registry.register_tool(tool)

        result = await registry.execute_tool("mock_tool", {"query": "hello"})
        assert "MockTool result" in result
        assert "query" in result

    @pytest.mark.asyncio
    async def test_execute_function_success(self, registry):
        """测试执行函数工具成功"""

        def my_func(text):
            return f"Processed: {text}"

        registry.register_function("my_func", "desc", my_func)

        result = await registry.execute_tool("my_func", "hello")
        assert result == "Processed: hello"

    @pytest.mark.asyncio
    async def test_execute_async_function_success(self, registry):
        """测试执行异步函数工具成功"""

        async def my_async_func(text):
            return f"Processed async: {text}"

        registry.register_function("my_async_func", "desc", my_async_func)

        result = await registry.execute_tool("my_async_func", "hello")
        assert result == "Processed async: hello"

    @pytest.mark.asyncio
    async def test_execute_function_with_dict_input(self, registry):
        """测试函数工具接收字典输入并自动提取 input"""

        def my_func(text):
            return f"Processed: {text}"

        registry.register_function("my_func", "desc", my_func)

        result = await registry.execute_tool("my_func", {"input": "hello"})
        assert result == "Processed: hello"

    @pytest.mark.asyncio
    async def test_execute_function_with_dict_query(self, registry):
        """测试函数工具接收单键字典输入并提取该值"""

        def my_func(text):
            return f"Processed: {text}"

        registry.register_function("my_func", "desc", my_func)

        result = await registry.execute_tool("my_func", {"query": "hello"})
        assert result == "Processed: hello"

    @pytest.mark.asyncio
    async def test_execute_function_with_dict_expression(self, registry):
        """测试函数工具接收单键字典输入并提取该值"""

        def my_func(text):
            return f"Processed: {text}"

        registry.register_function("my_func", "desc", my_func)

        result = await registry.execute_tool("my_func", {"expression": "1+1"})
        assert result == "Processed: 1+1"

    @pytest.mark.asyncio
    async def test_execute_function_with_invalid_dict(self, registry):
        """测试函数工具接收多键字典输入时抛出异常"""

        def my_func(text):
            return f"Processed: {text}"

        registry.register_function("my_func", "desc", my_func)

        with pytest.raises(ToolException, match="仅包含一个参数值的字典输入"):
            await registry.execute_tool("my_func", {"foo": "bar", "baz": "qux"})

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, registry):
        """测试执行不存在的工具"""
        with pytest.raises(ToolException, match="未找到名为"):
            await registry.execute_tool("non_existent", "hello")

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, registry):
        """测试执行 Tool 抛出 ToolException"""
        tool = MockTool(should_fail=True)
        registry.register_tool(tool)

        with pytest.raises(ToolException, match="Mock tool execution failed"):
            await registry.execute_tool("mock_tool", "hello")

    @pytest.mark.asyncio
    async def test_execute_function_exception(self, registry):
        """测试执行函数抛出普通异常"""

        def bad_func(text):
            raise ValueError("Something went wrong")

        registry.register_function("bad_func", "desc", bad_func)

        with patch.object(registry.logger, "exception") as mock_log:
            with pytest.raises(
                ToolException, match="执行工具 'bad_func' 时发生未知异常"
            ):
                await registry.execute_tool("bad_func", "hello")
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_mcp_client(self, registry):
        class MockMCPClient:
            def __init__(self):
                self.close = AsyncMock(return_value=None)

            async def list_tools(self):
                return [
                    MCPToolAdapter(
                        client=self,
                        remote_name="list_files",
                        description="列出文件",
                        input_schema={},
                    )
                ]

        client = MockMCPClient()
        registered = await registry.register_mcp_client("filesystem", client)

        assert registered == ["filesystem.list_files"]
        assert registry.get_tool("filesystem.list_files") is not None
        assert registry._managed_mcp_clients == [client]

    @pytest.mark.asyncio
    async def test_register_mcp_client_name_conflict(self, registry):
        registry.register_tool(MockTool(name="filesystem.list_files"))

        class MockMCPClient:
            async def list_tools(self):
                return [
                    MCPToolAdapter(
                        client=self,
                        remote_name="list_files",
                        description="列出文件",
                        input_schema={},
                    )
                ]

            async def close(self):
                return None

        with pytest.raises(ToolException, match="已存在"):
            await registry.register_mcp_client("filesystem", MockMCPClient())

    @pytest.mark.asyncio
    async def test_close_calls_all_tool_close(self, registry):
        tool_a = MockTool(name="a")
        tool_b = MockTool(name="b")
        tool_a.close = AsyncMock(return_value=None)
        tool_b.close = AsyncMock(return_value=None)
        registry.register_tool(tool_a)
        registry.register_tool(tool_b)

        await registry.close()

        tool_a.close.assert_awaited_once_with()
        tool_b.close.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_close_continues_on_tool_error(self, registry):
        tool_a = MockTool(name="a")
        tool_b = MockTool(name="b")
        tool_a.close = AsyncMock(side_effect=RuntimeError("boom"))
        tool_b.close = AsyncMock(return_value=None)
        registry.register_tool(tool_a)
        registry.register_tool(tool_b)

        with patch.object(registry.logger, "warning") as mock_warning:
            await registry.close()

        tool_a.close.assert_awaited_once_with()
        tool_b.close.assert_awaited_once_with()
        mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_managed_mcp_client_once(self, registry):
        class MockMCPClient:
            def __init__(self):
                self.close = AsyncMock(return_value=None)

            async def list_tools(self):
                return [
                    MCPToolAdapter(
                        client=self,
                        remote_name="list_files",
                        description="列出文件",
                        input_schema={},
                    ),
                    MCPToolAdapter(
                        client=self,
                        remote_name="read_file",
                        description="读取文件",
                        input_schema={},
                    ),
                ]

        client = MockMCPClient()
        await registry.register_mcp_client("filesystem", client)

        await registry.close()

        client.close.assert_awaited_once_with()

    def test_get_tools_description(self, registry):
        """测试获取工具描述字符串"""
        tool = MockTool()
        registry.register_tool(tool)
        registry.register_function("func", "Function Desc", lambda x: x)

        desc = registry.get_tools_description()

        assert "- mock_tool: Mock tool for testing" in desc
        assert "- func: Function Desc" in desc

    def test_get_tools_description_empty(self, registry):
        """测试空注册表的描述"""
        desc = registry.get_tools_description()
        assert desc == "暂无可用工具"

    def test_list_tools(self, registry):
        """测试列出所有工具名称"""
        registry.register_tool(MockTool())
        registry.register_function("func", "desc", lambda x: x)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert "mock_tool" in tools
        assert "func" in tools

    def test_unregister_tool(self, registry):
        """测试注销工具"""
        registry.register_tool(MockTool())
        assert "mock_tool" in registry.list_tools()

        registry.unregister("mock_tool")
        assert "mock_tool" not in registry.list_tools()

    def test_unregister_function(self, registry):
        """测试注销函数"""
        registry.register_function("func", "desc", lambda x: x)

        registry.unregister("func")
        assert "func" not in registry.list_tools()

    def test_unregister_not_found(self, registry):
        """测试注销不存在的工具"""
        with patch.object(registry.logger, "warning") as mock_log:
            registry.unregister("not_found")
            mock_log.assert_called_once()

    def test_clear(self, registry):
        """测试清空所有工具"""
        registry.register_tool(MockTool())
        registry.register_function("func", "desc", lambda x: x)

        registry.clear()

        assert len(registry.list_tools()) == 0
        assert registry._managed_mcp_clients == []
