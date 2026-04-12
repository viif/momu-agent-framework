from typing import Any
from unittest.mock import patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.base import Tool, ToolParameter
from momu_agent.tools.registry import ToolRegistry


class MockTool(Tool):
    """用于测试的 Mock 工具"""

    def __init__(self, name: str = "mock_tool", should_fail: bool = False):
        super().__init__(name=name, description="Mock tool for testing")
        self.should_fail = should_fail

    def run(self, parameters: dict[str, Any]) -> str:
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

    def test_execute_tool_success(self, registry):
        """测试执行 Tool 对象成功"""
        tool = MockTool()
        registry.register_tool(tool)

        result = registry.execute_tool("mock_tool", "hello")
        assert "MockTool result" in result

    def test_execute_function_success(self, registry):
        """测试执行函数工具成功"""

        def my_func(text):
            return f"Processed: {text}"

        registry.register_function("my_func", "desc", my_func)

        result = registry.execute_tool("my_func", "hello")
        assert result == "Processed: hello"

    def test_execute_tool_not_found(self, registry):
        """测试执行不存在的工具"""
        result = registry.execute_tool("non_existent", "hello")
        assert result.startswith("错误：未找到名为")

    def test_execute_tool_exception(self, registry):
        """测试执行 Tool 抛出 ToolException"""
        tool = MockTool(should_fail=True)
        registry.register_tool(tool)

        with patch.object(registry.logger, "error") as mock_log:
            result = registry.execute_tool("mock_tool", "hello")
            mock_log.assert_called_once()
            assert result.startswith("错误：执行工具")

    def test_execute_function_exception(self, registry):
        """测试执行函数抛出普通异常"""

        def bad_func(text):
            raise ValueError("Something went wrong")

        registry.register_function("bad_func", "desc", bad_func)

        with patch.object(registry.logger, "exception") as mock_log:
            result = registry.execute_tool("bad_func", "hello")
            mock_log.assert_called_once()
            assert result.startswith("错误：执行工具")

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
