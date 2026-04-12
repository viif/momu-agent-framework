from unittest.mock import AsyncMock, Mock, patch

import pytest

from momu_agent.agents.simple_agent import SimpleAgent
from momu_agent.tools.base import ToolParameter
from momu_agent.tools.registry import ToolRegistry


class MockSearchTool:
    """
    模拟工具对象
    严格遵循 Tool 基类的接口定义
    """

    def __init__(self):
        self.name = "search"
        self.description = "模拟搜索工具"

    def get_parameters(self):
        return [
            ToolParameter(
                name="query", type="string", description="搜索关键词", required=True
            )
        ]

    def run(self, parameters):
        """
        执行工具
        Args:
            parameters (Dict): 参数字典，例如 {"query": "上海天气"}
        """
        query = parameters.get("query", "未知")
        return f"搜索结果：关于 {query} 的信息"


@pytest.fixture
def mock_llm():
    """创建一个模拟的 LLM"""
    llm = Mock()
    llm.invoke = AsyncMock()
    return llm


@pytest.fixture
def mock_registry():
    """修正后的 mock_registry"""
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_description = Mock(return_value="search: 搜索工具")

    registry.get_tool = Mock(
        side_effect=lambda name: MockSearchTool() if name == "search" else None
    )

    return registry


def test_agent_no_tool_call(mock_llm):
    """测试没有工具时的普通对话"""
    # Arrange
    mock_llm.invoke.return_value = "这是最终答案。"

    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=None)

    # Act
    response = agent.run("你好，世界！")

    # Assert
    assert response == "这是最终答案。"
    assert len(agent._history) == 2  # 用户 + 助手


@patch("momu_agent.agents.simple_agent.run_parallel_tools")
def test_agent_with_tool_call(mock_run_parallel, mock_llm, mock_registry):
    """
    测试包含工具调用的场景
    模拟 LLM 先输出工具调用，然后输出最终结果
    """
    # Arrange
    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=mock_registry)

    # 模拟 LLM 的调用行为：第一次返回工具指令，第二次返回最终答案
    mock_llm.invoke.side_effect = [
        '`[TOOL_CALL:search:{"query": "上海天气"}]`',  # 第一轮输出
        "上海今天晴天。",  # 第二轮输出（工具执行后）
    ]

    # 模拟工具执行结果
    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "search",
            "input_data": '{"query": "上海天气"}',
            "result": "搜索结果：上海今天晴天",
            "status": "success",
        }
    ]

    # Act
    response = agent.run("查询上海天气")

    # Assert
    assert "上海今天晴天" in response
    assert mock_llm.invoke.call_count == 2  # 调用了两次 LLM (工具前 + 工具后)
    assert mock_run_parallel.call_count == 1  # 工具执行了一次


@patch("momu_agent.agents.simple_agent.run_parallel_tools")
def test_agent_max_iterations_stop(mock_run_parallel, mock_llm, mock_registry):
    """
    测试达到最大迭代次数时，Agent 停止运行并返回提示信息
    """
    # Arrange
    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=mock_registry)

    # 模拟 LLM 一直输出工具调用
    mock_llm.invoke.return_value = '[TOOL_CALL:search:{"query": "loop"}]'
    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "search",
            "input_data": '{"query": "loop"}',
            "result": "Result",
            "status": "success",
        }
    ]

    # Act
    # 设置最大迭代次数为 2
    response = agent.run("强制循环测试", max_tool_iterations=2)

    # Assert
    # 1. 验证 LLM 调用次数严格等于 2
    assert mock_llm.invoke.call_count == 2

    # 2. 验证返回值包含提示信息，而不是工具调用标记
    assert "已达到最大工具调用次数限制" in response
    assert "[TOOL_CALL" not in response


def test_enhanced_system_prompt(mock_llm, mock_registry):
    """测试系统提示词是否包含了工具协议"""
    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=mock_registry)
    prompt = agent._get_enhanced_system_prompt()

    assert "### 工具调用协议 ###" in prompt
    assert "search: 搜索工具" in prompt
    assert "你是一个有用的AI助手。" in prompt
