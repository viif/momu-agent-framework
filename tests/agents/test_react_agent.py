from unittest.mock import AsyncMock, Mock, patch

import pytest

from momu_agent.agents.react_agent import ReActAgent
from momu_agent.tools.base import ToolParameter
from momu_agent.tools.registry import ToolRegistry


class MockSearchTool:
    """
    模拟搜索工具
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
        query = parameters.get("query", "未知")
        return f"搜索结果：关于 {query} 的信息"


class MockCalculatorTool:
    """
    模拟计算器工具
    """

    def __init__(self):
        self.name = "calculator"
        self.description = "模拟计算器"

    def get_parameters(self):
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式",
                required=True,
            )
        ]

    def run(self, parameters):
        return "计算结果：150"


@pytest.fixture
def mock_llm():
    """创建一个模拟的 LLM"""
    llm = Mock()
    llm.invoke = AsyncMock()
    return llm


@pytest.fixture
def mock_registry():
    """创建一个模拟的工具注册表"""
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_description = Mock(
        return_value="search: 搜索工具; calculator: 计算器"
    )

    def get_tool_side_effect(name):
        if name == "search":
            return MockSearchTool()
        elif name == "calculator":
            return MockCalculatorTool()
        return None

    registry.get_tool = Mock(side_effect=get_tool_side_effect)
    return registry


@patch("momu_agent.agents.react_agent.run_parallel_tools")
async def test_react_single_step(mock_run_parallel, mock_llm, mock_registry):
    """
    测试：单步 ReAct 流程 (思考 -> 工具 -> 观察 -> 完成)
    """
    # Arrange
    agent = ReActAgent(
        name="TestReAct", llm=mock_llm, tool_registry=mock_registry, max_steps=5
    )

    # 模拟 LLM 的两次响应
    # 1. 思考 + 工具调用
    response_step_1 = """Thought: 我需要查询上海的天气。
Action: [TOOL_CALL:search:{"query": "上海天气"}]"""

    # 2. 思考 + 完成任务
    response_step_2 = """Thought: 我已经查到了天气信息。
Action: Finish[上海今天晴天]"""

    mock_llm.invoke.side_effect = [response_step_1, response_step_2]

    # 模拟工具执行结果
    mock_run_parallel.return_value = [
        {"status": "success", "result": "晴天，25度", "tool_name": "search"}
    ]

    # Act
    response = await agent.run("上海天气怎么样？")

    # Assert
    assert "上海今天晴天" in response
    assert mock_llm.invoke.call_count == 2  # 调用了两次 LLM
    assert mock_run_parallel.call_count == 1  # 执行了一次工具

    # 验证历史记录中包含了 Observation
    # ReAct 的历史记录格式通常包含 Thought, Action, Observation
    assert len(agent._history) == 2  # 用户输入 + 最终回答


@patch("momu_agent.agents.react_agent.run_parallel_tools")
async def test_react_parallel_calls(mock_run_parallel, mock_llm, mock_registry):
    """
    测试：一次 Action 中调用多个工具
    """
    # Arrange
    agent = ReActAgent(
        name="ParallelAgent", llm=mock_llm, tool_registry=mock_registry, max_steps=5
    )

    # LLM 一次性输出两个工具调用
    response_step_1 = """Thought: 我需要同时查询北京和上海的天气。
Action: 
[TOOL_CALL:search:{"query": "北京天气"}]
[TOOL_CALL:search:{"query": "上海天气"}]"""

    response_step_2 = """Thought: 查到了。
Action: Finish[北京晴，上海雨]"""

    mock_llm.invoke.side_effect = [response_step_1, response_step_2]

    # 模拟并行执行返回两个结果
    mock_run_parallel.return_value = [
        {"status": "success", "result": "北京晴", "tool_name": "search"},
        {"status": "success", "result": "上海雨", "tool_name": "search"},
    ]

    # Act
    response = await agent.run("北京上海天气")

    # Assert
    assert "北京晴" in response
    assert "上海雨" in response
    assert mock_run_parallel.call_count == 1
    call_args = mock_run_parallel.call_args
    tasks = call_args.kwargs["tasks"]
    assert len(tasks) == 2


@patch("momu_agent.agents.react_agent.run_parallel_tools")
async def test_react_max_steps_limit(mock_run_parallel, mock_llm, mock_registry):
    """
    测试：达到最大步数强制停止
    """
    # Arrange
    # 设置 max_steps=2
    agent = ReActAgent(
        name="LimitAgent", llm=mock_llm, tool_registry=mock_registry, max_steps=2
    )

    # LLM 一直循环调用工具，不调用 Finish
    response_loop = """Thought: 还在思考...
Action: [TOOL_CALL:search:{"query": "loop"}]"""

    mock_llm.invoke.return_value = response_loop
    mock_run_parallel.return_value = [
        {"status": "success", "result": "Result", "tool_name": "search"}
    ]

    # Act
    response = await agent.run("死循环测试")

    # Assert
    assert mock_llm.invoke.call_count == 2  # 严格等于 max_steps
    assert "已达到最大步数" in response
    assert "Finish" not in response
