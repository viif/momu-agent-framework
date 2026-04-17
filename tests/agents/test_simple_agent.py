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

    async def run(self, parameters):
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
    registry.get_function = Mock(return_value=None)

    return registry


async def test_agent_no_tool_call(mock_llm):
    """测试没有工具时的普通对话"""
    # Arrange
    mock_llm.invoke.return_value = "这是最终答案。"

    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=None)

    # Act
    response = await agent.run("你好，世界！")

    # Assert
    assert response == "这是最终答案。"
    assert len(agent._history) == 2  # 用户 + 助手


@patch("momu_agent.agents.simple_agent.run_parallel_tools")
async def test_agent_with_tool_call(mock_run_parallel, mock_llm, mock_registry):
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
    response = await agent.run("查询上海天气")

    # Assert
    assert "上海今天晴天" in response
    assert mock_llm.invoke.call_count == 2  # 调用了两次 LLM (工具前 + 工具后)
    assert mock_run_parallel.call_count == 1  # 工具执行了一次


@patch("momu_agent.agents.simple_agent.run_parallel_tools")
async def test_agent_max_iterations_stop(mock_run_parallel, mock_llm, mock_registry):
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
    response = await agent.run("强制循环测试", max_tool_iterations=2)

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


@patch("momu_agent.agents.simple_agent.run_parallel_tools")
async def test_agent_parallel_tool_calls(mock_run_parallel, mock_llm, mock_registry):
    """
    测试 LLM 一次性输出多个工具调用时，全部并发执行
    """
    # Arrange
    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=mock_registry)

    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:search:{"query": "北京天气"}][TOOL_CALL:search:{"query": "上海天气"}]',
        "北京晴，上海雨。",
    ]
    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "search",
            "input_data": {},
            "result": "北京晴",
            "status": "success",
        },
        {
            "task_id": 1,
            "tool_name": "search",
            "input_data": {},
            "result": "上海雨",
            "status": "success",
        },
    ]

    # Act
    response = await agent.run("查询北京和上海天气")

    # Assert
    assert "北京晴" in response
    assert "上海雨" in response
    # 验证 run_parallel_tools 收到了两个任务
    call_args = mock_run_parallel.call_args
    tasks = call_args.kwargs["tasks"]
    assert len(tasks) == 2


@patch("momu_agent.agents.simple_agent.run_parallel_tools")
async def test_agent_tool_execution_error(mock_run_parallel, mock_llm, mock_registry):
    """
    测试工具执行失败时，错误信息以 tool 消息传给 LLM，LLM 给出最终回答
    """
    # Arrange
    agent = SimpleAgent(name="TestAgent", llm=mock_llm, tool_registry=mock_registry)

    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:search:{"query": "上海天气"}]',
        "工具执行失败，无法获取天气信息。",
    ]
    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "search",
            "input_data": {},
            "result": "连接超时",
            "status": "error",
        }
    ]

    # Act
    response = await agent.run("上海天气？")

    # Assert
    assert "无法获取天气" in response
    assert mock_llm.invoke.call_count == 2
    # 验证 LLM 第二次调用时收到了包含错误信息的 tool 消息
    second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "执行失败" in tool_messages[0]["content"]
    assert "连接超时" in tool_messages[0]["content"]


async def test_agent_history_accumulates_across_runs(mock_llm):
    """
    测试多次调用 run() 后，历史记录正确累积（无工具模式）
    """
    # Arrange
    mock_llm.invoke.return_value = "好的。"
    agent = SimpleAgent(name="TestAgent", llm=mock_llm)

    # Act & Assert
    await agent.run("第一条消息")
    assert len(agent._history) == 2  # user + assistant

    await agent.run("第二条消息")
    assert len(agent._history) == 4  # 上一轮 2 条 + 本轮 2 条


def test_agent_custom_system_prompt(mock_llm, mock_registry):
    """
    测试自定义 system_prompt 在有工具时被正确保留在增强提示词中
    """
    agent = SimpleAgent(
        name="TestAgent",
        llm=mock_llm,
        system_prompt="你是专业气象顾问。",
        tool_registry=mock_registry,
    )
    prompt = agent._get_enhanced_system_prompt()

    assert "你是专业气象顾问。" in prompt
    assert "search: 搜索工具" in prompt
    assert "### 工具调用协议 ###" in prompt
