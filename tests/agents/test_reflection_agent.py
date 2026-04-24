from unittest.mock import AsyncMock, Mock, patch

import pytest

from momu_agent.agents.reflection_agent import Memory, ReflectionAgent
from momu_agent.core.exceptions import AgentException
from momu_agent.tools.base import ToolParameter
from momu_agent.tools.registry import ToolRegistry


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.model = "mock-model"
    llm.invoke = AsyncMock()
    return llm


# ── Memory 单元测试 ──────────────────────────────────────────────────────────


def test_memory_add_and_get_last_execution():
    mem = Memory()
    mem.add_record("execution", "第一版答案")
    mem.add_record("reflection", "有些地方可以改进")
    mem.add_record("execution", "改进后的答案")

    assert mem.get_last_execution() == "改进后的答案"


def test_memory_get_last_execution_empty():
    mem = Memory()
    assert mem.get_last_execution() == ""


def test_memory_get_trajectory_format():
    mem = Memory()
    mem.add_record("execution", "答案A")
    mem.add_record("reflection", "反馈B")
    trajectory = mem.get_trajectory()

    assert "答案A" in trajectory
    assert "反馈B" in trajectory
    assert "上一轮回答" in trajectory
    assert "反思反馈" in trajectory


# ── ReflectionAgent 成功路径 ─────────────────────────────────────────────────


async def test_initial_generation_then_early_exit(mock_llm):
    """
    反思反馈包含"无需改进"时，仅调用两次 LLM（初始 + 反思），不调用优化
    """
    agent = ReflectionAgent(name="TestAgent", llm=mock_llm, max_iterations=3)

    mock_llm.invoke.side_effect = [
        "初始回答",
        "无需改进，答案很完整",  # 反思 → 触发提前退出
    ]

    result = await agent.run("写一段自我介绍")

    assert result == "初始回答"
    assert mock_llm.invoke.call_count == 2  # 初始 + 反思，无优化
    assert len(agent._history) == 2


async def test_full_iteration_cycle(mock_llm):
    """
    单轮完整循环：初始 → 反思（需改进）→ 优化，然后第二轮反思提前退出
    """
    agent = ReflectionAgent(name="IterAgent", llm=mock_llm, max_iterations=3)

    mock_llm.invoke.side_effect = [
        "初始回答",
        "需要补充更多细节",  # 反思轮1 → 继续
        "改进后的回答",
        "无需改进",  # 反思轮2 → 提前退出
    ]

    result = await agent.run("解释递归算法")

    assert result == "改进后的回答"
    assert mock_llm.invoke.call_count == 4


async def test_max_iterations_exhausted(mock_llm):
    """
    达到 max_iterations 且从未触发提前退出，返回最后一次优化结果
    """
    agent = ReflectionAgent(name="MaxAgent", llm=mock_llm, max_iterations=2)

    mock_llm.invoke.side_effect = [
        "初始回答",  # 初始
        "还可以改进",  # 反思1
        "优化回答v1",  # 优化1
        "仍有改进空间",  # 反思2
        "优化回答v2",  # 优化2
    ]

    result = await agent.run("写一份报告")

    assert result == "优化回答v2"
    # 初始(1) + 反思(2) + 优化(2) = 5 次 LLM 调用
    assert mock_llm.invoke.call_count == 5


async def test_history_stored_after_run(mock_llm):
    """
    运行后 _history 包含用户输入和最终答案
    """
    agent = ReflectionAgent(name="HistoryAgent", llm=mock_llm)

    mock_llm.invoke.side_effect = ["初始回答", "无需改进"]

    await agent.run("测试任务")

    assert len(agent._history) == 2
    assert agent._history[0].role == "user"
    assert agent._history[0].content == "测试任务"
    assert agent._history[1].role == "assistant"
    assert agent._history[1].content == "初始回答"


async def test_memory_reset_between_runs(mock_llm):
    """
    每次调用 run() 都重置 memory，不保留上轮记录
    """
    agent = ReflectionAgent(name="ResetAgent", llm=mock_llm, max_iterations=1)

    mock_llm.invoke.side_effect = [
        "第一次初始",
        "无需改进",
        "第二次初始",
        "无需改进",
    ]

    await agent.run("任务一")
    first_records = len(agent.memory.records)

    await agent.run("任务二")
    second_records = len(agent.memory.records)

    # 每次运行后 memory 仅包含本次的记录（初始 + 反思 = 2）
    assert first_records == 2
    assert second_records == 2


async def test_call_llm_raises_agent_exception_on_empty_response(mock_llm):
    """
    测试：直接调用 _call_llm()，无工具模式下 LLM 空响应时抛出 AgentException
    """
    agent = ReflectionAgent(name="CallFailAgent", llm=mock_llm, max_iterations=1)
    mock_llm.invoke.return_value = ""

    with pytest.raises(AgentException, match="LLM 未返回有效响应"):
        await agent._call_llm("空响应测试")


@patch("momu_agent.agents.reflection_agent.run_parallel_tools")
async def test_invoke_with_tools_raises_on_max_iterations(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    测试：直接调用 _invoke_with_tools()，连续工具调用超限时抛出 AgentException
    """
    agent = ReflectionAgent(
        name="InvokeToolLimitAgent",
        llm=mock_llm,
        tool_registry=mock_registry,
        max_iterations=1,
        max_tool_iterations=2,
    )

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {},
            "result": "42",
            "status": "success",
        }
    ]
    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:calculator:{"expression": "1+1"}]',
        '[TOOL_CALL:calculator:{"expression": "2+2"}]',
    ]
    messages = [
        {"role": "system", "content": "工具模式测试"},
        {"role": "user", "content": "请计算"},
    ]

    with pytest.raises(AgentException, match="已达到最大工具调用次数限制"):
        await agent._invoke_with_tools(messages)


async def test_run_returns_error_when_initial_generation_empty(mock_llm):
    """
    测试：run() 中初始生成为空时，统一包装为执行错误消息并写入历史
    """
    agent = ReflectionAgent(name="RunInitialEmptyAgent", llm=mock_llm)
    mock_llm.invoke.return_value = ""

    result = await agent.run("初始空响应测试")

    assert "执行错误" in result
    assert "LLM 未返回有效响应" in result
    assert len(agent._history) == 2
    assert agent._history[0].role == "user"
    assert agent._history[1].role == "assistant"


async def test_custom_prompts_used(mock_llm):
    """
    custom_prompts 中的模板替换默认模板，LLM 接收到自定义内容
    """
    custom = {
        "initial": "自定义初始: {task}",
        "reflect": "自定义反思: {task} | {content}",
        "refine": "自定义优化: {task} | {last_attempt} | {feedback}",
    }
    agent = ReflectionAgent(
        name="CustomAgent", llm=mock_llm, max_iterations=1, custom_prompts=custom
    )

    mock_llm.invoke.side_effect = ["初始结果", "有改进空间", "优化结果"]

    await agent.run("自定义任务")

    initial_prompt = mock_llm.invoke.call_args_list[0][0][0][0]["content"]
    assert "自定义初始" in initial_prompt
    assert "自定义任务" in initial_prompt

    reflect_prompt = mock_llm.invoke.call_args_list[1][0][0][0]["content"]
    assert "自定义反思" in reflect_prompt


# ── 初始化 ────────────────────────────────────────────────────────────────────


def test_default_initialization(mock_llm):
    from momu_agent.agents.reflection_agent import DEFAULT_PROMPTS

    agent = ReflectionAgent(name="InitAgent", llm=mock_llm)

    assert agent.max_iterations == 3
    assert agent.prompts is DEFAULT_PROMPTS
    assert agent.max_history_length == 100


async def test_english_no_improvement_signal(mock_llm):
    """
    英文停止信号 'no need for improvement' 也能触发提前退出
    """
    agent = ReflectionAgent(name="EnAgent", llm=mock_llm, max_iterations=3)

    mock_llm.invoke.side_effect = [
        "Initial answer",
        "No need for improvement, the answer is comprehensive.",
    ]

    result = await agent.run("Explain recursion")

    assert result == "Initial answer"
    assert mock_llm.invoke.call_count == 2


# ── 工具调用测试 ──────────────────────────────────────────────────────────────


class MockCalculatorTool:
    name = "calculator"
    description = "计算数学表达式"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式",
                required=True,
            )
        ]

    async def run(self, parameters: dict) -> str:
        return str(eval(parameters["expression"]))  # noqa: S307


@pytest.fixture
def mock_registry():
    """创建模拟工具注册表"""
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_description = Mock(
        return_value="calculator: 计算数学表达式\n  - expression (string, 必填): 要计算的数学表达式"
    )
    registry.get_tool = Mock(return_value=MockCalculatorTool())
    registry.get_function = Mock(return_value=None)
    registry.get_all_tools = Mock(return_value=[MockCalculatorTool()])
    return registry


def test_init_with_tool_registry(mock_llm, mock_registry):
    """
    传入 tool_registry 后，agent 持有注册表引用且 max_tool_iterations 默认为 3
    """
    agent = ReflectionAgent(name="ToolAgent", llm=mock_llm, tool_registry=mock_registry)

    assert agent.tool_registry is mock_registry
    assert agent.max_tool_iterations == 3


def test_init_with_custom_max_tool_iterations(mock_llm, mock_registry):
    """
    自定义 max_tool_iterations 应被正确存储
    """
    agent = ReflectionAgent(
        name="ToolAgent",
        llm=mock_llm,
        tool_registry=mock_registry,
        max_tool_iterations=5,
    )

    assert agent.max_tool_iterations == 5


async def test_tool_system_prompt_included_in_messages(mock_llm, mock_registry):
    """
    有 tool_registry 时，LLM 收到的消息首条为 system role，且包含工具描述和调用协议
    """
    agent = ReflectionAgent(
        name="ToolAgent", llm=mock_llm, tool_registry=mock_registry, max_iterations=1
    )

    mock_llm.invoke.side_effect = ["初始回答", "无需改进"]

    await agent.run("测试工具提示词")

    # 初始生成时的消息（第一次 invoke）
    initial_messages = mock_llm.invoke.call_args_list[0][0][0]
    assert initial_messages[0]["role"] == "system"
    assert "calculator" in initial_messages[0]["content"]
    assert "TOOL_CALL" in initial_messages[0]["content"]


async def test_no_tool_registry_sends_user_only_message(mock_llm):
    """
    无 tool_registry 时，_call_llm 只发送单条 user 消息，不含 system role
    """
    agent = ReflectionAgent(name="NoToolAgent", llm=mock_llm, max_iterations=1)

    mock_llm.invoke.side_effect = ["初始回答", "无需改进"]

    await agent.run("无工具测试")

    initial_messages = mock_llm.invoke.call_args_list[0][0][0]
    assert len(initial_messages) == 1
    assert initial_messages[0]["role"] == "user"


@patch("momu_agent.agents.reflection_agent.run_parallel_tools")
async def test_tool_called_during_initial_generation(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    初始生成阶段 LLM 返回工具调用 → run_parallel_tools 被调用 →
    LLM 再次调用返回最终初始答案
    """
    agent = ReflectionAgent(
        name="ToolAgent", llm=mock_llm, tool_registry=mock_registry, max_iterations=1
    )

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {"expression": "2 + 3"},
            "result": "5",
            "status": "success",
        }
    ]
    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:calculator:{"expression": "2 + 3"}]',  # 初始生成：调用工具
        "计算结果是 5",  # 工具结果后的最终答案
        "无需改进",  # 反思
    ]

    result = await agent.run("2 加 3 是多少？")

    assert result == "计算结果是 5"
    assert mock_run_parallel.called
    assert mock_llm.invoke.call_count == 3  # 工具调用 + 继续生成 + 反思


@patch("momu_agent.agents.reflection_agent.run_parallel_tools")
async def test_tool_result_appended_as_tool_message(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    工具执行结果以 tool role 消息传入后续 LLM 调用
    """
    agent = ReflectionAgent(
        name="ToolAgent", llm=mock_llm, tool_registry=mock_registry, max_iterations=1
    )

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {"expression": "10 * 5"},
            "result": "50",
            "status": "success",
        }
    ]
    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:calculator:{"expression": "10 * 5"}]',
        "结果是 50",
        "无需改进",
    ]

    await agent.run("10 乘以 5 是多少？")

    # 工具调用后的继续生成（第二次 invoke）消息中应包含 tool role
    continuation_messages = mock_llm.invoke.call_args_list[1][0][0]
    roles = [m["role"] for m in continuation_messages]
    assert "tool" in roles
    tool_msg = next(m for m in continuation_messages if m["role"] == "tool")
    assert "50" in tool_msg["content"]


@patch("momu_agent.agents.reflection_agent.run_parallel_tools")
async def test_tool_error_formatted_in_message(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    工具执行失败时，错误信息以 ❌ 格式写入 tool 消息，执行继续
    """
    agent = ReflectionAgent(
        name="ToolAgent", llm=mock_llm, tool_registry=mock_registry, max_iterations=1
    )

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {"expression": "1/0"},
            "result": "division by zero",
            "status": "error",
        }
    ]
    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:calculator:{"expression": "1/0"}]',
        "无法计算，发生了除零错误",
        "无需改进",
    ]

    result = await agent.run("计算 1/0")

    continuation_messages = mock_llm.invoke.call_args_list[1][0][0]
    tool_msg = next(m for m in continuation_messages if m["role"] == "tool")
    assert "❌" in tool_msg["content"]
    assert result == "无法计算，发生了除零错误"


@patch("momu_agent.agents.reflection_agent.run_parallel_tools")
async def test_max_tool_iterations_in_single_call(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    单次 _call_llm 内 LLM 持续返回工具调用，达到 max_tool_iterations 后返回警告信息
    """
    agent = ReflectionAgent(
        name="ToolAgent",
        llm=mock_llm,
        tool_registry=mock_registry,
        max_iterations=1,
        max_tool_iterations=2,
    )

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {},
            "result": "42",
            "status": "success",
        }
    ]
    # 初始生成连续 2 次工具调用（达到上限）→ 反思
    mock_llm.invoke.side_effect = [
        '[TOOL_CALL:calculator:{"expression": "1+1"}]',
        '[TOOL_CALL:calculator:{"expression": "2+2"}]',
        "无需改进",
    ]

    result = await agent.run("持续调用工具测试")

    assert "已达到最大工具调用次数限制" in result
    assert mock_run_parallel.call_count == 2


@patch("momu_agent.agents.reflection_agent.run_parallel_tools")
async def test_tool_called_during_refine_phase(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    优化阶段 LLM 也可以调用工具
    """
    agent = ReflectionAgent(
        name="ToolAgent", llm=mock_llm, tool_registry=mock_registry, max_iterations=1
    )

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {"expression": "100 * 0.8"},
            "result": "80.0",
            "status": "success",
        }
    ]
    mock_llm.invoke.side_effect = [
        "初始回答（未计算）",  # 初始生成
        "需要精确计算折扣价格",  # 反思
        '[TOOL_CALL:calculator:{"expression": "100 * 0.8"}]',  # 优化阶段调用工具
        "折扣后价格为 80 元",  # 工具结果后的优化答案
    ]

    result = await agent.run("100 元打八折是多少？")

    assert result == "折扣后价格为 80 元"
    assert mock_run_parallel.call_count == 1
