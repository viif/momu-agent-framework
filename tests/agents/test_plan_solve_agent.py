from unittest.mock import AsyncMock, Mock, patch

import pytest

from momu_agent.agents.plan_solve_agent import PlanSolveAgent
from momu_agent.core.exceptions import AgentException
from momu_agent.tools.base import ToolParameter
from momu_agent.tools.registry import ToolRegistry


@pytest.fixture
def mock_llm():
    """创建模拟 LLM"""
    llm = Mock()
    llm.model = "mock-model"
    llm.invoke = AsyncMock()
    return llm


# ---------- 成功路径 ----------


async def test_plan_and_execute_success(mock_llm):
    """
    测试：规划 + 执行完整流程（2 步计划）
    """
    agent = PlanSolveAgent(name="TestAgent", llm=mock_llm)

    plan_response = '```python\n["步骤1：计算总价", "步骤2：计算折扣后价格"]\n```'
    step1_response = "总价为 100 元"
    step2_response = "折扣后价格为 80 元"

    mock_llm.invoke.side_effect = [plan_response, step1_response, step2_response]

    result = await agent.run("商品原价 100 元，打八折后是多少？")

    assert result == step2_response
    assert mock_llm.invoke.call_count == 3  # 1 次规划 + 2 次执行
    assert len(agent._history) == 2  # user + assistant


async def test_history_stored_after_run(mock_llm):
    """
    测试：运行后历史记录包含用户输入和最终答案
    """
    agent = PlanSolveAgent(name="HistoryAgent", llm=mock_llm)

    mock_llm.invoke.side_effect = [
        '```python\n["唯一步骤"]\n```',
        "最终答案",
    ]

    await agent.run("测试问题")

    assert len(agent._history) == 2
    assert agent._history[0].role == "user"
    assert agent._history[0].content == "测试问题"
    assert agent._history[1].role == "assistant"
    assert agent._history[1].content == "最终答案"


async def test_executor_accumulates_history(mock_llm):
    """
    测试：执行器在每步中传入累积的历史上下文
    """
    agent = PlanSolveAgent(name="ContextAgent", llm=mock_llm)

    mock_llm.invoke.side_effect = [
        '```python\n["步骤A", "步骤B", "步骤C"]\n```',
        "结果A",
        "结果B",
        "结果C",
    ]

    await agent.run("三步测试")

    assert mock_llm.invoke.call_count == 4  # 1 次规划 + 3 次执行

    # 第 3 步（index=3）的 prompt 应包含前两步的结果
    third_call_messages = mock_llm.invoke.call_args_list[3][0][0]
    prompt_content = third_call_messages[0]["content"]
    assert "结果A" in prompt_content
    assert "结果B" in prompt_content


# ---------- 失败路径 ----------


async def test_plan_parse_failure_returns_error(mock_llm):
    """
    测试：规划器无法解析 LLM 响应时，返回错误消息并不进行执行
    """
    agent = PlanSolveAgent(name="ParseFailAgent", llm=mock_llm)
    mock_llm.invoke.return_value = "这不是一个合法的 Python 列表"

    result = await agent.run("解析失败测试")

    assert "无法生成有效的行动计划" in result
    assert mock_llm.invoke.call_count == 1  # 只调用了规划，未进入执行


async def test_empty_plan_stores_history(mock_llm):
    """
    测试：空计划时，历史记录仍存储用户输入和错误消息
    """
    agent = PlanSolveAgent(name="EmptyPlanAgent", llm=mock_llm)
    mock_llm.invoke.return_value = "无效响应"

    await agent.run("空计划测试")

    assert len(agent._history) == 2
    assert agent._history[0].role == "user"
    assert agent._history[1].role == "assistant"
    assert "无法生成有效的行动计划" in agent._history[1].content


async def test_planner_raises_agent_exception_on_invalid_plan(mock_llm):
    """
    测试：直接调用 planner.plan()，当规划结果无法解析时抛出 AgentException
    """
    agent = PlanSolveAgent(name="PlannerFailAgent", llm=mock_llm)
    mock_llm.invoke.return_value = "不是合法计划"

    with pytest.raises(AgentException, match="无法生成有效的行动计划"):
        await agent.planner.plan("规划异常测试")


async def test_executor_raises_agent_exception_on_empty_response(mock_llm):
    """
    测试：直接调用 executor.execute()，当执行阶段 LLM 返回空响应时抛出 AgentException
    """
    agent = PlanSolveAgent(name="ExecutorFailAgent", llm=mock_llm)
    mock_llm.invoke.return_value = ""

    with pytest.raises(AgentException, match="执行阶段 LLM 未返回有效响应"):
        await agent.executor.execute("执行异常测试", ["步骤1"])


async def test_run_returns_error_when_planner_llm_empty(mock_llm):
    """
    测试：run() 中规划阶段 LLM 空响应时，统一包装为执行错误消息并写入历史
    """
    agent = PlanSolveAgent(name="RunPlannerEmptyAgent", llm=mock_llm)
    mock_llm.invoke.return_value = ""

    result = await agent.run("空响应测试")

    assert "执行错误" in result
    assert "规划阶段 LLM 未返回有效响应" in result
    assert len(agent._history) == 2
    assert agent._history[0].role == "user"
    assert agent._history[1].role == "assistant"


# ---------- 自定义提示词 ----------


async def test_custom_planner_prompt(mock_llm):
    """
    测试：使用自定义规划提示词时，规划器调用 LLM 传入的 prompt 包含自定义内容
    """
    custom_planner = "自定义规划提示: {question}"
    agent = PlanSolveAgent(
        name="CustomAgent",
        llm=mock_llm,
        custom_prompts={"planner": custom_planner},
    )

    mock_llm.invoke.side_effect = [
        '```python\n["唯一步骤"]\n```',
        "答案",
    ]

    await agent.run("测试问题")

    planner_call_messages = mock_llm.invoke.call_args_list[0][0][0]
    assert "自定义规划提示" in planner_call_messages[0]["content"]
    assert "测试问题" in planner_call_messages[0]["content"]


async def test_custom_executor_prompt(mock_llm):
    """
    测试：使用自定义执行提示词时，执行器调用 LLM 传入的 prompt 包含自定义内容
    """
    custom_executor = "自定义执行: {question} | {plan} | {history} | {current_step}"
    agent = PlanSolveAgent(
        name="CustomExecAgent",
        llm=mock_llm,
        custom_prompts={"executor": custom_executor},
    )

    mock_llm.invoke.side_effect = [
        '```python\n["步骤1"]\n```',
        "执行结果",
    ]

    await agent.run("执行测试")

    executor_call_messages = mock_llm.invoke.call_args_list[1][0][0]
    assert "自定义执行" in executor_call_messages[0]["content"]


# ---------- 初始化 ----------


def test_default_initialization(mock_llm):
    """
    测试：默认参数初始化后，planner 和 executor 使用默认提示词模板
    """
    from momu_agent.agents.plan_solve_agent import (
        DEFAULT_EXECUTOR_PROMPT,
        DEFAULT_PLANNER_PROMPT,
    )

    agent = PlanSolveAgent(name="InitAgent", llm=mock_llm)

    assert agent.planner.prompt_template == DEFAULT_PLANNER_PROMPT
    assert agent.executor.prompt_template == DEFAULT_EXECUTOR_PROMPT
    assert agent.max_history_length == 100


# ---------- 工具调用测试 ----------


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

    def run(self, parameters: dict) -> str:
        return str(eval(parameters["expression"]))  # noqa: S307


@pytest.fixture
def mock_registry():
    """创建模拟工具注册表"""
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_description = Mock(
        return_value="calculator: 计算数学表达式\n  - expression (string, 必填): 要计算的数学表达式"
    )
    registry.get_tool = Mock(return_value=MockCalculatorTool())
    return registry


def test_plan_solve_agent_init_with_tool_registry(mock_llm, mock_registry):
    """
    测试：传入 tool_registry 后，agent 与 executor 均持有该注册表引用，
    且 max_tool_iterations 默认为 3
    """
    agent = PlanSolveAgent(name="ToolAgent", llm=mock_llm, tool_registry=mock_registry)

    assert agent.tool_registry is mock_registry
    assert agent.executor.tool_registry is mock_registry
    assert agent.executor.max_tool_iterations == 3


async def test_executor_includes_tool_system_prompt(mock_llm, mock_registry):
    """
    测试：有 tool_registry 时，执行器向 LLM 发送的消息首条为 system role，
    且包含工具描述信息
    """
    agent = PlanSolveAgent(name="ToolAgent", llm=mock_llm, tool_registry=mock_registry)

    mock_llm.invoke.side_effect = [
        '```python\n["计算步骤"]\n```',
        "计算结果为 42",
    ]

    await agent.run("测试工具提示词")

    executor_messages = mock_llm.invoke.call_args_list[1][0][0]
    assert executor_messages[0]["role"] == "system"
    assert "calculator" in executor_messages[0]["content"]
    assert "TOOL_CALL" in executor_messages[0]["content"]


@patch("momu_agent.agents.plan_solve_agent.run_parallel_tools")
async def test_executor_step_calls_tool_and_continues(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    测试：执行步骤中 LLM 返回工具调用 → run_parallel_tools 被调用 →
    LLM 再次被调用获取最终答案
    """
    agent = PlanSolveAgent(name="ToolAgent", llm=mock_llm, tool_registry=mock_registry)

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {"expression": "100 * 3"},
            "result": "300",
            "status": "success",
        }
    ]
    mock_llm.invoke.side_effect = [
        '```python\n["计算总价"]\n```',
        '[TOOL_CALL:calculator:{"expression": "100 * 3"}]',
        "总价为 300 元",
    ]

    result = await agent.run("买了 3 件 100 元的商品，总价是多少？")

    assert result == "总价为 300 元"
    assert mock_run_parallel.called
    assert mock_llm.invoke.call_count == 3  # 1 规划 + 1 工具调用 + 1 最终答案


@patch("momu_agent.agents.plan_solve_agent.run_parallel_tools")
async def test_executor_tool_result_in_continuation_messages(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    测试：工具执行结果作为 tool role 消息传入下一次 LLM 调用
    """
    agent = PlanSolveAgent(name="ToolAgent", llm=mock_llm, tool_registry=mock_registry)

    mock_run_parallel.return_value = [
        {
            "task_id": 0,
            "tool_name": "calculator",
            "input_data": {"expression": "5 * 20"},
            "result": "100",
            "status": "success",
        }
    ]
    mock_llm.invoke.side_effect = [
        '```python\n["计算步骤"]\n```',
        '[TOOL_CALL:calculator:{"expression": "5 * 20"}]',
        "结果是 100",
    ]

    await agent.run("5 乘以 20 是多少？")

    # 第 3 次调用（index=2）是工具调用后的继续调用
    continuation_messages = mock_llm.invoke.call_args_list[2][0][0]
    roles = [m["role"] for m in continuation_messages]
    assert "tool" in roles
    # tool 消息应包含工具结果
    tool_msg = next(m for m in continuation_messages if m["role"] == "tool")
    assert "100" in tool_msg["content"]


@patch("momu_agent.agents.plan_solve_agent.run_parallel_tools")
async def test_executor_tool_error_in_step(mock_run_parallel, mock_llm, mock_registry):
    """
    测试：工具执行失败时，错误信息以 ❌ 格式写入 tool 消息，执行继续
    """
    agent = PlanSolveAgent(name="ToolAgent", llm=mock_llm, tool_registry=mock_registry)

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
        '```python\n["计算步骤"]\n```',
        '[TOOL_CALL:calculator:{"expression": "1/0"}]',
        "无法计算，发生了除零错误",
    ]

    result = await agent.run("计算 1/0")

    continuation_messages = mock_llm.invoke.call_args_list[2][0][0]
    tool_msg = next(m for m in continuation_messages if m["role"] == "tool")
    assert "❌" in tool_msg["content"]
    assert result == "无法计算，发生了除零错误"


@patch("momu_agent.agents.plan_solve_agent.run_parallel_tools")
async def test_executor_max_tool_iterations_in_step(
    mock_run_parallel, mock_llm, mock_registry
):
    """
    测试：单步中 LLM 持续返回工具调用，达到 max_tool_iterations 后返回警告信息
    """
    agent = PlanSolveAgent(
        name="ToolAgent",
        llm=mock_llm,
        tool_registry=mock_registry,
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
    # 规划 1 次 + 步骤内连续 2 次工具调用（达到上限）
    mock_llm.invoke.side_effect = [
        '```python\n["唯一步骤"]\n```',
        '[TOOL_CALL:calculator:{"expression": "1+1"}]',
        '[TOOL_CALL:calculator:{"expression": "2+2"}]',
    ]

    result = await agent.run("持续调用工具测试")

    assert "已达到最大工具调用次数限制" in result
    assert mock_run_parallel.call_count == 2


async def test_no_tool_registry_behavior_unchanged(mock_llm):
    """
    测试：未传入 tool_registry 时，执行器行为与原始版本完全一致（无 system 消息，单次 LLM 调用）
    """
    agent = PlanSolveAgent(name="NoToolAgent", llm=mock_llm)

    mock_llm.invoke.side_effect = [
        '```python\n["唯一步骤"]\n```',
        "直接回答",
    ]

    result = await agent.run("无工具测试")

    assert result == "直接回答"
    # 执行步骤的消息列表第一条应为 user role，不含 system
    executor_messages = mock_llm.invoke.call_args_list[1][0][0]
    assert executor_messages[0]["role"] == "user"
    assert len(executor_messages) == 1
