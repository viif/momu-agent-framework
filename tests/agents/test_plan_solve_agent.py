from unittest.mock import AsyncMock, Mock

import pytest

from momu_agent.agents.plan_solve_agent import PlanSolveAgent


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
