from unittest.mock import AsyncMock, Mock

import pytest

from momu_agent.agents.reflection_agent import Memory, ReflectionAgent


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


# ── 自定义提示词 ──────────────────────────────────────────────────────────────


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
