"""
PlanSolveAgent 使用示例

演示三种使用模式：
1. 多步骤数学推理 — 展示规划分解与逐步执行流程
2. 自定义提示词 — 展示如何通过 custom_prompts 定制规划器和执行器
3. 工具调用 — 展示执行阶段结合计算器工具完成精确计算

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

import asyncio

from momu_agent.agents.plan_solve_agent import PlanSolveAgent
from momu_agent.core.llm import LLM
from momu_agent.tools.builtin.calculator import CalculatorTool
from momu_agent.tools.registry import ToolRegistry
from momu_agent.utils.config import Config
from momu_agent.utils.logger import setup_logger


def build_llm(config: Config) -> LLM:
    return LLM(
        model=config.model_id,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=int(config.timeout),
    )


# ---------- 示例 1：多步骤数学推理 ----------
async def demo_math_reasoning(config: Config):
    print("=" * 50)
    print("示例 1：多步骤数学推理")
    print("=" * 50)

    agent = PlanSolveAgent(
        name="MathPlanAgent",
        llm=build_llm(config),
        max_history_length=config.max_history_length,
    )

    question = (
        "某工厂第一季度生产了 1200 件产品，第二季度比第一季度增长了 25%，"
        "第三季度又比第二季度减少了 10%。前三季度总产量是多少？"
    )
    print(f"\n用户: {question}")
    response = await agent.run(question)
    print(f"Agent: {response}")


# ---------- 示例 2：自定义提示词 ----------
async def demo_custom_prompts(config: Config):
    print("\n" + "=" * 50)
    print("示例 2：自定义提示词")
    print("=" * 50)

    custom_planner = """
你是一位严谨的逻辑分析师。请将以下问题拆解为 2~4 个分析步骤，
每个步骤聚焦于一个独立的逻辑推断。

问题: {question}

严格按以下格式输出:
```python
["分析步骤1", "分析步骤2", ...]
```
"""

    custom_executor = """
你是一位逻辑推理专家。请根据下方信息，仅回答"当前步骤"所要求的内容。

问题: {question}
完整分析计划: {plan}
已完成步骤: {history}
当前步骤: {current_step}

仅输出当前步骤的结论：
"""

    agent = PlanSolveAgent(
        name="LogicAnalystAgent",
        llm=build_llm(config),
        max_history_length=config.max_history_length,
        custom_prompts={"planner": custom_planner, "executor": custom_executor},
    )

    question = (
        "所有会飞的动物都有翅膀。蝙蝠会飞。鸵鸟有翅膀但不会飞。"
        '请分析：蝙蝠和鸵鸟在"会飞"与"有翅膀"这两个属性上各自的情况，并总结规律。'
    )
    print(f"\n用户: {question}")
    response = await agent.run(question)
    print(f"Agent: {response}")


# ---------- 示例 3：使用计算器工具 ----------
async def demo_with_calculator_tool(config: Config):
    print("\n" + "=" * 50)
    print("示例 3：使用计算器工具")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = PlanSolveAgent(
        name="ToolPlanAgent",
        llm=build_llm(config),
        max_history_length=config.max_history_length,
        tool_registry=registry,
        max_tool_iterations=3,
    )

    question = (
        "某投资组合由三只股票组成：A 股 50 股，单价 32.5 元；"
        "B 股 120 股，单价 18.8 元；C 股 30 股，单价 76.2 元。"
        "请计算该投资组合的总市值。"
    )
    print(f"\n用户: {question}")
    response = await agent.run(question)
    print(f"Agent: {response}")


async def main():
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_math_reasoning(config)
    await demo_custom_prompts(config)
    await demo_with_calculator_tool(config)


if __name__ == "__main__":
    asyncio.run(main())
