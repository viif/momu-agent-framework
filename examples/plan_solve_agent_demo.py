"""
PlanSolveAgent 使用示例

演示两种使用模式：
1. 多步骤数学推理 — 展示规划分解与逐步执行流程
2. 自定义提示词 — 展示如何通过 custom_prompts 定制规划器和执行器

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

import asyncio

from momu_agent.agents.plan_solve_agent import PlanSolveAgent
from momu_agent.core.config import Config
from momu_agent.core.llm import LLM
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

    questions = [
        (
            "一个储蓄罐里有硬币：1 元硬币 15 枚，5 角硬币 20 枚，1 角硬币 30 枚。"
            "如果取出总金额的 40%，能取出多少钱？"
        ),
        (
            "某工厂第一季度生产了 1200 件产品，第二季度比第一季度增长了 25%，"
            "第三季度又比第二季度减少了 10%。前三季度总产量是多少？"
        ),
    ]

    for q in questions:
        print(f"\n用户: {q}")
        response = await agent.run(q)
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


async def main():
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_math_reasoning(config)
    await demo_custom_prompts(config)


if __name__ == "__main__":
    asyncio.run(main())
