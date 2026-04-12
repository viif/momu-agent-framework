"""
ReflectionAgent 使用示例

演示两种使用模式：
1. 通用写作优化 — 展示初始生成、反思、优化的迭代流程
2. 代码生成专家 — 通过自定义提示词将 Agent 定制为代码审查场景

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

import asyncio

from momu_agent.agents.reflection_agent import ReflectionAgent
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


# ---------- 示例 1：通用写作优化 ----------
async def demo_writing_refinement(config: Config):
    print("=" * 50)
    print("示例 1：通用写作优化")
    print("=" * 50)

    agent = ReflectionAgent(
        name="写作助手",
        llm=build_llm(config),
        max_history_length=config.max_history_length,
        max_iterations=2,
    )

    tasks = [
        "用三句话介绍人工智能的发展历程，面向没有技术背景的读者。",
        "写一段关于团队协作重要性的职场感悟，约 100 字。",
    ]

    for task in tasks:
        print(f"\n用户: {task}")
        response = await agent.run(task)
        print(f"Agent: {response}\n")


# ---------- 示例 2：代码生成专家 ----------
async def demo_code_generation(config: Config):
    print("=" * 50)
    print("示例 2：代码生成专家（自定义提示词）")
    print("=" * 50)

    code_prompts = {
        "initial": """
请根据以下需求编写 Python 代码：

需求: {task}

要求：
- 代码结构清晰，包含必要的注释
- 处理常见的边界情况
- 提供简短的使用示例
""",
        "reflect": """
请对以下 Python 代码进行专业代码审查：

# 原始需求:
{task}

# 待审查代码:
{content}

请从以下维度评审：
1. 正确性：逻辑是否正确，边界情况是否处理
2. 可读性：命名、注释、结构是否清晰
3. 健壮性：异常处理是否完善

如果代码已符合生产标准，请回答"无需改进"。
""",
        "refine": """
请根据审查意见改进以下 Python 代码：

# 原始需求:
{task}

# 当前代码:
{last_attempt}

# 审查意见:
{feedback}

请提供改进后的完整代码：
""",
    }

    agent = ReflectionAgent(
        name="代码专家",
        llm=build_llm(config),
        max_history_length=config.max_history_length,
        max_iterations=2,
        custom_prompts=code_prompts,
    )

    task = "实现一个函数，接收一个整数列表，返回其中所有素数组成的新列表。"
    print(f"\n用户: {task}")
    response = await agent.run(task)
    print(f"Agent:\n{response}\n")


async def main():
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_writing_refinement(config)
    await demo_code_generation(config)


if __name__ == "__main__":
    asyncio.run(main())
