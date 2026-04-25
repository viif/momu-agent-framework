"""
MemoryTool + SimpleAgent 使用示例

演示三种使用方式：
1. 通过 Agent 调用 memory 工具添加记忆
2. 通过 Agent 调用 memory 工具检索记忆
3. 通过 Agent 调用 memory 工具查看统计信息

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

from __future__ import annotations

import asyncio
import time

from momu_agent.agents import SimpleAgent
from momu_agent.core.llm import LLM
from momu_agent.tools.builtin.memory import MemoryTool
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


async def demo_memory_with_agent(config: Config) -> None:
    print("=" * 50)
    print("MemoryTool 示例：SimpleAgent 调用")
    print("=" * 50)

    demo_user_id = f"memory_demo_{int(time.time())}"
    registry = ToolRegistry()
    registry.register_tool(
        MemoryTool(user_id=demo_user_id, memory_types=["working", "episodic"])
    )

    agent = SimpleAgent(
        name="MemoryAgent",
        llm=build_llm(config),
        system_prompt=(
            "你是一个记忆助手。"
            "涉及记忆操作时必须优先调用 memory 工具，"
            "拿到工具结果后再给出简短中文说明。"
        ),
        tool_registry=registry,
        max_history_length=config.max_history_length,
    )

    print(f"使用 user_id: {demo_user_id}\n")

    questions = [
        "帮我记一下，我平时喜欢喝燕麦拿铁，不另外加糖。",
        "再帮我记一条今天发生的事：我刚跑通了 MemoryTool 的 SimpleAgent 示例。",
        "我刚才提到的咖啡偏好是什么来着？",
        "顺便看看现在记忆库里有多少内容。",
    ]

    for question in questions:
        print(f"用户: {question}")
        response = await agent.run(question, max_tool_iterations=3)
        print(f"Agent: {response}\n")


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_memory_with_agent(config)


if __name__ == "__main__":
    asyncio.run(main())
