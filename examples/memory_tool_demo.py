"""
MemoryTool 使用示例

演示三种使用方式：
1. 添加不同类型的记忆
2. 检索已保存的记忆
3. 查看记忆统计信息

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env

注意：
- 该示例直接调用 MemoryTool，不依赖 agent 自动触发工具调用
- 示例会为本次运行生成独立 user_id，避免与历史数据混在一起
"""

from __future__ import annotations

import asyncio
import time

from momu_agent.tools import MemoryTool
from momu_agent.utils.config import Config
from momu_agent.utils.logger import setup_logger


async def demo_add_memories(tool: MemoryTool) -> None:
    print("=" * 50)
    print("示例 1：添加记忆")
    print("=" * 50)

    items = [
        {
            "action": "add",
            "content": "用户喜欢燕麦拿铁，不加糖。",
            "memory_type": "working",
            "importance": 0.7,
            "auto_classify": False,
        },
        {
            "action": "add",
            "content": "Python 异步工具调用是 MomuAgent 的核心概念之一。",
            "memory_type": "semantic",
            "metadata": {"concepts": ["python", "async", "tool"]},
            "auto_classify": False,
        },
        {
            "action": "add",
            "content": "今天完成了记忆工具 demo 的编写。",
            "memory_type": "episodic",
            "metadata": {"session_id": "demo-session-1"},
            "auto_classify": False,
        },
    ]

    for payload in items:
        result = await tool.run(payload)
        print(result)
        print()


async def demo_search(tool: MemoryTool) -> None:
    print("=" * 50)
    print("示例 2：检索记忆")
    print("=" * 50)

    result = await tool.run(
        {
            "action": "search",
            "query": "异步工具",
            "limit": 5,
        }
    )
    print(result)
    print()

    result = await tool.run(
        {
            "action": "search",
            "query": "Python",
            "memory_type": "semantic",
            "limit": 5,
        }
    )
    print(result)
    print()


async def demo_stats(tool: MemoryTool) -> None:
    print("=" * 50)
    print("示例 3：查看统计信息")
    print("=" * 50)

    result = await tool.run({"action": "stats"})
    print(result)
    print()


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    demo_user_id = f"memory_demo_{int(time.time())}"
    tool = MemoryTool(user_id=demo_user_id)

    print(f"使用 user_id: {demo_user_id}")
    print()

    await demo_add_memories(tool)
    await demo_search(tool)
    await demo_stats(tool)


if __name__ == "__main__":
    asyncio.run(main())
