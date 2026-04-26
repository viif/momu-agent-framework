"""
MCP Client + SimpleAgent 使用示例

演示：
1. 通过 MCPClient 连接 stdio MCP Server
2. 将远程工具注册到 ToolRegistry
3. 直接执行远程工具
4. 通过 SimpleAgent 调用远程工具

运行前请先复制 .env.example 为 .env 并填写相关配置。
本示例默认使用 Node.js 的 npx 启动 filesystem MCP Server。
"""

from __future__ import annotations

import asyncio

from momu_agent.agents import SimpleAgent
from momu_agent.core.llm import LLM
from momu_agent.mcp import MCPClient
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


async def demo_mcp_with_agent(config: Config) -> None:
    print("=" * 50)
    print("MCP Client 示例：SimpleAgent 调用")
    print("=" * 50)

    registry = ToolRegistry()
    client = MCPClient()

    command = [
        "npx",
        "-y",
        "@modelcontextprotocol/server-filesystem",
        ".",
    ]

    await client.connect(command)
    registered = await registry.register_mcp_client("filesystem", client)
    print("已注册工具:")
    for tool_name in registered:
        print(f"- {tool_name}")
    print()

    direct_result = await registry.execute_tool(
        "filesystem.list_directory",
        {"path": "."},
    )
    print("直接调用结果:")
    print(direct_result)
    print()

    agent = SimpleAgent(
        name="MCPAgent",
        llm=build_llm(config),
        system_prompt=(
            "你是一个文件助手。需要查看文件系统时必须优先调用 filesystem 工具，"
            "拿到工具结果后再用简短中文回答。"
        ),
        tool_registry=registry,
        max_history_length=config.max_history_length,
    )

    question = "请查看当前目录下有哪些文件，并简要说明这个项目里最值得先看的入口。"
    print(f"用户: {question}")
    response = await agent.run(question, max_tool_iterations=3)
    print(f"Agent: {response}")


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_mcp_with_agent(config)


if __name__ == "__main__":
    asyncio.run(main())
