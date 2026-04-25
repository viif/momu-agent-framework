"""
RAGTool + SimpleAgent 使用示例

演示三种使用方式：
1. 通过 Agent 调用 rag 工具导入本地 PDF 文档
2. 通过 Agent 调用 rag 工具基于知识库检索信息
3. 通过 Agent 调用 rag 工具进行问答与统计

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from momu_agent.agents import SimpleAgent
from momu_agent.core.llm import LLM
from momu_agent.tools.builtin.rag import RAGTool
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


async def demo_rag_with_agent(config: Config) -> None:
    print("=" * 50)
    print("RAGTool 示例：SimpleAgent 调用")
    print("=" * 50)

    namespace = f"rag_demo_{int(time.time())}"
    registry = ToolRegistry()
    registry.register_tool(RAGTool(namespace=namespace, top_k=3, max_chars=1200))

    agent = SimpleAgent(
        name="RAGAgent",
        llm=build_llm(config),
        system_prompt=(
            "你是一个 RAG 助手。"
            "涉及知识库写入、检索、问答、统计时必须优先调用 rag 工具，"
            "拿到工具结果后再给出简短中文说明。"
        ),
        tool_registry=registry,
        max_history_length=config.max_history_length,
    )

    print(f"使用命名空间: {namespace}\n")

    document_path = Path(__file__).resolve().parent / "docs4rag" / "README_v0.2.0.md"

    questions = [
        f"我准备把项目说明文档接入知识库，先帮我导入这个 Markdown 文件：{document_path.as_posix()}。",
        "我想快速了解这个框架有哪些核心能力，你先从知识库里帮我检索最相关的内容。",
        "如果我是第一次接触这个项目，你基于知识库用简短的话介绍一下它的定位和主要特性。",
        "顺便看一下当前知识库的统计信息。",
    ]

    for question in questions:
        print(f"用户: {question}")
        response = await agent.run(question, max_tool_iterations=3)
        print(f"Agent: {response}\n")


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_rag_with_agent(config)


if __name__ == "__main__":
    asyncio.run(main())
