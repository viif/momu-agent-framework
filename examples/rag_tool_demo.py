"""
RAGTool + SimpleAgent 使用示例

演示三种使用方式：
1. 通过 Agent 调用 rag 工具添加内联文本并检索
2. 通过 Agent 调用 rag 工具添加本地文档并检索
3. 通过 Agent 调用 rag 工具进行问答与统计

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

from __future__ import annotations

import asyncio
import tempfile
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

    questions = [
        (
            "请调用 rag 工具 add_text，写入文档 doc_python，"
            "text 为：MomuAgent 是一个灵活的智能体框架。"
            "它的 Tool.run 接口是异步的，ToolRegistry 会统一 await 工具执行结果。"
            "框架内置了 CalculatorTool、SearchTool 和 RAGTool。"
        ),
        (
            "请调用 rag 工具 add_text，写入文档 doc_rag，"
            "text 为：RAGTool 支持 add_text、add_document、search、ask 和 stats 动作。"
            "search 会先检索相关片段，ask 会把片段作为上下文交给 LLM 生成回答。"
        ),
        "请调用 rag 工具 search，查询“RAGTool 有哪些动作？”，limit=3。",
    ]

    for question in questions:
        print(f"用户: {question}")
        response = await agent.run(question, max_tool_iterations=3)
        print(f"Agent: {response}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        document_path = Path(temp_dir) / "rag_notes.md"
        document_path.write_text(
            "# RAG 使用笔记\n\n"
            "- add_document 适合导入已有文件\n"
            "- search 返回相关片段和分数\n"
            "- ask 会基于检索上下文生成答案\n",
            encoding="utf-8",
        )

        question_add_doc = (
            "请调用 rag 工具 add_document 导入文件："
            f"{document_path.as_posix()}"
        )
        print(f"用户: {question_add_doc}")
        response = await agent.run(question_add_doc, max_tool_iterations=3)
        print(f"Agent: {response}\n")

    follow_up_questions = [
        "请调用 rag 工具 search，查询“ask 动作做什么？”，limit=3。",
        "请调用 rag 工具 ask，问题是“RAGTool 的 ask 动作和 search 动作有什么区别？”，limit=3，max_chars=800。",
        "请调用 rag 工具 stats 查看当前统计信息。",
    ]

    for question in follow_up_questions:
        print(f"用户: {question}")
        response = await agent.run(question, max_tool_iterations=3)
        print(f"Agent: {response}\n")


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_rag_with_agent(config)


if __name__ == "__main__":
    asyncio.run(main())
