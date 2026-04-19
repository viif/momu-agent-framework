"""
RAGTool 使用示例

演示三种使用方式：
1. 添加内联文本到知识库
2. 添加本地文档并检索
3. 基于检索上下文进行问答

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env

注意：
- RAG 功能依赖嵌入模型，首次运行可能下载 sentence-transformers 模型
- `ask` 动作会调用 LLM，请确保 LLM 相关环境变量已配置
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from momu_agent.tools.builtin.rag import RAGTool
from momu_agent.utils.config import Config
from momu_agent.utils.logger import setup_logger


async def demo_add_text_and_search(tool: RAGTool, namespace: str) -> None:
    print("=" * 50)
    print("示例 1：添加内联文本并检索")
    print("=" * 50)

    texts = [
        (
            "doc_python",
            "MomuAgent 是一个灵活的智能体框架。"
            "它的 Tool.run 接口是异步的，ToolRegistry 会统一 await 工具执行结果。"
            "框架内置了 CalculatorTool、SearchTool 和 RAGTool。",
        ),
        (
            "doc_rag",
            "RAGTool 支持 add_text、add_document、search、ask 和 stats 动作。"
            "search 会先检索相关片段，ask 会把片段作为上下文交给 LLM 生成回答。",
        ),
    ]

    for document_id, text in texts:
        result = await tool.run(
            {
                "action": "add_text",
                "text": text,
                "document_id": document_id,
                "namespace": namespace,
            }
        )
        print(result)
        print()

    result = await tool.run(
        {
            "action": "search",
            "query": "RAGTool 有哪些动作？",
            "namespace": namespace,
            "limit": 3,
        }
    )
    print(result)
    print()


async def demo_add_document(tool: RAGTool, namespace: str) -> None:
    print("=" * 50)
    print("示例 2：添加本地文档并检索")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        document_path = Path(temp_dir) / "rag_notes.md"
        document_path.write_text(
            "# RAG 使用笔记\n\n"
            "- add_document 适合导入已有文件\n"
            "- search 返回相关片段和分数\n"
            "- ask 会基于检索上下文生成答案\n",
            encoding="utf-8",
        )

        result = await tool.run(
            {
                "action": "add_document",
                "file_path": str(document_path),
                "namespace": namespace,
            }
        )
        print(result)
        print()

        result = await tool.run(
            {
                "action": "search",
                "query": "ask 动作做什么？",
                "namespace": namespace,
                "limit": 3,
            }
        )
        print(result)
        print()


async def demo_ask(tool: RAGTool, namespace: str) -> None:
    print("=" * 50)
    print("示例 3：基于检索上下文问答")
    print("=" * 50)

    question = "RAGTool 的 ask 动作和 search 动作有什么区别？"
    print(f"问题: {question}")
    print()

    result = await tool.run(
        {
            "action": "ask",
            "question": question,
            "namespace": namespace,
            "limit": 3,
            "max_chars": 800,
        }
    )
    print(result)
    print()


async def demo_stats(tool: RAGTool, namespace: str) -> None:
    print("=" * 50)
    print("示例 4：查看统计信息")
    print("=" * 50)

    result = await tool.run({"action": "stats", "namespace": namespace})
    print(result)
    print()


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    namespace = f"rag_demo_{int(time.time())}"
    tool = RAGTool(namespace=namespace, top_k=3, max_chars=1200)

    print(f"使用命名空间: {namespace}")
    print()

    await demo_add_text_and_search(tool, namespace)
    await demo_add_document(tool, namespace)
    await demo_ask(tool, namespace)
    await demo_stats(tool, namespace)


if __name__ == "__main__":
    asyncio.run(main())
