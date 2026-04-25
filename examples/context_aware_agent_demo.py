"""
ContextAwareAgent + ContextBuilder + NoteTool + TerminalTool 示例

演示流程：
1. 继承 SimpleAgent 定义 ContextAwareAgent
2. 在 run 前使用 ContextBuilder 构建结构化上下文
3. 通过 ToolRegistry 接入 NoteTool 与 TerminalTool

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

from __future__ import annotations

import asyncio

from momu_agent.agents import SimpleAgent
from momu_agent.context import ContextBuilder, ContextConfig
from momu_agent.core.llm import LLM
from momu_agent.tools.builtin import NoteTool, TerminalTool
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


class ContextAwareAgent(SimpleAgent):
    """在 SimpleAgent 基础上增加 ContextBuilder 预处理能力。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        system_prompt: str,
        max_history_length: int,
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            max_history_length=max_history_length,
            tool_registry=tool_registry,
        )
        self.context_builder = context_builder

    async def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        optimized_context = await self.context_builder.build(
            user_query=input_text,
            conversation_history=self.get_history(),
            system_instructions=self.system_prompt,
        )
        enhanced_input = (
            "以下是已构建的结构化上下文，请先理解后再执行必要工具调用。\n\n"
            f"{optimized_context}"
        )
        return await super().run(
            enhanced_input,
            max_tool_iterations=max_tool_iterations,
            **kwargs,
        )


async def demo_context_aware_agent(config: Config) -> None:
    print("=" * 50)
    print("ContextAwareAgent 示例")
    print("=" * 50)

    llm = build_llm(config)

    registry = ToolRegistry()
    registry.register_tool(NoteTool(workspace="./.demo_notes"))
    registry.register_tool(
        TerminalTool(workspace="./examples", allow_cd=False, timeout=15)
    )

    context_builder = ContextBuilder(
        llm=llm,
        config=ContextConfig(
            max_tokens=4000,
            reserve_ratio=0.15,
            min_relevance=0.2,
            enable_compression=True,
        ),
    )

    agent = ContextAwareAgent(
        name="ContextAwareAgent",
        llm=llm,
        context_builder=context_builder,
        tool_registry=registry,
        system_prompt=(
            "你是一个上下文感知助手。"
            "当问题涉及文件观察时优先调用 terminal 工具；"
            "当问题涉及记录与追踪时优先调用 note 工具；"
            "调用 note create 时不要传 tags 参数；"
            "当用户要求写入笔记时，content 必须尽量完整，不得用笼统短句代替。"
            "拿到工具结果后再给出简洁中文结论。"
        ),
        max_history_length=config.max_history_length,
    )

    questions = [
        "帮我看看当前目录里有哪些文件，并做一个关于这些文件内容的总结笔记。",
        "告诉我文件内容总结笔记中的内容。",
    ]

    for question in questions:
        print(f"用户: {question}")
        response = await agent.run(question, max_tool_iterations=5)
        print(f"Agent: {response}\n")


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    await demo_context_aware_agent(config)


if __name__ == "__main__":
    asyncio.run(main())
