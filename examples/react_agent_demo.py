"""
ReActAgent 使用示例

演示三种使用模式：
1. 纯推理（无工具）——展示 Thought/Action/Finish 流程
2. 带计算器工具的多步推理
3. 带搜索工具的知识查询（需在 .env 中配置 TAVILY_API_KEY 或 SERPAPI_API_KEY）

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

import asyncio

from momu_agent.agents.react_agent import ReActAgent
from momu_agent.core.config import Config
from momu_agent.core.llm import LLM
from momu_agent.tools.builtin.calculator import CalculatorTool
from momu_agent.tools.builtin.search import SearchTool
from momu_agent.tools.registry import ToolRegistry
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


# ---------- 示例 1：纯推理（无工具） ----------
async def demo_pure_reasoning(config: Config):
    print("=" * 50)
    print("示例 1：纯推理（无工具）")
    print("=" * 50)

    agent = ReActAgent(
        name="ReasoningAgent",
        llm=build_llm(config),
        max_history_length=config.max_history_length,
        max_steps=3,
    )

    question = "一个班有 30 名学生，其中 60% 是女生。男生比女生少几人？"
    print(f"用户: {question}")
    response = await agent.run(question)
    print(f"Agent: {response}\n")


# ---------- 示例 2：带计算器工具的多步推理 ----------
async def demo_with_calculator(config: Config):
    print("=" * 50)
    print("示例 2：带计算器工具的多步推理")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(
        name="MathAgent",
        llm=build_llm(config),
        tool_registry=registry,
        max_history_length=config.max_history_length,
        max_steps=5,
    )

    questions = [
        "计算 (sqrt(256) + log(1000)) * pi 的近似值，保留两位小数。",
        "一个圆的半径为 7，求其面积和周长（π 取精确值）。",
    ]
    for q in questions:
        print(f"用户: {q}")
        response = await agent.run(q)
        print(f"Agent: {response}\n")


# ---------- 示例 3：带搜索工具的知识查询 ----------
async def demo_with_search(config: Config):
    print("=" * 50)
    print("示例 3：带搜索工具的知识查询")
    print("=" * 50)

    if not config.tavily_api_key and not config.serpapi_api_key:
        print("⚠️  未配置搜索 API Key（TAVILY_API_KEY / SERPAPI_API_KEY），跳过此示例。\n")
        return

    registry = ToolRegistry()
    registry.register_tool(
        SearchTool(
            tavily_api_key=config.tavily_api_key,
            serpapi_key=config.serpapi_api_key,
        )
    )

    agent = ReActAgent(
        name="SearchAgent",
        llm=build_llm(config),
        tool_registry=registry,
        max_history_length=config.max_history_length,
        max_steps=5,
    )

    question = "2024 年诺贝尔物理学奖颁给了谁？他们的主要贡献是什么？"
    print(f"用户: {question}")
    response = await agent.run(question)
    print(f"Agent: {response}\n")


async def main():
    # 从项目根目录的 .env 文件加载配置
    config = Config.from_env()

    # 按配置初始化日志级别
    setup_logger(level=config.log_level)

    await demo_pure_reasoning(config)
    await demo_with_calculator(config)
    await demo_with_search(config)


if __name__ == "__main__":
    asyncio.run(main())
