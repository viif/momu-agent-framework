"""
Skills + SimpleAgent 使用示例

演示四种使用方式：
1. 通过 Agent 列出本地可用技能
2. 通过 Agent 加载技能并总结真实 Markdown 文档
3. 通过 Agent 加载技能并审查真实 Python 示例
4. 通过 Agent 刷新 skills 缓存

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env

本示例默认读取项目根目录下的 ./skills，并通过 terminal 工具读取本地文件。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from momu_agent.agents import SimpleAgent
from momu_agent.core.exceptions import AgentException
from momu_agent.core.llm import LLM
from momu_agent.skills import SkillLoader
from momu_agent.tools.builtin.skills import SkillsTool
from momu_agent.tools.builtin.terminal import TerminalTool
from momu_agent.tools.registry import ToolRegistry
from momu_agent.utils.config import Config
from momu_agent.utils.logger import setup_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"
MARKDOWN_TARGET = REPO_ROOT / "examples" / "docs4demo" / "README_v0.2.0.md"
PYTHON_TARGET = REPO_ROOT / "examples" / "simple_agent_demo.py"


def build_llm(config: Config) -> LLM:
    return LLM(
        model=config.model_id,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=int(config.timeout),
    )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(SkillsTool(skills_dir=str(SKILLS_DIR)))
    registry.register_tool(
        TerminalTool(workspace=str(REPO_ROOT), allow_cd=False, timeout=15)
    )
    return registry


def build_agent(config: Config) -> SimpleAgent:
    return SimpleAgent(
        name="SkillsAgent",
        llm=build_llm(config),
        system_prompt=(
            "你是擅长使用本地技能的助手。"
            "你只有两个可执行工具：skills 和 terminal。"
            "markdown-summary、python-review 是技能名，不是工具名，绝不能直接把它们当工具调用。"
            "skills 只负责列出、加载、刷新技能说明，不能读取本地文件。"
            "terminal 才负责读取本地文件内容。"
            "如果任务涉及总结或审查本地文件，先用 skills 加载最合适的技能，再用 terminal 读取目标文件，然后基于技能说明和文件内容给出结果。"
            "只能调用工具列表中真实注册的工具名。"
            "获取结果后，用简洁自然的中文直接回答。"
        ),
        tool_registry=build_registry(),
        max_history_length=config.max_history_length,
    )


def ensure_skills_available() -> bool:
    loader = SkillLoader(SKILLS_DIR)
    skill_names = loader.list_skills()
    if not skill_names:
        print("⚠️  当前 ./skills 下没有可用技能，跳过示例。")
        print("   请先添加 skills/<name>/SKILL.md。\n")
        return False

    print(f"检测到本地技能目录: {SKILLS_DIR}")
    print("可用技能:")
    for name in skill_names:
        print(f"- {name}")
    print()
    return True


def ensure_demo_targets_available() -> bool:
    missing_files = [
        path for path in [MARKDOWN_TARGET, PYTHON_TARGET] if not path.exists()
    ]
    if not missing_files:
        return True

    print("⚠️  以下 demo 目标文件不存在，跳过示例：")
    for path in missing_files:
        print(f"- {path}")
    print()
    return False


async def run_agent_question(agent: SimpleAgent, question: str) -> None:
    print(f"用户: {question}")
    try:
        response = await agent.run(question, max_tool_iterations=5)
    except AgentException as exc:
        print(f"⚠️  示例执行失败：{exc}")
        print("   请检查 .env 中的模型配置、账户额度或工具调用结果后重试。\n")
        return
    print(f"Agent: {response}\n")


async def demo_list_skills(config: Config) -> None:
    print("=" * 50)
    print("示例 1：通过 Agent 列出可用技能")
    print("=" * 50)

    agent = build_agent(config)
    question = "我想看看你现在能用哪些本地技能，顺便说说它们分别适合做什么。"
    await run_agent_question(agent, question)


async def demo_load_markdown_skill(config: Config) -> None:
    print("=" * 50)
    print("示例 2：通过 Agent 加载技能并总结真实 Markdown 文档")
    print("=" * 50)

    agent = build_agent(config)
    question = (
        "请帮我处理一个本地 Markdown 文档总结任务。"
        "请先选择并加载最合适的本地技能，再读取 examples/docs4demo/README_v0.2.0.md 的内容，"
        "最后给我一份简短的结构化摘要。"
    )
    await run_agent_question(agent, question)


async def demo_load_review_skill(config: Config) -> None:
    print("=" * 50)
    print("示例 3：通过 Agent 加载技能并审查真实 Python 示例")
    print("=" * 50)

    agent = build_agent(config)
    question = (
        "请帮我审查一个本地 Python 示例文件。"
        "请先选择并加载最合适的本地技能，再读取 examples/simple_agent_demo.py 的内容，"
        "重点看看可读性、错误处理和工具使用方式，最后给我审查步骤和几个重点发现。"
    )
    await run_agent_question(agent, question)


async def demo_reload_skills(config: Config) -> None:
    print("=" * 50)
    print("示例 4：通过 Agent 刷新 skills 缓存")
    print("=" * 50)

    agent = build_agent(config)
    question = (
        "我刚更新了本地 skills 目录里的技能内容。"
        "请帮我刷新一下本地技能缓存，让你接下来使用的是最新版本，"
        "并顺便告诉我为什么要这么做。"
    )
    await run_agent_question(agent, question)


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    if not ensure_skills_available():
        return

    if not ensure_demo_targets_available():
        return

    await demo_list_skills(config)
    await demo_load_markdown_skill(config)
    await demo_load_review_skill(config)
    await demo_reload_skills(config)


if __name__ == "__main__":
    asyncio.run(main())
