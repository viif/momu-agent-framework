# momu-agent-framework

参考 HelloAgents 实现的智能体框架，提供多种 Agent、统一工具系统、Skills、本地记忆系统、RAG、上下文工程、MCP 集成与 OpenAI 兼容 LLM 接入能力。

## ✨ 特性

- **全异步主路径**：`Agent.run()`、工具执行、工具链、上下文构建、MCP 客户端均为异步接口
- **多种 Agent 实现**：`SimpleAgent`、`ReActAgent`、`PlanSolveAgent`、`ReflectionAgent`
- **统一工具注册**：`ToolRegistry` 支持 `Tool` 对象、同步函数、异步函数与 MCP 远程工具
- **顺序与并发执行**：`ToolChain` / `ToolChainManager` 负责顺序编排，`ToolExecutor` 提供并发执行
- **流式输出**：`SimpleAgent.stream_run()` 支持逐段异步输出
- **内置工具**：`CalculatorTool`、`SearchTool`、`RAGTool`、`MemoryTool`、`NoteTool`、`SkillsTool`、`TerminalTool`
- **Skills 能力**：从本地 `skills/<name>/SKILL.md` 扫描、缓存、按需加载技能，并附带资源提示
- **记忆系统**：统一管理 `WorkingMemory`、`EpisodicMemory`、`SemanticMemory`
- **RAG 能力**：支持文档切分、索引、检索、排序与基于上下文问答
- **上下文工程**：`ContextBuilder` 实现 Gather-Select-Structure-Compress（GSSC）流程
- **MCP 集成**：`MCPClient` 可通过 stdio 连接 MCP Server，并将远程工具注册到 Agent
- **OpenAI 兼容**：`LLM` 可接入任何兼容 OpenAI 接口的模型服务
- **顶层导入**：`momu_agent` 顶层导出常用 Agent、`LLM`、`Config`、工具注册器、工具执行器、计算器、搜索、RAG 与 Skills 工具

## 📦 安装

需要 Python 3.11+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

### 基础安装

```bash
uv sync
```

### 按需安装扩展

```bash
uv sync --extra search
uv sync --extra memory
uv sync --extra rag
uv sync --extra mcp
```

### 安装全部扩展

```bash
uv sync --all-extras
```

## ⚙️ 配置

复制 `.env.example` 为 `.env` 并填写模型配置：

```bash
cp .env.example .env
```

最小配置示例：

```env
LLM_MODEL_ID=qwen-turbo
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

常用可选项：`TEMPERATURE`、`MAX_TOKENS`、`TIMEOUT`、`MAX_HISTORY_LENGTH`、`LOG_LEVEL`、`TAVILY_API_KEY`、`SERPAPI_API_KEY`、`EMBED_MODEL_NAME`。

## 🚀 快速上手

```python
import asyncio

from momu_agent.agents import SimpleAgent
from momu_agent.core.llm import LLM
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


async def main() -> None:
    config = Config.from_env()
    setup_logger(level=config.log_level)

    agent = SimpleAgent(
        name="Support",
        llm=build_llm(config),
        system_prompt="你是一名 Python 技术支持助手，回答简洁准确。",
        max_history_length=config.max_history_length,
    )

    print(await agent.run("什么是 GIL？"))
    async for chunk in agent.stream_run("介绍 Python 异步编程的最佳实践。"):
        print(chunk, end="", flush=True)


asyncio.run(main())
```

## 🧩 Skills 使用

`SkillsTool` 默认从当前工作目录下的 `./skills` 读取技能。每个技能使用独立目录，入口文件为 `SKILL.md`，需要包含 YAML frontmatter，至少声明 `name` 和 `description`。

```text
skills/
└── pdf/
    ├── SKILL.md
    ├── scripts/
    ├── examples/
    └── references/
```

示例：

```md
---
name: pdf
description: 处理 PDF 文件
---

请根据用户输入完成 PDF 解析与信息提取。
```

`SkillsTool` 支持以下动作：

- `list`：列出全部技能描述
- `get`：加载指定技能内容，参数 `name` 必填，`args` 可选，会替换技能正文中的 `$ARGUMENTS`
- `reload`：重新扫描 `skills/` 目录并刷新缓存

当前仓库自带两个示例技能：

- `skills/pdf/SKILL.md`
- `skills/code-review/SKILL.md`

并包含对应示例脚本：`examples/skills_demo.py`。

## 🧰 内置工具

- `CalculatorTool`：安全表达式计算，另提供 `calculate` 便捷函数
- `SearchTool`：Tavily / SerpAPI 搜索封装，另提供 `search` 便捷函数
- `RAGTool`：文档添加、文本添加、检索、问答与统计，另提供 `rag_add_document`、`rag_add_text`、`rag_search`、`rag_ask`、`rag_get_stats`
- `MemoryTool`：记忆添加、检索与统计，另提供 `memory_add`、`memory_search`、`memory_get_stats`
- `NoteTool`：结构化笔记工具
- `SkillsTool`：本地技能发现、加载与缓存刷新
- `TerminalTool`：白名单式命令行工具，限制在工作目录内，适合仓库探索与文本查看

## ▶️ 运行示例

```bash
uv run python examples/simple_agent_demo.py
uv run python examples/react_agent_demo.py
uv run python examples/plan_solve_agent_demo.py
uv run python examples/reflection_agent_demo.py
uv run python examples/memory_tool_demo.py
uv run python examples/rag_tool_demo.py
uv run python examples/context_aware_agent_demo.py
uv run python examples/mcp_client_demo.py
uv run python examples/skills_demo.py
```

## 🗂️ 目录结构

```text
momu_agent/
├── agents/             # Simple/ReAct/PlanSolve/Reflection
│   └── parser/         # 工具调用解析器
├── context/            # ContextBuilder（GSSC）
├── core/               # Agent 抽象、LLM、消息、异常
├── mcp/                # MCPClient / MCPToolAdapter
├── memory/             # working / episodic / semantic / manager
├── rag/                # 文档处理与检索管线
├── skills/             # Skill / SkillLoader
├── storage/            # document / vector / graph 存储
├── tools/              # Tool 抽象、Registry、Chain、Executor
│   └── builtin/        # calculator / memory / note / rag / search / skills / terminal
└── utils/              # config / embedding / logger
```

## 🛠️ 开发

```bash
uv run pytest
uv run pytest tests/context/test_builder.py
uv run pytest tests/tools/builtin/test_skills.py
uv run ruff check .
uv run ruff format --check .
uv run ruff check --fix .
uv run ruff format .
```

测试覆盖目录包括：`agents`、`context`、`core`、`mcp`、`memory`、`rag`、`skills`、`storage`、`tools`、`utils`。

CI 在 `main` 分支的 push/PR 上运行：

- `lint`：Python 3.13 + Ruff 检查
- `test`：Python 3.11 / 3.12 / 3.13 安装全部扩展依赖并运行 pytest（含 sentence-transformers 缓存与 CPU 版 torch 安装）

## 📚 参考

本项目参考了 [HelloAgents](https://github.com/jjyaoao/HelloAgents) 的设计思想与实现架构。

## 🙏 致谢

感谢 Datawhale 提供的 [《HelloAgents》](https://github.com/datawhalechina/hello-agents) 教程。
