# CLAUDE.md

本文件旨在为 Claude Code (claude.ai/code) 提供在 `momu-agent-framework` 仓库中工作的指导。

## 项目概述

`momu-agent-framework` 是参考 HelloAgents 框架构建的智能体框架。`momu_agent/` 是主包，当前提供：

- 多种 Agent 实现：`SimpleAgent`、`ReActAgent`、`PlanSolveAgent`、`ReflectionAgent`
- 工具系统：`Tool` 抽象、`ToolRegistry`、并发执行器、顺序工具链
- 内置工具：`calculator`、`search`、`rag`、`memory`、`note`、`skills`、`terminal`
- Skills 能力：本地 `skills/<name>/SKILL.md` 扫描、缓存、按需加载与资源提示
- 记忆系统：`WorkingMemory`、`EpisodicMemory`、`SemanticMemory` 与 `MemoryManager`
- RAG：文档切分、索引、检索、排序与问答聚合
- 上下文工程：`ContextBuilder` 的 Gather-Select-Structure-Compress 流程
- MCP 集成：基于 stdio 的异步 MCP 客户端与远程工具适配
- OpenAI 兼容 LLM 接入

## 包结构

```text
momu_agent/
├── __init__.py
├── agents/                 # Agent 实现
│   ├── simple_agent.py     # 基础工具调用 Agent，支持流式输出
│   ├── react_agent.py      # ReAct Agent
│   ├── plan_solve_agent.py # 规划-执行 Agent
│   ├── reflection_agent.py # 反思迭代 Agent
│   └── parser/
│       └── tool_parser.py  # 工具调用解析器
├── context/
│   └── builder.py          # ContextBuilder（GSSC）
├── core/
│   ├── agent.py            # Agent 抽象基类
│   ├── exceptions.py       # Agent / Tool / LLM 异常
│   ├── llm.py              # OpenAI 兼容 LLM 封装
│   └── message.py          # 消息数据结构
├── mcp/
│   ├── client.py           # MCPClient（stdio 异步客户端）
│   └── tool_adapter.py     # MCPToolAdapter
├── memory/
│   ├── base.py             # Memory 基类与配置
│   ├── working.py          # WorkingMemory
│   ├── episodic.py         # EpisodicMemory
│   ├── semantic.py         # SemanticMemory
│   └── manager.py          # MemoryManager
├── rag/
│   ├── document.py         # 文档解析与切分
│   └── pipeline.py         # 检索与问答流程
├── skills/
│   └── loader.py           # Skill / SkillLoader
├── storage/
│   ├── document.py         # 文档存储
│   ├── vector.py           # 向量存储
│   └── graph.py            # 图存储（Kuzu）
├── tools/
│   ├── base.py             # Tool 抽象与 ToolParameter
│   ├── registry.py         # ToolRegistry
│   ├── chain.py            # ToolChain / ToolChainManager
│   ├── executor.py         # ToolExecutor 与批量/并发执行
│   └── builtin/
│       ├── calculator.py   # 计算器工具
│       ├── memory.py       # 记忆工具
│       ├── note.py         # 结构化笔记工具
│       ├── rag.py          # RAG 工具
│       ├── search.py       # 搜索工具（Tavily / SerpAPI）
│       ├── skills.py       # SkillsTool
│       └── terminal.py     # 安全命令行工具
└── utils/
    ├── config.py           # .env 配置加载
    ├── embedding.py        # 向量嵌入辅助
    └── logger.py           # 日志工具
```

## 示例与测试

当前仓库包含以下示例：

- `examples/simple_agent_demo.py`
- `examples/react_agent_demo.py`
- `examples/plan_solve_agent_demo.py`
- `examples/reflection_agent_demo.py`
- `examples/memory_tool_demo.py`
- `examples/rag_tool_demo.py`
- `examples/context_aware_agent_demo.py`
- `examples/mcp_client_demo.py`
- `examples/skills_demo.py`

`examples/docs4demo/README_v0.2.0.md` 用于 skills 示例中的真实 Markdown 目标文件。

测试覆盖目录包括：`agents`、`context`、`core`、`mcp`、`memory`、`rag`、`skills`、`storage`、`tools`、`utils`。

## 常用命令

本项目使用 `uv` 进行依赖和环境管理。

在 Windows 下通过 Claude Code 运行示例时，若终端默认编码不是 UTF-8，日志中的 emoji 可能触发 `UnicodeEncodeError`。运行示例前请统一附加 UTF-8 前置环境变量：`PYTHONUTF8=1 PYTHONIOENCODING=utf-8`。

```bash
# 开发环境安装
uv sync
uv sync --all-extras

# 按需安装扩展
uv sync --extra search
uv sync --extra memory
uv sync --extra rag
uv sync --extra mcp

# 运行示例（Windows 下建议保留 UTF-8 前置参数）
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/simple_agent_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/react_agent_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/plan_solve_agent_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/reflection_agent_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/memory_tool_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/rag_tool_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/context_aware_agent_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/mcp_client_demo.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python examples/skills_demo.py

# 运行测试
uv run pytest
uv run pytest tests/context/test_builder.py
uv run pytest tests/tools/builtin/test_skills.py

# 检查代码风格（不修改）
uv run ruff check .
uv run ruff format --check .

# 自动修复
uv run ruff check --fix .
uv run ruff format .
```

## 配置

运行前需从 `.env.example` 复制并填写 `.env`：

```bash
cp .env.example .env
```

必填环境变量：

| 变量 | 说明 |
|------|------|
| `LLM_MODEL_ID` | 模型 ID，如 `qwen-turbo` |
| `LLM_API_KEY` | API 密钥 |
| `LLM_BASE_URL` | API 基础 URL（OpenAI 兼容） |

常用可选环境变量：`TEMPERATURE`、`MAX_TOKENS`、`TIMEOUT`、`MAX_HISTORY_LENGTH`、`LOG_LEVEL`、`TAVILY_API_KEY`、`SERPAPI_API_KEY`、`EMBED_MODEL_NAME`。

说明：MCP 本身通常不依赖 `.env` 中的专用变量，而是通过代码中的 stdio 启动命令连接外部 MCP Server。

## Skills

- `SkillsTool` 默认读取当前工作目录下的 `./skills`
- 每个技能目录以 `SKILL.md` 作为入口文件
- `SKILL.md` 需包含 YAML frontmatter，至少声明 `name` 和 `description`
- `SkillLoader` 维护 `metadata_cache` 和 `skills_cache`
- `SkillLoader` 支持 `get_descriptions()`、`get_skill()`、`list_skills()`、`reload()`
- 技能目录可选资源包括 `scripts/`、`examples/`、`references/`
- `SkillsTool` 支持 `action=list|get|reload`
- `get` 需要 `name`，可选 `args` 会替换技能正文中的 `$ARGUMENTS`
- 当前仓库内已有示例技能：`markdown-summary`、`python-review`

## 工具系统补充约定

- `Tool.run()` 为 `async` 抽象方法；新增 Tool 子类需实现 `async def run()`
- `ToolRegistry` 同时支持注册 `Tool` 对象、同步函数、异步函数与 MCP Client
- `ToolExecutor` 提供并发执行能力，`run_parallel_tools` / `run_batch_tool` 为便捷入口
- `ToolChain.execute()` / `ToolChainManager.execute_chain()` 为顺序编排能力
- `TerminalTool` 是白名单式只读命令工具，主要用于仓库探索与文本查看

## 异步约定

- `Agent.run()` 为 `async` 抽象方法，调用时需 `await`
- `SimpleAgent.stream_run()` 为 `async` 流式接口，调用时需 `async for`
- `ToolRegistry.execute_tool()` 为异步执行路径
- `ContextBuilder.build()` 为 `async` 方法
- `MCPClient.connect()`、`list_tools()`、`call_tool()`、`close()` 均为 `async` 方法
- 远程 MCP 工具通过 `ToolRegistry.register_mcp_client()` 注册后，以 `client_name.tool_name` 形式暴露
- 示例入口统一使用 `asyncio.run(main())`
- 测试使用 pytest-asyncio，配置 `asyncio_mode = "auto"`

## 依赖与扩展

- 核心依赖包含 `colorama`、`openai`、`pydantic`、`python-dotenv`、`PyYAML`、`tiktoken`
- 可选扩展包含 `search`、`memory`、`rag`、`mcp`
- `search` 额外依赖 `serpapi`、`tavily-python`
- `memory` 额外依赖 `aiosqlite`、`chromadb`、`kuzu`、`scikit-learn`、`sentence-transformers`、`spacy`
- `rag` 额外依赖 `aiosqlite`、`chromadb`、`markitdown`、`sentence-transformers`
- `mcp` 额外依赖 `mcp`
- 开发依赖包含 `pytest`、`pytest-asyncio`、`ruff`

## 代码风格

使用 Ruff，配置如下：

- 行长度：88
- 目标版本：Python 3.11+
- 启用规则：`E`（pycodestyle 错误）、`F`（pyflakes）、`I`（isort）
- 忽略 `E501`（行过长）
- 字符串使用双引号
- 类型注解使用 Python 3.10+ 语法：`X | None`、`list[...]`、`dict[...]`

## Git 提交规范

提交信息格式：`type: description`

常用 type：
- `feat`：新功能
- `fix`：缺陷修复
- `refactor`：重构（非功能变更、非缺陷修复）
- `test`：测试相关
- `chore`：构建、依赖、CI 等维护性改动
- `style`：代码格式调整（不影响逻辑）
- `docs`：文档变更

## CI

CI 在推送或提交 PR 到 `main` 分支时触发，分两个任务：

- `lint`：Python 3.13 环境下执行 Ruff 检查与格式检查
- `test`：在 Python 3.11、3.12、3.13 上安装全部扩展依赖并运行 pytest（含 sentence-transformers 缓存与 CPU 版 torch 安装）
