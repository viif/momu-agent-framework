# CLAUDE.md

本文件旨在为 Claude Code (claude.ai/code) 提供在 `momu-agent-framework` 仓库中工作的指导。

## 项目概述

`momu-agent-framework` 是参考 HelloAgents 框架构建的智能体框架。`momu_agent/` 是主包，提供 Agent、工具系统、记忆系统、存储层、RAG、上下文工程能力和 OpenAI 兼容 LLM 接入能力。

## 包结构

```text
momu_agent/
├── core/               # 核心模块
│   ├── agent.py        # Agent 抽象基类
│   ├── exceptions.py   # 异常定义
│   ├── llm.py          # LLM 封装（OpenAI 兼容接口）
│   └── message.py      # 消息数据结构
├── tools/              # 工具系统
│   ├── base.py         # Tool 抽象基类、ToolParameter（Tool.run 为 async）
│   ├── registry.py     # ToolRegistry（execute_tool 为 async，支持 Tool 对象与函数两种注册方式）
│   ├── chain.py        # ChainStep / ToolChain / ToolChainManager（顺序工具链）
│   ├── executor.py     # 异步并发工具执行器
│   └── builtin/        # 内置工具
│       ├── calculator.py  # 计算器工具
│       ├── search.py      # 搜索工具（Tavily / SerpAPI）
│       ├── rag.py         # RAG 工具
│       ├── memory.py      # 记忆工具
│       ├── note.py        # 结构化笔记工具
│       └── terminal.py    # 安全命令行工具
├── agents/             # Agent 实现
│   ├── simple_agent.py      # SimpleAgent（async，支持工具调用和流式输出）
│   ├── react_agent.py       # ReActAgent（Thought → Action → Observation 循环）
│   ├── plan_solve_agent.py  # PlanSolveAgent（规划分解 → 逐步执行）
│   ├── reflection_agent.py  # ReflectionAgent（初始生成 → 反思迭代 → 最终答案）
│   └── parser/
│       └── tool_parser.py   # 工具调用解析器
├── context/            # 上下文工程
│   └── builder.py      # ContextBuilder（Gather-Select-Structure-Compress）
├── memory/             # 记忆系统
│   ├── base.py         # Memory 基类与配置
│   ├── working.py      # WorkingMemory
│   ├── episodic.py     # EpisodicMemory
│   ├── semantic.py     # SemanticMemory（向量+图谱）
│   └── manager.py      # MemoryManager（多记忆类型统一管理）
├── storage/            # 存储层
│   ├── document.py     # 文档存储
│   ├── vector.py       # 向量存储
│   └── graph.py        # 图存储（Kuzu）
├── rag/                # RAG 管线与文档处理
│   ├── document.py     # 文档与切分
│   └── pipeline.py     # 检索、索引、排序与聚合
└── utils/
    ├── config.py       # 配置管理（从 .env 加载）
    ├── embedding.py    # 向量嵌入辅助
    └── logger.py       # 日志工具
```

## 常用命令

本项目使用 `uv` 进行依赖和环境管理。

```bash
# 开发环境安装
uv sync
uv sync --all-extras

# 按需安装扩展
uv sync --extra search
uv sync --extra memory
uv sync --extra rag

# 运行示例
uv run python examples/simple_agent_demo.py
uv run python examples/react_agent_demo.py
uv run python examples/plan_solve_agent_demo.py
uv run python examples/reflection_agent_demo.py
uv run python examples/rag_tool_demo.py
uv run python examples/memory_tool_demo.py
uv run python examples/context_aware_agent_demo.py

# 运行测试
uv run pytest
uv run pytest tests/agents/test_simple_agent.py
uv run pytest tests/context/test_builder.py

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

可选环境变量：`TEMPERATURE`、`MAX_TOKENS`、`TIMEOUT`、`MAX_HISTORY_LENGTH`、`LOG_LEVEL`、`TAVILY_API_KEY`、`SERPAPI_API_KEY`、`EMBED_MODEL_NAME`。

## 依赖与扩展

- 核心依赖包含 `colorama`、`openai`、`pydantic`、`python-dotenv`、`tiktoken`
- 可选扩展包含 `search`、`memory`、`rag`
- `memory` 额外依赖 `aiosqlite`、`chromadb`、`kuzu`、`scikit-learn`、`sentence-transformers`、`spacy`
- `rag` 额外依赖 `aiosqlite`、`chromadb`、`markitdown`、`sentence-transformers`

## 异步约定

- `Agent.run()` 为 `async` 抽象方法，调用时需 `await`
- `SimpleAgent.stream_run()` 为 `async` 流式接口，调用时需 `async for`
- `Tool.run()` 为 `async` 抽象方法；新增 Tool 子类需实现 `async def run()`
- `ToolRegistry.execute_tool()` 为 `async` 方法；Tool 对象调用与函数工具调用统一走异步执行路径
- `ToolChain.execute()` / `ToolChainManager.execute_chain()` 为 `async` 方法
- `ContextBuilder.build()` 为 `async` 方法
- 示例和脚本入口使用 `asyncio.run(main())`
- 测试使用 pytest-asyncio，配置 `asyncio_mode = "auto"`，async 测试函数无需额外标注

## 代码风格

使用 Ruff，配置如下：
- 行长度：88
- 目标版本：Python 3.11+
- 启用规则：`E`（pycodestyle 错误）、`F`（pyflakes）、`I`（isort）
- 忽略 `E501`（行过长）
- 字符串使用双引号
- 类型注解使用 Python 3.10+ 语法：`X | None`、`list[...]`、`dict[...]`，不使用 `typing.Optional/List/Dict`

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

CI 在推送/PR 到 `main` 分支时触发，分两个任务：
- `lint`：Python 3.13 环境下执行 Ruff 检查与格式检查
- `test`：在 Python 3.11、3.12、3.13 上安装全部扩展依赖并运行 pytest（含 sentence-transformers 缓存与 CPU 版 torch 安装）
