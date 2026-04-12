# CLAUDE.md

本文件旨在为 Claude Code (claude.ai/code) 提供在 `momu-agent-framework` 仓库中工作的指导。

## 项目概述

`momu-agent-framework` 是参考 HelloAgents 框架构建的智能体框架。`momu_agent/` 是主包，提供了完整的 Agent、工具系统和 LLM 接入能力。

## 包结构

```
momu_agent/
├── core/               # 核心模块
│   ├── agent.py        # Agent 抽象基类
│   ├── config.py       # 配置管理（从 .env 加载）
│   ├── exceptions.py   # 异常定义
│   ├── llm.py          # LLM 封装（OpenAI 兼容接口）
│   └── message.py      # 消息数据结构
├── tools/              # 工具系统
│   ├── base.py         # Tool 抽象基类、ToolParameter
│   ├── registry.py     # ToolRegistry（支持 Tool 对象和函数两种注册方式）
│   ├── chain.py        # ToolChain / ToolChainManager（顺序工具链）
│   ├── async_executor.py # 异步并发工具执行器
│   └── builtin/        # 内置工具
│       ├── calculator.py  # 计算器工具
│       └── search.py      # 搜索工具（Tavily / SerpAPI）
├── agents/             # Agent 实现
│   ├── simple_agent.py # SimpleAgent（支持工具调用和流式输出）
│   └── parser/
│       └── tool_parser.py # 工具调用解析器
└── utils/
    └── logger.py       # 日志工具
```

## 常用命令

本项目使用 `uv` 进行依赖和环境管理。

```bash
# 安装依赖
uv sync --frozen --all-extras

# 运行示例
uv run python examples/simple_agent_demo.py

# 运行所有测试
uv run pytest

# 运行单个测试
uv run pytest tests/agents/test_simple_agent.py

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

可选环境变量：`TEMPERATURE`、`MAX_TOKENS`、`TIMEOUT`、`MAX_HISTORY_LENGTH`、`LOG_LEVEL`、`TAVILY_API_KEY`、`SERPAPI_API_KEY`。

## 代码风格

使用 Ruff，配置如下：
- 行长度：88
- 目标版本：Python 3.11+
- 启用规则：`E`（pycodestyle 错误）、`F`（pyflakes）、`I`（isort）
- 忽略 `E501`（行过长）
- 字符串使用双引号

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
- `lint`：检查 Ruff 格式和风格
- `test`：在 Python 3.11、3.12、3.13 上运行 pytest
