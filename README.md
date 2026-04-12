# momu-agent-framework

参考 HelloAgents 实现的智能体框架，提供 Agent、工具系统和 LLM 接入能力。

## 特性

- **全异步设计** — `Agent.run()` 和 `Agent.stream_run()` 均为 `async` 方法，天然支持高并发场景
- **Agent 体系** — 抽象基类 `Agent` + 开箱即用的 `SimpleAgent`，支持多轮对话与历史记录管理
- **工具系统** — `ToolRegistry` 统一管理工具，支持 `Tool` 对象和函数两种注册方式
- **工具链** — `ToolChain` / `ToolChainManager` 支持多工具顺序编排
- **并发执行** — `AsyncToolExecutor` 异步并发调用多个工具
- **流式输出** — `SimpleAgent.stream_run` 支持逐 token 异步流式响应
- **内置工具** — 计算器（`CalculatorTool`）、搜索（Tavily / SerpAPI）
- **OpenAI 兼容** — `LLM` 接入任何兼容 OpenAI 接口的模型

## 安装

需要 Python 3.11+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
git clone https://github.com/your-username/momu-agent-framework.git
cd momu-agent-framework
uv sync --frozen --all-extras
```

## 配置

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

必填项：

```env
LLM_MODEL_ID=qwen-turbo
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 快速上手

### 基础对话

```python
import asyncio
from momu_agent.agents import SimpleAgent
from momu_agent.core.config import Config
from momu_agent.core.llm import LLM

config = Config.from_env()
llm = LLM(
    model=config.model_id,
    api_key=config.api_key,
    base_url=config.base_url,
)

async def main():
    agent = SimpleAgent(name="MyAgent", llm=llm, system_prompt="你是一个有用的助手。")
    response = await agent.run("你好！")
    print(response)

asyncio.run(main())
```

### 流式输出

```python
async def main():
    agent = SimpleAgent(name="MyAgent", llm=llm)
    async for chunk in agent.stream_run("请讲一个小故事。"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### 工具调用

```python
from momu_agent.tools.builtin.calculator import CalculatorTool
from momu_agent.tools.registry import ToolRegistry

async def main():
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = SimpleAgent(
        name="ToolAgent",
        llm=llm,
        system_prompt="你是一个数学助手，遇到计算问题时请使用工具。",
        tool_registry=registry,
    )
    response = await agent.run("sqrt(144) + 2^10 等于多少？")
    print(response)

asyncio.run(main())
```

## 运行示例

```bash
uv run python examples/simple_agent_demo.py
```

## 包结构

```
momu_agent/
├── core/               # 核心模块（Agent基类、Config、LLM、Message）
├── tools/              # 工具系统（Registry、Chain、AsyncExecutor、内置工具）
├── agents/             # Agent实现（SimpleAgent、ToolParser）
└── utils/              # 工具函数（日志）
```

## 开发

```bash
# 运行测试
uv run pytest

# 代码检查
uv run ruff check .
uv run ruff format --check .

# 自动修复
uv run ruff check --fix .
uv run ruff format .
```

CI 在 Python 3.11 / 3.12 / 3.13 上自动运行 lint 和测试。
