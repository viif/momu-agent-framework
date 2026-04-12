# momu-agent-framework

参考 HelloAgents 实现的智能体框架，提供 Agent、工具系统和 LLM 接入能力。

## 特性

- **全异步设计** — `Agent.run()` 为 `async` 方法，天然支持高并发场景
- **Agent 体系** — 抽象基类 `Agent` + 四种开箱即用的 Agent：`SimpleAgent`（多轮对话/流式输出）、`ReActAgent`（Thought→Action→Observation 循环）、`PlanSolveAgent`（规划分解→逐步执行）、`ReflectionAgent`（生成→反思迭代→精炼）
- **工具系统** — `ToolRegistry` 统一管理工具，支持 `Tool` 对象和函数两种注册方式
- **工具链** — `ToolChain` / `ToolChainManager` 支持多工具顺序编排
- **并发执行** — `AsyncToolExecutor` 异步并发调用多个工具
- **流式输出** — `SimpleAgent.stream_run` 支持逐 token 异步流式响应
- **内置工具** — 计算器（支持四则运算、幂运算、`sqrt/sin/cos/log` 等数学函数）、联网搜索（Tavily / SerpAPI）
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

所有示例共用以下初始化代码：

```python
import asyncio
from momu_agent.core.config import Config
from momu_agent.core.llm import LLM
from momu_agent.utils.logger import setup_logger

config = Config.from_env()
setup_logger(level=config.log_level)
llm = LLM(model=config.model_id, api_key=config.api_key, base_url=config.base_url)
```

### SimpleAgent — 多轮对话与流式输出

```python
from momu_agent.agents import SimpleAgent

async def main():
    agent = SimpleAgent(name="Assistant", llm=llm, system_prompt="你是一个有用的助手。")

    # 普通调用
    print(await agent.run("你好！"))

    # 流式输出
    async for chunk in agent.stream_run("请讲一个小故事。"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### ReActAgent — Thought → Action → Observation 推理循环

```python
from momu_agent.agents import ReActAgent
from momu_agent.tools.builtin.calculator import CalculatorTool
from momu_agent.tools.registry import ToolRegistry

async def main():
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(name="ReAct", llm=llm, tool_registry=registry, max_steps=5)
    print(await agent.run("(2^10 + sqrt(144)) * 3 等于多少？"))

asyncio.run(main())
```

### PlanSolveAgent — 规划分解 → 逐步执行

```python
from momu_agent.agents import PlanSolveAgent

async def main():
    agent = PlanSolveAgent(name="Planner", llm=llm)
    print(await agent.run("分析大语言模型的主要应用场景及其局限性"))

asyncio.run(main())
```

### ReflectionAgent — 初始生成 → 反思迭代 → 精炼

```python
from momu_agent.agents import ReflectionAgent

async def main():
    agent = ReflectionAgent(name="Reflector", llm=llm, max_iterations=2)
    print(await agent.run("写一段介绍量子计算的文字，要求通俗易懂"))

asyncio.run(main())
```

## 运行更多示例

```bash
uv run python examples/simple_agent_demo.py
uv run python examples/react_agent_demo.py
uv run python examples/plan_solve_agent_demo.py
uv run python examples/reflection_agent_demo.py
```

## 包结构

```
momu_agent/
├── core/               # 核心模块（Agent基类、Config、LLM、Message）
├── tools/              # 工具系统（Registry、Chain、AsyncExecutor、内置工具）
├── agents/             # Agent实现（SimpleAgent、ReActAgent、PlanSolveAgent、ReflectionAgent）
│   └── parser/         # 工具调用解析器
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
