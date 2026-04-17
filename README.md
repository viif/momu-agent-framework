# momu-agent-framework

参考 HelloAgents 实现的智能体框架，提供 Agent、工具系统和 LLM 接入能力。

## 🚀 特性

- **全异步设计** — `Agent.run()` 与工具执行主路径均为 `async`，天然支持高并发场景
- **Agent 体系** — 抽象基类 `Agent` + 四种开箱即用的 Agent：`SimpleAgent`（多轮对话/流式输出）、`ReActAgent`（Thought→Action→Observation 循环）、`PlanSolveAgent`（规划分解→逐步执行）、`ReflectionAgent`（生成→反思迭代→精炼）
- **工具系统** — `ToolRegistry` 统一管理工具，支持 `Tool` 对象、同步函数和异步函数注册，工具执行入口为异步
- **工具链** — `ToolChain` / `ToolChainManager` 支持多工具顺序编排（异步执行）
- **并发执行** — `ToolExecutor` 异步并发调用多个工具（支持超时控制与结果聚合）
- **流式输出** — `SimpleAgent.stream_run` 支持逐 token 异步流式响应
- **内置工具** — 计算器（支持四则运算、幂运算、`sqrt/sin/cos/log` 等数学函数）、联网搜索（Tavily / SerpAPI）
- **OpenAI 兼容** — `LLM` 接入任何兼容 OpenAI 接口的模型

## ⬇️ 安装

需要 Python 3.11+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

### 基础功能

仅安装核心依赖：

```bash
uv sync
```

### 拓展功能

按需安装不同扩展功能：

- 搜索能力（`search`）：

```bash
uv sync --extra search
```

- 记忆能力（`memory`）：

```bash
uv sync --extra memory
```


## ⚙️ 配置

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

## ✨ 快速上手

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

### SimpleAgent — 多轮上下文对话

适合需要记忆上下文的连续问答场景，如客服、教学辅导。

```python
from momu_agent.agents import SimpleAgent

async def main():
    agent = SimpleAgent(
        name="Support",
        llm=llm,
        system_prompt="你是一名 Python 技术支持助手，回答简洁准确。",
    )
    # 多轮对话，Agent 自动保留历史上下文
    print(await agent.run("什么是 GIL？"))
    print(await agent.run("它对多线程爬虫有什么具体影响？"))  # 能理解上文"它"

    # 流式输出，适合长文本场景
    async for chunk in agent.stream_run("用一句话总结以上内容。"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### ReActAgent — 工具辅助的逐步推理

适合需要借助外部工具分步推导的问题，如数学计算、事实核查。Agent 通过 Thought → Action → Observation 循环完成推理。

```python
from momu_agent.agents import ReActAgent
from momu_agent.tools.builtin.calculator import CalculatorTool
from momu_agent.tools.registry import ToolRegistry

async def main():
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(name="MathSolver", llm=llm, tool_registry=registry)
    # Agent 先推导需要哪些计算，再调用计算器，最后得出答案
    print(await agent.run("正方形面积为 144，求其对角线长度（保留两位小数）"))

asyncio.run(main())
```

### PlanSolveAgent — 规划分解复杂任务

适合目标明确但步骤繁多的任务，如方案设计、报告撰写。Agent 先生成结构化计划，再逐步执行，步骤间共享上下文。

```python
from momu_agent.agents import PlanSolveAgent

async def main():
    agent = PlanSolveAgent(name="Planner", llm=llm)
    print(await agent.run(
        "为一个初创电商网站制定 SEO 优化方案，"
        "涵盖现状诊断、关键词策略、内容优化、技术改进四个方面"
    ))

asyncio.run(main())
```

### ReflectionAgent — 迭代优化高质量输出

适合对输出质量要求高的创作任务，如文档写作、代码生成。Agent 先产出初稿，再自我审查并改进，直到满意为止。

```python
from momu_agent.agents import ReflectionAgent

async def main():
    agent = ReflectionAgent(name="Coder", llm=llm, max_iterations=2)
    # Agent 先生成初版代码，自我审查找出边界条件和可读性问题，再产出改进版
    print(await agent.run(
        "用 Python 实现一个二分查找函数，要求处理边界条件并附带类型注解"
    ))

asyncio.run(main())
```

## ▶️ 运行示例

```bash
uv run python examples/simple_agent_demo.py
uv run python examples/react_agent_demo.py
uv run python examples/plan_solve_agent_demo.py
uv run python examples/reflection_agent_demo.py
```

## 📂 目录结构

```
momu_agent/
├── core/               # 核心模块（Agent基类、Config、LLM、Message）
├── tools/              # 工具系统（Registry、Chain、AsyncExecutor、内置工具）
├── agents/             # Agent实现（SimpleAgent、ReActAgent、PlanSolveAgent、ReflectionAgent）
│   └── parser/         # 工具调用解析器
└── utils/              # 工具函数（日志）
```

## 💻 开发

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

## 📚 参考

本项目参考了 [HelloAgents](https://github.com/jjyaoao/HelloAgents) 的设计思想与实现架构。

## 🙏 致谢

感谢 Datawhale 提供的[《HelloAgents》](https://github.com/datawhalechina/hello-agents)教程。
