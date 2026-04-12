"""
SimpleAgent 使用示例

演示三种使用模式：
1. 基础对话（无工具）
2. 带工具调用的对话（注册计算器工具）
3. 流式输出

运行前请先复制 .env.example 为 .env 并填写相关配置：
  cp .env.example .env
"""

from momu_agent.agents import SimpleAgent
from momu_agent.core.config import Config
from momu_agent.core.llm import MomuAgentLLM
from momu_agent.tools.builtin.calculator import CalculatorTool
from momu_agent.tools.registry import ToolRegistry
from momu_agent.utils.logger import setup_logger


def build_llm(config: Config) -> MomuAgentLLM:
    return MomuAgentLLM(
        model=config.model_id,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=int(config.timeout),
    )


# ---------- 示例 1：基础对话（无工具） ----------
def demo_basic_chat(config: Config):
    print("=" * 50)
    print("示例 1：基础对话（无工具）")
    print("=" * 50)

    agent = SimpleAgent(
        name="BasicAgent",
        llm=build_llm(config),
        system_prompt="你是一个简洁、友好的 AI 助手，回答请保持简短。",
        max_history_length=config.max_history_length,
    )

    response = agent.run("你好！请用一句话介绍一下你自己。")
    print(f"Agent: {response}\n")

    # 多轮对话：历史记录会自动保留
    response = agent.run("上一个问题我问了什么？")
    print(f"Agent: {response}\n")


# ---------- 示例 2：带工具调用的对话 ----------
def demo_with_tools(config: Config):
    print("=" * 50)
    print("示例 2：带工具调用（计算器）")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = SimpleAgent(
        name="ToolAgent",
        llm=build_llm(config),
        system_prompt="你是一个数学助手，遇到计算问题时请使用工具。",
        tool_registry=registry,
        max_history_length=config.max_history_length,
    )

    questions = [
        "请计算 sqrt(144) + 2^10 的结果",
        "sin(pi/6) 等于多少？",
    ]
    for q in questions:
        print(f"用户: {q}")
        response = agent.run(q, max_tool_iterations=3)
        print(f"Agent: {response}\n")


# ---------- 示例 3：流式输出 ----------
def demo_stream(config: Config):
    print("=" * 50)
    print("示例 3：流式输出")
    print("=" * 50)

    agent = SimpleAgent(
        name="StreamAgent",
        llm=build_llm(config),
        system_prompt="你是一个讲故事的 AI，语言生动有趣。",
        max_history_length=config.max_history_length,
    )

    print("用户: 请用三句话讲一个关于机器人的小故事。")
    print("Agent: ", end="", flush=True)
    for chunk in agent.stream_run("请用三句话讲一个关于机器人的小故事。"):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # 从项目根目录的 .env 文件加载配置
    config = Config.from_env()

    # 按配置初始化日志级别
    setup_logger(level=config.log_level)

    demo_basic_chat(config)
    demo_with_tools(config)
    demo_stream(config)
