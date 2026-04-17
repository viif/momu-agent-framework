"""Reflection Agent 实现 — 自我反思与迭代优化的智能体"""

from typing import Any

from ..core.agent import Agent
from ..core.exceptions import AgentException
from ..core.llm import LLM
from ..core.message import Message
from ..tools.registry import ToolRegistry
from ..tools.executor import run_parallel_tools
from ..utils.logger import get_logger
from .parser.tool_parser import ToolParser

DEFAULT_PROMPTS: dict[str, str] = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
""",
}

_NO_IMPROVEMENT_SIGNALS = ("无需改进", "no need for improvement")


class Memory:
    """短期记忆模块，记录执行结果与反思轨迹"""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []
        self.logger = get_logger(__name__)

    def add_record(self, record_type: str, content: str) -> None:
        """添加一条记录，type 为 'execution' 或 'reflection'"""
        self.records.append({"type": record_type, "content": content})
        self.logger.debug(f"🧠 记忆已更新，新增 '{record_type}' 记录")

    def get_last_execution(self) -> str:
        """返回最近一次 execution 记录的内容，无则返回空字符串"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

    def get_trajectory(self) -> str:
        """将所有记录格式化为可读字符串"""
        lines: list[str] = []
        for record in self.records:
            if record["type"] == "execution":
                lines.append(f"--- 上一轮回答 ---\n{record['content']}")
            elif record["type"] == "reflection":
                lines.append(f"--- 反思反馈 ---\n{record['content']}")
        return "\n\n".join(lines)


class ReflectionAgent(Agent):
    """
    Reflection Agent — 自我反思与迭代优化的智能体

    工作流程：
    1. 初始生成：根据任务产出第一版回答
    2. 迭代循环（最多 max_iterations 次）：
       a. 反思：对当前回答进行审查，给出改进建议
       b. 若反馈包含"无需改进"则提前结束循环
       c. 优化：根据反馈生成改进后的回答
    3. 返回最终回答

    适合文档写作、代码生成、分析报告等需要迭代优化的任务。
    支持可选的工具调用，在生成和优化阶段均可调用外部工具。
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str | None = None,
        max_history_length: int = 100,
        max_iterations: int = 3,
        custom_prompts: dict[str, str] | None = None,
        tool_registry: ToolRegistry | None = None,
        max_tool_iterations: int = 3,
    ) -> None:
        """
        初始化 ReflectionAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词（保留供子类扩展）
            max_history_length: 最大对话历史长度
            max_iterations: 最大反思迭代次数（默认 3）
            custom_prompts: 自定义提示词，支持 "initial"、"reflect"、"refine" 三个键
            tool_registry: 工具注册表，用于生成和优化阶段的工具调用
            max_tool_iterations: 每次 LLM 调用中最大工具调用轮次（默认 3）
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            max_history_length=max_history_length,
        )
        self.max_iterations = max_iterations
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations
        self.parser = ToolParser()
        self.memory = Memory()

    async def run(self, input_text: str, **kwargs) -> str:
        """
        运行 ReflectionAgent

        Args:
            input_text: 任务描述
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            最终优化后的回答
        """
        self.logger.info(f"🤖 '{self.name}' 收到任务: {input_text}")
        self.add_message(Message(input_text, "user"))

        # 每次运行重置记忆
        self.memory = Memory()

        try:
            # ── 初始生成 ──────────────────────────────────────────────
            self.logger.info("🤖 正在进行初始生成...")
            initial_result = await self._call_llm(
                self.prompts["initial"].format(task=input_text), **kwargs
            )
            self.memory.add_record("execution", initial_result)

            # ── 反思迭代循环 ──────────────────────────────────────────
            for i in range(1, self.max_iterations + 1):
                self.logger.info(f"🤖 --- 第 {i}/{self.max_iterations} 轮反思 ---")

                # 反思
                last_result = self.memory.get_last_execution()
                feedback = await self._call_llm(
                    self.prompts["reflect"].format(
                        task=input_text, content=last_result
                    ),
                    **kwargs,
                )
                self.memory.add_record("reflection", feedback)
                self.logger.debug(f"🤖 反思反馈: {feedback!r}")

                # 提前退出判断
                if any(sig in feedback.lower() for sig in _NO_IMPROVEMENT_SIGNALS):
                    self.logger.info("🤖 反思认为回答已无需改进，提前结束迭代")
                    break

                # 优化
                self.logger.info(f"🤖 根据反馈优化回答（第 {i} 轮）...")
                refined = await self._call_llm(
                    self.prompts["refine"].format(
                        task=input_text,
                        last_attempt=last_result,
                        feedback=feedback,
                    ),
                    **kwargs,
                )
                self.memory.add_record("execution", refined)

            # ── 收尾 ──────────────────────────────────────────────────
            final_answer = self.memory.get_last_execution()
            self.logger.info(f"🤖 任务完成，最终答案: {final_answer!r}")
            self.add_message(Message(final_answer, "assistant"))
            return final_answer
        except Exception as e:
            error_msg = f"🤖 执行错误: {str(e)}"
            self.logger.error(error_msg)
            self.add_message(Message(error_msg, "assistant"))
            return error_msg

    def _build_tool_system_prompt(self) -> str:
        """构建包含工具信息的系统提示词"""
        if not self.tool_registry:
            return "你是一个有用的AI助手。"
        tools_description = self.tool_registry.get_tools_description()
        return (
            "你是一个有用的AI助手。在完成任务时，你可以使用以下工具：\n\n"
            "<tools_definitions>\n"
            "你必须严格根据以下工具列表来辅助完成任务。如果工具无法解决问题，请直接回答。\n"
            f"工具列表:\n{tools_description}\n"
            "</tools_definitions>" + self.parser.TOOL_CALL_PROTOCOL
        )

    @staticmethod
    def _format_tool_result(res: dict[str, Any]) -> str:
        """将单条工具执行结果格式化为可读字符串"""
        if res.get("status") == "error":
            return (
                f"❌ 工具 {res['tool_name']} 执行失败：{res.get('result', '未知错误')}"
            )
        return f"🔧 工具 {res['tool_name']} 执行结果：\n{res.get('result', '无输出')}"

    async def _invoke_with_tools(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """执行带工具调用循环的 LLM 调用"""
        assert self.tool_registry is not None  # 调用方已确保非 None
        for _ in range(self.max_tool_iterations):
            result = await self.llm.invoke(messages, **kwargs)
            if not result:
                raise AgentException("LLM 未返回有效响应。")
            tool_calls = self.parser.extract_tool_calls(result)

            if not tool_calls:
                return result

            self.logger.info(f"🤖 检测到 {len(tool_calls)} 个工具调用，正在执行...")
            tasks = [
                self.parser.prepare_tool_task(
                    call["tool_name"], call["raw_params"], self.tool_registry
                )
                for call in tool_calls
            ]
            tool_results = await run_parallel_tools(
                registry=self.tool_registry, tasks=tasks, timeout=30.0
            )

            clean_response = self.parser.strip_tool_calls(result, tool_calls)
            messages.append({"role": "assistant", "content": clean_response})
            for i, (call, res) in enumerate(zip(tool_calls, tool_results)):
                messages.append(
                    {
                        "role": "tool",
                        "content": self._format_tool_result(res),
                        "tool_call_id": f"call_{hash(call['tool_name'] + str(i))}",
                    }
                )

        self.logger.warning(
            f"🤖 已达到最大工具调用次数 ({self.max_tool_iterations})，强制终止。"
        )
        raise AgentException(f"已达到最大工具调用次数限制 ({self.max_tool_iterations})")

    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """用单条用户消息调用 LLM（有工具注册表则运行工具调用循环）"""
        if not self.tool_registry:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.invoke(messages, **kwargs)
            if not response:
                raise AgentException("LLM 未返回有效响应。")
            return response

        messages = [
            {"role": "system", "content": self._build_tool_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return await self._invoke_with_tools(messages, **kwargs)
