"""Plan and Solve Agent 实现 — 规划分解与逐步执行的智能体"""

import ast
from typing import Any

from ..core.agent import Agent
from ..core.exceptions import AgentException
from ..core.llm import LLM
from ..core.message import Message
from ..tools.registry import ToolRegistry
from ..tools.tool_executor import run_parallel_tools
from ..utils.logger import get_logger
from .parser.tool_parser import ToolParser

DEFAULT_PLANNER_PROMPT = """
你是一个顶级的 AI 规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个 Python 列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的 AI 执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class Planner:
    """规划器 — 将复杂问题分解为有序的简单步骤列表"""

    def __init__(self, llm: LLM, prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_PLANNER_PROMPT
        self.logger = get_logger(__name__)

    async def plan(self, question: str, **kwargs) -> list[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            步骤字符串列表

        Raises:
            AgentException: LLM 响应为空或计划解析失败
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        self.logger.info("🤖 正在生成计划...")
        response_text = await self.llm.invoke(messages, **kwargs)
        if not response_text:
            raise AgentException("规划阶段 LLM 未返回有效响应。")
        self.logger.debug(f"🤖 规划器原始响应:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
        except (ValueError, SyntaxError, IndexError) as e:
            self.logger.warning(f"🤖 解析计划失败: {e}，原始响应: {response_text!r}")
            raise AgentException("无法生成有效的行动计划，任务终止。") from e

        if not isinstance(plan, list):
            raise AgentException("无法生成有效的行动计划，任务终止。")
        if not plan:
            raise AgentException("无法生成有效的行动计划，任务终止。")
        if not all(isinstance(step, str) and step.strip() for step in plan):
            raise AgentException("无法生成有效的行动计划，任务终止。")

        self.logger.info(f"🤖 计划生成成功，共 {len(plan)} 个步骤")
        return plan


class Executor:
    """执行器 — 按计划逐步执行并累积上下文，支持可选工具调用"""

    def __init__(
        self,
        llm: LLM,
        prompt_template: str | None = None,
        tool_registry: ToolRegistry | None = None,
        max_tool_iterations: int = 3,
    ):
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_EXECUTOR_PROMPT
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations
        self.parser = ToolParser()
        self.logger = get_logger(__name__)

    def _build_tool_system_prompt(self) -> str:
        """构建包含工具信息的系统提示词"""
        if not self.tool_registry:
            return "你是一位顶级的 AI 执行专家。"
        tools_description = self.tool_registry.get_tools_description()
        return (
            "你是一位顶级的 AI 执行专家。在完成当前步骤时，你可以使用以下工具：\n\n"
            "<tools_definitions>\n"
            "你必须严格根据以下工具列表来辅助完成步骤任务。如果工具无法解决问题，请直接回答。\n"
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

    async def _execute_step_with_tools(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> str:
        """在单步执行中支持工具调用循环"""
        if not self.tool_registry:
            result = await self.llm.invoke(messages, **kwargs)
            if not result:
                raise AgentException("执行阶段 LLM 未返回有效响应。")
            return result

        for _ in range(self.max_tool_iterations):
            result = await self.llm.invoke(messages, **kwargs)
            if not result:
                raise AgentException("执行阶段 LLM 未返回有效响应。")
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
            f"🤖 步骤已达到最大工具调用次数 ({self.max_tool_iterations})，强制终止。"
        )
        raise AgentException(f"已达到最大工具调用次数限制 ({self.max_tool_iterations})")

    async def execute(self, question: str, plan: list[str], **kwargs) -> str:
        """
        按计划逐步执行任务

        Args:
            question: 原始问题
            plan: 执行计划（步骤列表）
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            最后一个步骤的执行结果作为最终答案
        """
        history = ""
        final_answer = ""

        self.logger.info(f"🤖 开始执行计划，共 {len(plan)} 个步骤")
        for i, step in enumerate(plan, 1):
            self.logger.info(f"🤖 执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )

            if self.tool_registry:
                system_prompt = self._build_tool_system_prompt()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                result = await self._execute_step_with_tools(messages, **kwargs)
            else:
                messages = [{"role": "user", "content": prompt}]
                result = await self.llm.invoke(messages, **kwargs)
                if not result:
                    raise AgentException("执行阶段 LLM 未返回有效响应。")

            history += f"步骤 {i}: {step}\n结果: {result}\n\n"
            final_answer = result
            self.logger.debug(f"🤖 步骤 {i} 完成，结果: {result!r}")

        self.logger.info("🤖 所有步骤执行完毕")
        return final_answer


class PlanSolveAgent(Agent):
    """
    Plan and Solve Agent — 规划分解与逐步执行的智能体

    工作流程：
    1. 规划阶段：LLM 将复杂问题分解为有序步骤列表
    2. 执行阶段：LLM 逐步执行每个步骤，并在每步中保留历史上下文

    适合多步骤推理、数学问题、复杂分析等任务。
    不依赖外部工具，完全基于 LLM 推理能力。
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str | None = None,
        max_history_length: int = 100,
        custom_prompts: dict[str, str] | None = None,
        tool_registry: ToolRegistry | None = None,
        max_tool_iterations: int = 3,
    ):
        """
        初始化 PlanSolveAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词（暂未注入到规划/执行提示中，保留供子类扩展）
            max_history_length: 最大对话历史长度
            custom_prompts: 自定义提示词模板，支持 "planner" 和 "executor" 两个键
            tool_registry: 工具注册表，用于执行阶段的工具调用
            max_tool_iterations: 每步执行中最大工具调用轮次
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            max_history_length=max_history_length,
        )

        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None

        self.tool_registry = tool_registry
        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(
            self.llm, executor_prompt, tool_registry, max_tool_iterations
        )

    async def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Plan and Solve Agent

        Args:
            input_text: 用户输入的问题
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            最终答案
        """
        self.logger.info(f"🤖 '{self.name}' 收到问题: {input_text}")
        self.add_message(Message(input_text, "user"))

        try:
            # 规划阶段
            plan = await self.planner.plan(input_text, **kwargs)

            # 执行阶段
            final_answer = await self.executor.execute(input_text, plan, **kwargs)
            self.logger.info(f"🤖 任务完成，最终答案: {final_answer!r}")
            self.add_message(Message(final_answer, "assistant"))
            return final_answer
        except Exception as e:
            error_msg = f"🤖 执行错误: {str(e)}"
            self.logger.error(error_msg)
            self.add_message(Message(error_msg, "assistant"))
            return error_msg
