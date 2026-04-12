"""Plan and Solve Agent 实现 — 规划分解与逐步执行的智能体"""

import ast

from ..core.agent import Agent
from ..core.llm import LLM
from ..core.message import Message
from ..utils.logger import get_logger

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
            步骤字符串列表；解析失败时返回空列表
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        self.logger.info("正在生成计划...")
        response_text = await self.llm.invoke(messages, **kwargs) or ""
        self.logger.debug(f"规划器原始响应:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            if isinstance(plan, list):
                self.logger.info(f"计划生成成功，共 {len(plan)} 个步骤")
                return plan
            self.logger.warning("解析结果不是列表，返回空计划")
            return []
        except (ValueError, SyntaxError, IndexError) as e:
            self.logger.warning(f"解析计划失败: {e}，原始响应: {response_text!r}")
            return []


class Executor:
    """执行器 — 按计划逐步执行并累积上下文"""

    def __init__(self, llm: LLM, prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_EXECUTOR_PROMPT
        self.logger = get_logger(__name__)

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

        self.logger.info(f"开始执行计划，共 {len(plan)} 个步骤")
        for i, step in enumerate(plan, 1):
            self.logger.info(f"执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]

            result = await self.llm.invoke(messages, **kwargs) or ""
            history += f"步骤 {i}: {step}\n结果: {result}\n\n"
            final_answer = result
            self.logger.debug(f"步骤 {i} 完成，结果: {result!r}")

        self.logger.info("所有步骤执行完毕")
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
    ):
        """
        初始化 PlanSolveAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词（暂未注入到规划/执行提示中，保留供子类扩展）
            max_history_length: 最大对话历史长度
            custom_prompts: 自定义提示词模板，支持 "planner" 和 "executor" 两个键
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            max_history_length=max_history_length,
        )

        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    async def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Plan and Solve Agent

        Args:
            input_text: 用户输入的问题
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            最终答案
        """
        self.logger.info(f"'{self.name}' 收到问题: {input_text}")

        # 规划阶段
        plan = await self.planner.plan(input_text, **kwargs)
        if not plan:
            error_msg = "无法生成有效的行动计划，任务终止。"
            self.logger.warning(error_msg)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(error_msg, "assistant"))
            return error_msg

        # 执行阶段
        final_answer = await self.executor.execute(input_text, plan, **kwargs)
        self.logger.info(f"任务完成，最终答案: {final_answer!r}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
