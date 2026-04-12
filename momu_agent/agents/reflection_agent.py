"""Reflection Agent 实现 — 自我反思与迭代优化的智能体"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..core.message import Message
from ..utils.logger import get_logger

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
        self.logger.debug(f"记忆已更新，新增 '{record_type}' 记录")

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
    不依赖外部工具，完全基于 LLM 推理能力。
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str | None = None,
        max_history_length: int = 100,
        max_iterations: int = 3,
        custom_prompts: dict[str, str] | None = None,
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
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            max_history_length=max_history_length,
        )
        self.max_iterations = max_iterations
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
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
        self.logger.info(f"'{self.name}' 收到任务: {input_text}")

        # 每次运行重置记忆
        self.memory = Memory()

        # ── 初始生成 ──────────────────────────────────────────────
        self.logger.info("正在进行初始生成...")
        initial_result = await self._call_llm(
            self.prompts["initial"].format(task=input_text), **kwargs
        )
        self.memory.add_record("execution", initial_result)

        # ── 反思迭代循环 ──────────────────────────────────────────
        for i in range(1, self.max_iterations + 1):
            self.logger.info(f"--- 第 {i}/{self.max_iterations} 轮反思 ---")

            # 反思
            last_result = self.memory.get_last_execution()
            feedback = await self._call_llm(
                self.prompts["reflect"].format(task=input_text, content=last_result),
                **kwargs,
            )
            self.memory.add_record("reflection", feedback)
            self.logger.debug(f"反思反馈: {feedback!r}")

            # 提前退出判断
            if any(sig in feedback.lower() for sig in _NO_IMPROVEMENT_SIGNALS):
                self.logger.info("反思认为回答已无需改进，提前结束迭代")
                break

            # 优化
            self.logger.info(f"根据反馈优化回答（第 {i} 轮）...")
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
        self.logger.info(f"任务完成，最终答案: {final_answer!r}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """用单条用户消息调用 LLM，返回完整响应"""
        messages = [{"role": "user", "content": prompt}]
        return await self.llm.invoke(messages, **kwargs) or ""
