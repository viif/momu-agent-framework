"""ReAct Agent实现 - 推理与行动结合的智能体"""

import re
from typing import Any

from ..core.agent import Agent
from ..core.exceptions import AgentException
from ..core.llm import LLM
from ..core.message import Message
from ..tools.registry import ToolRegistry
from ..tools.executor import run_parallel_tools
from ..utils.logger import get_logger
from .parser.tool_parser import ToolParser

DEFAULT_SYSTEM_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 根据思考结果选择行动路径，必须严格遵循以下格式之一：
1. 调用工具：
   - 必须严格遵循以下调用协议，不要输出任何多余字符：
   {tool_protocol}
2. 完成任务：
   - 当你有足够信息得出结论时，使用此格式：`Finish[最终答案]`

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循上述协议
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数"""

DEFAULT_STEP_PROMPT = """## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""


class ReActAgent(Agent):
    """ReAct (Reasoning and Acting) Agent"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str | None = None,
        step_prompt: str | None = None,
        max_history_length: int = 100,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 5,
    ):
        """
        初始化 ReActAgent。

        Args:
            name: Agent 名称。
            llm: 底层 LLM 实例。
            system_prompt: 系统提示词模板；若为 None 则使用默认模板。
            step_prompt: 每轮执行提示词模板；若为 None 则使用默认模板。
            max_history_length: 最大对话历史长度。
            tool_registry: 工具注册表；为 None 时仅进行推理，不执行工具调用。
            max_steps: ReAct 最大迭代步数，超过后将终止并返回提示信息。
        """
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        super().__init__(name, llm, self.system_prompt, max_history_length)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.step_prompt = step_prompt or DEFAULT_STEP_PROMPT
        self.logger = get_logger(__name__)
        self.parser = ToolParser()

    def _build_step_messages(
        self, question: str, history: list[str]
    ) -> list[dict[str, str]]:
        """构建当前步骤的消息列表（system + user）"""
        tools_desc = (
            self.tool_registry.get_tools_description()
            if self.tool_registry
            else "暂无可用工具"
        )
        protocol = self.parser.TOOL_CALL_PROTOCOL if self.tool_registry else ""
        system_content = self.system_prompt.format(
            tools=tools_desc, tool_protocol=protocol
        )
        step_content = self.step_prompt.format(
            question=question, history="\n\n".join(history)
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": step_content},
        ]

    def _format_observations(self, results: list[dict[str, Any]]) -> str:
        """将工具执行结果聚合为 Observation 文本"""
        observations = []
        for res in results:
            tool_name = res["tool_name"]
            if res.get("status") == "error":
                obs = f"❌ 工具 {tool_name} 执行失败：{res.get('result', '未知错误')}"
            else:
                obs = f"✅ 工具 {tool_name} 结果：{res.get('result', '无输出')}"
            observations.append(obs)
        return "\n".join(observations)

    async def run(self, input_text: str, **kwargs) -> str:
        """
        运行 ReAct Agent 主循环（Thought → Action → Observation）。

        Args:
            input_text: 用户输入的问题。
            **kwargs: 透传给 LLM.invoke 的额外参数（如 temperature、max_tokens）。

        Returns:
            当模型输出 Finish[...] 时返回最终答案；
            若达到最大步数或执行中出现异常，返回对应错误/终止信息。
        """
        self.logger.info(f"🤖 {self.name} 开始处理问题: {input_text}")
        self.add_message(Message(input_text, "user"))

        # 记录 ReAct 的逐步历史（Thought / Action / Observation），
        # 每一轮都会回填到下一轮提示中，帮助模型基于上下文持续推理。
        react_history: list[str] = []

        try:
            for step in range(1, self.max_steps + 1):
                self.logger.info(f"🤖 --- 第 {step} 步 / 最大 {self.max_steps} 步 ---")

                # 1) 构建当前轮 messages 并调用 LLM 生成 Thought/Action。
                messages = self._build_step_messages(input_text, react_history)
                response_text = await self.llm.invoke(messages, **kwargs)
                if not response_text:
                    raise AgentException("LLM未能返回有效响应。")

                # 2) 解析 Thought / Action；Action 是后续分支逻辑的核心依据。
                thought, action = self._parse_output(response_text)
                if thought:
                    self.logger.info(f"🤖 思考: {thought}")
                self.logger.info(f"🤖 原始动作: {action}")
                if not action:
                    raise AgentException("无法解析 Action。")

                # 3) 若 Action 为 Finish[...]，说明任务已完成，直接返回最终答案。
                if action.startswith("Finish"):
                    final_answer = self._parse_finish_action(action)
                    self.logger.info(f"🤖 最终答案: {final_answer}")
                    react_history.append(f"Thought: {thought}\nAction: {action}")
                    self.add_message(Message(final_answer, "assistant"))
                    return final_answer

                # 4) 解析工具调用；当动作不是 Finish 时，必须是合法工具调用协议。
                tool_calls = self.parser.extract_tool_calls(action)
                if not tool_calls:
                    raise AgentException(f"无效的工具调用格式: {action}")
                if not self.tool_registry:
                    raise AgentException("工具注册表未初始化")

                self.logger.info(f"🤖 检测到 {len(tool_calls)} 个工具调用，准备执行...")

                # 5) 将工具调用转换为任务后并发执行，提升 I/O 密集场景效率。
                tasks = [
                    self.parser.prepare_tool_task(
                        call["tool_name"], call["raw_params"], self.tool_registry
                    )
                    for call in tool_calls
                ]
                results = await run_parallel_tools(
                    registry=self.tool_registry,
                    tasks=tasks,
                    timeout=30.0,
                )

                # 6) 汇总 Observation 并写入历史，作为下一轮推理输入。
                observation = self._format_observations(results)
                self.logger.info(f"🤖 观察结果:\n{observation}")
                react_history.append(
                    f"Thought: {thought}\nAction: {action}\nObservation:\n{observation}"
                )

            # 若循环结束仍未 Finish，说明在 max_steps 内未收敛到可交付结论。
            final_answer = f"🤖 已达到最大步数 ({self.max_steps})，Agent 未能得出结论。"
            self.logger.warning(final_answer)
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        except Exception as e:
            # 统一异常出口：记录日志、写入会话消息并返回可读错误信息。
            error_msg = f"🤖 执行错误: {str(e)}"
            self.logger.error(error_msg)
            self.add_message(Message(error_msg, "assistant"))
            return error_msg

    def _parse_output(self, text: str) -> tuple[str | None, str | None]:
        thought_match = re.search(
            r"Thought:\s*(.*?)(?:\n\s*Action:|\Z)", text, re.DOTALL
        )
        action_match = re.search(
            r"Action:\s*(.*?)(?:\n\s*Thought:|\Z)", text, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_finish_action(self, action_text: str) -> str:
        match = re.match(r"Finish\s*\[?(.*?)\]?$", action_text.strip(), re.DOTALL)
        return match.group(1).strip() if match else action_text
