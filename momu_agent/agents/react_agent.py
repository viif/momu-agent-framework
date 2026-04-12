"""ReAct Agent实现 - 推理与行动结合的智能体"""

import re
from typing import Any

from ..core.agent import Agent
from ..core.exceptions import AgentException
from ..core.llm import LLM
from ..core.message import Message
from ..tools.async_executor import run_parallel_tools
from ..tools.registry import ToolRegistry
from ..utils.logger import get_logger
from .parser.tool_parser import ToolParser

# 定义 System Prompt（用于定义通用规则）
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

# 定义 Step Prompt（用于每次循环，包含历史和当前任务）
DEFAULT_STEP_PROMPT = """## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""


class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent
    """

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
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        super().__init__(name, llm, self.system_prompt, max_history_length)

        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.step_prompt = step_prompt if step_prompt else DEFAULT_STEP_PROMPT
        self.logger = get_logger(__name__)
        self.parser = ToolParser()

    async def run(self, input_text: str, **kwargs) -> str:
        self.logger.info(f"🤖 {self.name} 开始处理问题: {input_text}")
        self.add_message(Message(input_text, "user"))

        react_history: list[str] = []
        current_step = 0

        try:
            while current_step < self.max_steps:
                current_step += 1
                self.logger.info(
                    f"🤖 --- 第 {current_step} 步 / 最大 {self.max_steps} 步 ---"
                )

                # 1. 构建增强的 System Prompt
                tools_desc = (
                    self.tool_registry.get_tools_description()
                    if self.tool_registry
                    else "暂无可用工具"
                )
                protocol_content = (
                    self.parser.TOOL_CALL_PROTOCOL if self.tool_registry else ""
                )

                final_system_content = self.system_prompt.format(
                    tools=tools_desc, tool_protocol=protocol_content
                )

                # 2. 构建 Step Prompt
                history_str = "\n\n".join(react_history)
                step_prompt = self.step_prompt.format(
                    question=input_text, history=history_str
                )

                # 3. 准备消息列表
                messages = [{"role": "system", "content": final_system_content}]
                messages.append({"role": "user", "content": step_prompt})

                # 4. 调用LLM
                response_text = await self.llm.invoke(messages, **kwargs)

                if not response_text:
                    self.logger.error("🤖 LLM未能返回有效响应。")
                    raise AgentException("LLM未能返回有效响应。")

                # 5. 解析输出
                thought, action = self._parse_output(response_text)

                if thought:
                    self.logger.info(f"🤖 思考: {thought}")
                self.logger.info(f"🤖 原始动作: {action}")

                if not action:
                    self.logger.error("🤖 无法解析 Action。")
                    raise AgentException("无法解析 Action。")

                # 6. 检查是否完成
                if action.startswith("Finish"):
                    final_answer = self._parse_finish_action(action)
                    self.logger.info(f"🤖 最终答案: {final_answer}")

                    react_history.append(f"Thought: {thought}\nAction: {action}")

                    self.add_message(Message(final_answer, "assistant"))
                    return final_answer

                # 7. 使用 ToolParser 解析工具调用
                tool_calls = self.parser.extract_tool_calls(action)

                if not tool_calls:
                    self.logger.error(f"🤖 无效的工具调用格式: {action}")
                    raise AgentException(f"无效的工具调用格式: {action}")

                assert self.tool_registry is not None, "工具注册表未初始化"

                self.logger.info(f"🤖 检测到 {len(tool_calls)} 个工具调用，准备执行...")

                # 6. 准备任务列表
                tasks = []
                for call in tool_calls:
                    task = self._prepare_tool_task(
                        call["tool_name"], call["raw_params"]
                    )
                    tasks.append(task)

                # 7. 并发执行所有工具
                results = await run_parallel_tools(
                    registry=self.tool_registry,
                    tasks=tasks,
                    timeout=30.0,
                )

                # 8. 聚合观察结果
                observations = []
                for i, res in enumerate(results):
                    tool_name = tasks[i]["tool_name"]
                    if res.get("status") == "error":
                        obs_text = f"❌ 工具 {tool_name} 执行失败：{res.get('result', '未知错误')}"
                    else:
                        obs_text = (
                            f"✅ 工具 {tool_name} 结果：{res.get('result', '无输出')}"
                        )
                    observations.append(obs_text)

                # 用换行符拼接所有观察结果
                observation = "\n".join(observations)
                self.logger.info(f"🤖 👀 观察结果:\n{observation}")

                # 9. 更新历史记录
                # 将本轮的 Thought, Action 和 聚合后的 Observation 加入历史
                history_entry = (
                    f"Thought: {thought}\nAction: {action}\nObservation:\n{observation}"
                )
                react_history.append(history_entry)

            # 达到最大步数
            final_answer = (
                f"🤖 ⚠️ 已达到最大步数 ({self.max_steps})，Agent 未能得出结论。"
            )
            self.logger.warning(f"{final_answer}")
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        except Exception as e:
            error_msg = f"🤖 执行错误: {str(e)}"
            self.logger.error(f"🤖 {error_msg}")
            final_answer = error_msg
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

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

    def _prepare_tool_task(self, tool_name: str, raw_parameters: str) -> dict[str, Any]:
        if not self.tool_registry:
            return {"tool_name": tool_name, "error": "未配置工具注册表"}

        try:
            tool_obj = self.tool_registry.get_tool(tool_name)
            if not tool_obj:
                return {"tool_name": tool_name, "error": "工具未注册"}

            parsed_params = self.parser.parse_typed_parameters(
                tool_name, raw_parameters, tool_obj
            )
            return {"tool_name": tool_name, "input_data": parsed_params}
        except Exception as e:
            return {"tool_name": tool_name, "error": str(e)}
