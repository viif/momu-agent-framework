"""简单Agent实现"""

from typing import Any, AsyncIterator

from ..core.agent import Agent
from ..core.exceptions import AgentException
from ..core.llm import LLM
from ..core.message import Message
from ..tools.registry import ToolRegistry
from ..tools.executor import run_parallel_tools
from ..utils.logger import get_logger
from .parser.tool_parser import ToolParser


class SimpleAgent(Agent):
    """简单的对话Agent，支持可选的工具调用"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str | None = None,
        max_history_length: int = 100,
        tool_registry: ToolRegistry | None = None,
    ):
        super().__init__(name, llm, system_prompt, max_history_length)
        self.tool_registry = tool_registry
        self.logger = get_logger(__name__)
        self.parser = ToolParser()

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具调用协议和工具列表描述"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"
        if not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        return (
            base_prompt
            + "\n\n<tools_definitions>\n"
            + "你必须严格根据以下工具列表来回答用户问题。如果工具无法解决问题，请直接回答。\n"
            + f"工具列表:\n{tools_description}\n"
            + "</tools_definitions>"
            + self.parser.TOOL_CALL_PROTOCOL
        )

    def _build_messages(self, input_text: str) -> list[dict[str, str]]:
        """构建发送给 LLM 的消息列表"""
        messages = [{"role": "system", "content": self._get_enhanced_system_prompt()}]
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})
        return messages

    @staticmethod
    def _format_tool_result(res: dict[str, Any]) -> str:
        """将单条工具执行结果格式化为可读字符串"""
        if res.get("status") == "error":
            return (
                f"❌ 工具 {res['tool_name']} 执行失败：{res.get('result', '未知错误')}"
            )
        return f"🔧 工具 {res['tool_name']} 执行结果：\n{res.get('result', '无输出')}"

    async def _execute_tool_calls_async(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[str]:
        """异步并发执行所有工具调用，返回格式化结果列表"""
        if not tool_calls or not self.tool_registry:
            return []

        tasks = [
            self.parser.prepare_tool_task(
                call["tool_name"], call["raw_params"], self.tool_registry
            )
            for call in tool_calls
        ]
        try:
            results = await run_parallel_tools(
                registry=self.tool_registry,
                tasks=tasks,
                timeout=30.0,
            )
            return [self._format_tool_result(res) for res in results]
        except Exception as e:
            self.logger.error(f"🤖 工具执行器崩溃：{str(e)}")
            return [f"❌ 工具执行器崩溃：{str(e)}"]

    async def _run_with_tools_async(
        self, messages: list, max_tool_iterations: int, **kwargs
    ) -> str:
        """支持工具调用的运行逻辑（异步）"""
        try:
            for _ in range(max_tool_iterations):
                response = await self.llm.invoke(messages, **kwargs)
                tool_calls = self.parser.extract_tool_calls(response)

                if not tool_calls:
                    return response

                self.logger.info(
                    f"🤖 检测到 {len(tool_calls)} 个工具调用，正在并发执行..."
                )
                tool_results = await self._execute_tool_calls_async(tool_calls)

                clean_response = self.parser.strip_tool_calls(response, tool_calls)
                messages.append({"role": "assistant", "content": clean_response})
                self.add_message(Message(clean_response, "assistant"))

                for i, (call, result) in enumerate(zip(tool_calls, tool_results)):
                    tool_call_id = f"call_{hash(call['tool_name'] + str(i))}"
                    messages.append(
                        {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call_id,
                        }
                    )
                    self.add_message(Message(result, "tool"))

            self.logger.warning(
                f"🤖 Agent 达到最大迭代次数 ({max_tool_iterations})，强制终止。"
            )
            return (
                f"⚠️ 已达到最大工具调用次数限制 ({max_tool_iterations})，"
                "Agent 未能得出最终结论。请尝试重新表述您的问题或检查工具配置。"
            )
        except Exception as e:
            error_msg = f"🤖 Agent 执行过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            raise AgentException(error_msg) from e

    async def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """运行Agent，支持可选工具调用"""
        self.logger.info(f"🤖 {self.name} 正在处理: {input_text}")
        messages = self._build_messages(input_text)

        try:
            if self.tool_registry:
                # 先记录 user 消息，确保历史顺序正确（user → tool交互 → assistant）
                self.add_message(Message(input_text, "user"))
                response = await self._run_with_tools_async(
                    messages, max_tool_iterations, **kwargs
                )
                self.add_message(Message(response, "assistant"))
            else:
                response = await self.llm.invoke(messages, **kwargs)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(response, "assistant"))
        except AgentException:
            raise
        except Exception as e:
            error_msg = f"🤖 运行失败: {str(e)}"
            self.logger.error(error_msg)
            raise AgentException(error_msg) from e

        self.logger.info("🤖 Agent 响应完成")
        return response

    async def stream_run(self, input_text: str, **kwargs) -> AsyncIterator[str]:
        """
        异步流式运行Agent
        - 有工具注册表：先完整执行，再逐字符 yield
        - 无工具注册表：真正的流式输出
        """
        if self.tool_registry:
            self.logger.info("🤖 检测到工具注册表，流式模式降级为同步执行")
            try:
                response = await self.run(input_text, **kwargs)
                for char in response:
                    yield char
            except AgentException:
                raise
            except Exception as e:
                error_msg = f"🤖 流式运行（工具模式）发生错误: {str(e)}"
                self.logger.error(error_msg)
                raise AgentException(error_msg) from e
            return

        messages = self._build_messages(input_text)
        full_response = ""
        try:
            async for chunk in self.llm.think(messages, **kwargs):
                full_response += chunk
                yield chunk
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))
            self.logger.info("🤖 流式响应完成")
        except Exception as e:
            self.logger.error(f"🤖 流式消费过程中发生错误: {str(e)}")
            raise AgentException(f"流式驱动错误: {str(e)}") from e
