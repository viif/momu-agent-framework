"""简单Agent实现"""

import asyncio
import re
from typing import Any, Dict, Iterator, List, Optional

from ..core.agent import Agent
from ..core.exceptions import AgentException
from ..core.llm import LLM
from ..core.message import Message
from ..tools.async_executor import run_parallel_tools
from ..tools.registry import ToolRegistry
from ..utils.logger import get_logger
from .parser.tool_parser import ToolParser


class SimpleAgent(Agent):
    """简单的对话Agent，支持可选的工具调用"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        max_history_length: int = 100,
    ):
        """
        初始化SimpleAgent
        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            max_history_length: 最大历史记录长度
        """
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

        prompt_parts = []
        prompt_parts.append(base_prompt)
        prompt_parts.append("\n\n<tools_definitions>")
        prompt_parts.append(
            "你必须严格根据以下工具列表来回答用户问题。如果工具无法解决问题，请直接回答。"
        )
        prompt_parts.append(f"工具列表:\n{tools_description}")
        prompt_parts.append("</tools_definitions>")

        prompt_parts.append(self.parser.TOOL_CALL_PROTOCOL)

        return "".join(prompt_parts)

    def _prepare_tool_task(self, tool_name: str, raw_parameters: str) -> Dict[str, Any]:
        """
        准备工具调用任务（数据转换层）
        使用 ToolParser 进行参数解析和类型转换
        """
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

    async def _execute_tool_calls_async(
        self, tool_calls: List[Dict[str, Any]], registry: ToolRegistry
    ) -> List[str]:
        """
        异步并发执行所有工具调用
        """
        if not tool_calls:
            return []

        tasks = []
        for call in tool_calls:
            task = self._prepare_tool_task(call["tool_name"], call["raw_params"])
            tasks.append(task)

        try:
            results = await run_parallel_tools(
                registry=registry,
                tasks=tasks,
                max_workers=4,
                timeout=30.0,
            )

            formatted_results = []
            for res in results:
                tool_name = res.get("tool_name")
                if res.get("status") == "error":
                    formatted_results.append(
                        f"❌ 工具 {tool_name} 执行失败：{res.get('result', '未知错误')}"
                    )
                else:
                    output = res.get("result", "无输出")
                    formatted_results.append(
                        f"🔧 工具 {tool_name} 执行结果：\n{output}"
                    )
            return formatted_results
        except Exception as e:
            self.logger.error(f"🤖 工具执行器崩溃：{str(e)}")
            return [f"❌ 工具执行器崩溃：{str(e)}"]

    async def _run_with_tools_async(
        self,
        messages: list,
        input_text: str,
        max_tool_iterations: int,
        registry: ToolRegistry,
        **kwargs,
    ) -> str:
        """
        支持异步工具调用的运行逻辑
        增加了异常捕获，防止内部错误导致程序崩溃
        """
        try:
            current_iteration = 0
            final_response = ""

            while current_iteration < max_tool_iterations:
                response = await self.llm.invoke(messages, **kwargs)
                tool_calls = self.parser.extract_tool_calls(response)

                if tool_calls:
                    self.logger.info(
                        f"🤖 检测到 {len(tool_calls)} 个工具调用，正在并发执行..."
                    )
                    tool_results = await self._execute_tool_calls_async(
                        tool_calls, registry
                    )

                    # 1. 清理并添加 Assistant 的回复（工具调用指令）
                    clean_response = response
                    for call in tool_calls:
                        escaped_original = re.escape(
                            f"[TOOL_CALL:{call['tool_name']}:{call['raw_params']}]"
                        )
                        clean_response = re.sub(escaped_original, "", clean_response)

                    messages.append({"role": "assistant", "content": clean_response})
                    self.add_message(Message(clean_response, "assistant"))

                    # 2. 添加 Tool 的执行结果
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

                    current_iteration += 1
                    continue

                final_response = response
                break

            if not final_response:
                final_response = (
                    f"⚠️ 已达到最大工具调用次数限制 ({max_tool_iterations})，"
                    "Agent 未能得出最终结论。请尝试重新表述您的问题或检查工具配置。"
                )
                self.logger.warning(
                    f"🤖 ⚠️ Agent 达到最大迭代次数 ({max_tool_iterations})，强制终止。"
                )

            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_response, "assistant"))

            self.logger.info("🤖 Agent 响应完成")
            return final_response

        except Exception as e:
            error_msg = f"🤖 Agent 执行过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            raise AgentException(error_msg) from e

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """重写的运行方法 - 实现简单对话逻辑，支持可选工具调用"""
        self.logger.info(f"🤖 {self.name} 正在处理: {input_text}")

        messages = []
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        if not self.tool_registry:
            try:
                response = asyncio.run(self.llm.invoke(messages, **kwargs))
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(response, "assistant"))
                self.logger.info("🤖 无工具调用，响应完成")
                return response
            except Exception as e:
                error_msg = f"🤖 同步运行模式下 LLM 调用失败: {str(e)}"
                self.logger.error(error_msg)
                raise AgentException(error_msg) from e

        try:
            result = asyncio.run(
                self._run_with_tools_async(
                    messages,
                    input_text,
                    max_tool_iterations,
                    self.tool_registry,
                    **kwargs,
                )
            )
            return result
        except Exception as e:
            if not isinstance(e, AgentException):
                error_msg = f"🤖 运行工具链时发生错误: {str(e)}"
                self.logger.error(error_msg)
                raise AgentException(error_msg) from e
            raise

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent
        策略：
        1. 若无工具注册表：使用真正的异步流式输出 (self.llm.think)
        2. 若有工具注册表：降级为同步执行 (self.run)，一次性返回结果
        """
        # --- 分支1：有工具注册表 -> 直接调用同步的 run 方法 ---
        if self.tool_registry:
            self.logger.info("🤖 检测到工具注册表，流式模式降级为同步执行")
            try:
                final_response = self.run(input_text, **kwargs)
                for char in final_response:
                    yield char
            except AgentException:
                raise
            except Exception as e:
                error_msg = f"🤖 流式运行（工具模式）发生错误: {str(e)}"
                self.logger.error(error_msg)
                raise AgentException(error_msg) from e
            return

        # --- 分支2：无工具注册表 -> 真正的异步流式 ---
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})

        # 定义一个异步协程来消费 llm.think 的流
        async def _consume_stream():
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
                raise

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        stream_gen = _consume_stream()
        try:
            while True:
                coro = stream_gen.__anext__()
                try:
                    chunk = loop.run_until_complete(coro)
                    yield chunk
                except StopAsyncIteration:
                    break
        except Exception as e:
            self.logger.error(f"🤖 同步流式驱动发生错误: {str(e)}")
            raise AgentException(f"流式驱动错误: {str(e)}") from e
