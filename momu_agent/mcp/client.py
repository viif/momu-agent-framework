"""MCP 客户端。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, TypeVar, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .tool_adapter import MCPToolAdapter

if TYPE_CHECKING:
    from ..tools.base import Tool

T = TypeVar("T")


class MCPClient:
    """基于 stdio 的异步 MCP 客户端。"""

    def __init__(self):
        self._command: list[str] | None = None
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._connect_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self.logger = get_logger(__name__)

    @property
    def session(self) -> ClientSession | None:
        return self._session

    async def connect(self, command: list[str]) -> None:
        if not command:
            raise ToolException("MCP 启动命令不能为空")

        self.logger.debug("🔧 准备建立 MCP 连接: %s", command)

        async with self._connect_lock:
            if self._session is not None and self._command == command:
                self.logger.debug("🔧 MCP 连接已存在，复用当前会话")
                return

            if self._session is not None and self._command != command:
                self.logger.info("🔧 MCP 启动命令变化，关闭旧连接后重新建立会话")
                await self.close()

            self._command = command.copy()
            stack = AsyncExitStack()

            try:
                server_params = StdioServerParameters(
                    command=command[0],
                    args=command[1:],
                )
                read_stream, write_stream = await stack.enter_async_context(
                    cast(Any, stdio_client(server_params))
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
            except Exception as e:
                self.logger.exception("🔧 建立 MCP 连接失败: %s", e)
                await stack.aclose()
                raise ToolException(f"建立 MCP 连接失败: {e}") from e

            self._stack = stack
            self._session = session
            self.logger.info("🔧 MCP 连接建立成功: %s", command)

    async def get_session(self) -> ClientSession:
        if self._session is None:
            if not self._command:
                self.logger.error("🔧 MCPClient 尚未连接，无法获取会话")
                raise ToolException("MCPClient 尚未连接")
            self.logger.debug("🔧 当前无可用 MCP 会话，尝试自动重连")
            await self.connect(self._command)

        if self._session is None:
            self.logger.error("🔧 MCP 会话重连后仍不可用")
            raise ToolException("MCP 会话不可用")
        return self._session

    def _is_retryable_connection_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return any(
            keyword in message
            for keyword in (
                "connection closed",
                "stream closed",
                "closed resource",
                "broken pipe",
                "eof",
            )
        )

    async def _call_with_reconnect(
        self,
        operation_name: str,
        operation: Callable[[ClientSession], Awaitable[T]],
    ) -> T:
        session = await self.get_session()

        try:
            return await operation(session)
        except Exception as e:
            if not self._is_retryable_connection_error(e):
                raise
            if not self._command:
                raise

            self.logger.warning(
                "🔧 %s 遇到连接错误，准备重连后重试一次: %s",
                operation_name,
                e,
            )
            await self.close()
            await self.connect(self._command)
            retry_session = await self.get_session()
            return await operation(retry_session)

    async def list_tools(self) -> list[Tool]:
        self.logger.debug("🔧 开始获取 MCP 工具列表")

        try:
            result = await self._call_with_reconnect(
                "获取 MCP 工具列表",
                lambda session: session.list_tools(),
            )
        except Exception as e:
            self.logger.exception("🔧 获取 MCP 工具列表失败: %s", e)
            raise ToolException(f"获取 MCP 工具列表失败: {e}") from e

        tools: list[Tool] = []
        for tool in result.tools:
            tools.append(
                MCPToolAdapter(
                    client=self,
                    remote_name=tool.name,
                    description=tool.description or tool.title or tool.name,
                    input_schema=tool.inputSchema,
                )
            )

        self.logger.info("🔧 已加载 %s 个 MCP 工具", len(tools))
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.logger.debug("🔧 开始调用 MCP 工具 '%s'", name)

        try:
            return await self._call_with_reconnect(
                f"调用 MCP 工具 '{name}'",
                lambda session: session.call_tool(name, arguments=arguments),
            )
        except Exception as e:
            self.logger.exception("🔧 MCP 工具 '%s' 调用失败: %s", name, e)
            raise ToolException(f"MCP 工具 '{name}' 调用失败: {e}") from e

    async def close(self) -> None:
        async with self._close_lock:
            if self._stack is None:
                self._session = None
                self.logger.debug("🔧 当前没有可关闭的 MCP 连接")
                return

            self.logger.debug("🔧 开始关闭 MCP 连接")
            stack = self._stack
            self._stack = None
            self._session = None
            await stack.aclose()
            self.logger.info("🔧 MCP 连接已关闭")
