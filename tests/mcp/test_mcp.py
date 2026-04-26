from typing import cast
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

from momu_agent.core.exceptions import ToolException
from momu_agent.mcp import MCPClient, MCPToolAdapter


class DummySession:
    def __init__(self, result: CallToolResult | None = None):
        self.result = result or CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent=None,
            isError=False,
        )
        self.call_tool = AsyncMock(return_value=self.result)
        self.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    MCPTool(
                        name="list_files",
                        description="列出文件",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "目录路径",
                                },
                                "recursive": {
                                    "type": "boolean",
                                    "description": "是否递归",
                                    "default": False,
                                },
                            },
                            "required": ["path"],
                        },
                    )
                ]
            )
        )


@pytest.mark.asyncio
async def test_mcp_tool_adapter_get_parameters():
    client = MCPClient()
    adapter = MCPToolAdapter(
        client=client,
        remote_name="list_files",
        name="filesystem.list_files",
        description="列出文件",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径"},
                "options": {"type": "object", "description": "额外选项"},
                "patterns": {"type": "array", "description": "匹配模式"},
            },
            "required": ["path"],
        },
    )

    params = adapter.get_parameters()

    assert [param.name for param in params] == ["path", "options", "patterns"]
    assert params[0].type == "string"
    assert params[1].type == "object"
    assert params[2].type == "array"
    assert params[0].required is True
    assert params[1].required is False


@pytest.mark.asyncio
async def test_mcp_tool_adapter_run_returns_text():
    client = MCPClient()
    client.call_tool = AsyncMock(return_value=DummySession().result)
    adapter = MCPToolAdapter(
        client=client,
        remote_name="list_files",
        description="列出文件",
        input_schema={},
    )

    result = await adapter.run({"path": "."})

    assert result == "ok"
    client.call_tool.assert_awaited_once_with("list_files", {"path": "."})


@pytest.mark.asyncio
async def test_mcp_tool_adapter_run_returns_structured_json():
    result_payload = CallToolResult(
        content=[],
        structuredContent={"items": ["a", "b"]},
        isError=False,
    )
    client = MCPClient()
    client.call_tool = AsyncMock(return_value=result_payload)
    adapter = MCPToolAdapter(
        client=client,
        remote_name="list_files",
        description="列出文件",
        input_schema={},
    )

    result = await adapter.run({"path": "."})

    assert '"items"' in result
    assert '"a"' in result


@pytest.mark.asyncio
async def test_mcp_tool_adapter_run_raises_tool_exception_on_error_result():
    error_result = CallToolResult(
        content=[TextContent(type="text", text="failed")],
        structuredContent=None,
        isError=True,
    )
    client = MCPClient()
    client.call_tool = AsyncMock(return_value=error_result)
    adapter = MCPToolAdapter(
        client=client,
        remote_name="list_files",
        description="列出文件",
        input_schema={},
    )

    with pytest.raises(ToolException, match="failed"):
        await adapter.run({"path": "."})


@pytest.mark.asyncio
async def test_mcp_tool_adapter_close_does_not_close_client():
    client = MCPClient()
    client.close = AsyncMock(return_value=None)
    adapter = MCPToolAdapter(
        client=client,
        remote_name="list_files",
        description="列出文件",
        input_schema={},
    )

    await adapter.close()

    client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_client_list_tools_uses_session_result():
    client = MCPClient()
    session = DummySession()
    client.get_session = AsyncMock(return_value=session)

    tools = await client.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "list_files"
    assert tools[0].description == "列出文件"


@pytest.mark.asyncio
async def test_mcp_client_list_tools_reconnects_once_on_closed_connection():
    client = MCPClient()
    client._command = ["npx", "server"]
    first_session = DummySession()
    second_session = DummySession()
    first_session.list_tools.side_effect = RuntimeError("Connection closed")
    client.get_session = AsyncMock(side_effect=[first_session, second_session])
    client.close = AsyncMock(return_value=None)
    client.connect = AsyncMock(return_value=None)

    tools = await client.list_tools()

    assert len(tools) == 1
    client.close.assert_awaited_once_with()
    client.connect.assert_awaited_once_with(["npx", "server"])
    assert first_session.list_tools.await_count == 1
    assert second_session.list_tools.await_count == 1


@pytest.mark.asyncio
async def test_mcp_client_list_tools_raises_on_non_connection_error():
    client = MCPClient()
    session = DummySession()
    session.list_tools.side_effect = ValueError("bad response")
    client.get_session = AsyncMock(return_value=session)
    client.close = AsyncMock(return_value=None)
    client.connect = AsyncMock(return_value=None)

    with pytest.raises(ToolException, match="bad response"):
        await client.list_tools()

    client.close.assert_not_awaited()
    client.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_client_call_tool_reconnects_once_on_closed_connection():
    client = MCPClient()
    client._command = ["npx", "server"]
    first_session = DummySession()
    second_session = DummySession()
    first_session.call_tool.side_effect = RuntimeError("Connection closed")
    client.get_session = AsyncMock(side_effect=[first_session, second_session])
    client.close = AsyncMock(return_value=None)
    client.connect = AsyncMock(return_value=None)

    result = await client.call_tool("list_files", {"path": "."})

    assert result == second_session.result
    client.close.assert_awaited_once_with()
    client.connect.assert_awaited_once_with(["npx", "server"])
    first_session.call_tool.assert_awaited_once_with(
        "list_files", arguments={"path": "."}
    )
    second_session.call_tool.assert_awaited_once_with(
        "list_files", arguments={"path": "."}
    )


@pytest.mark.asyncio
async def test_mcp_client_close_is_idempotent():
    client = MCPClient()
    stack = AsyncMock()
    client._stack = stack
    client._session = cast(ClientSession, object())

    await client.close()
    await client.close()

    stack.aclose.assert_awaited_once_with()
    assert client.session is None


@pytest.mark.asyncio
async def test_mcp_client_get_session_requires_connect_command():
    client = MCPClient()

    with pytest.raises(ToolException, match="尚未连接"):
        await client.get_session()
