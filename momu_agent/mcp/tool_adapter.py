"""MCP 工具适配器。"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult

from ..core.exceptions import ToolException
from ..tools.base import Tool, ToolParameter
from ..utils.logger import get_logger


class MCPToolAdapter(Tool):
    """将远程 MCP 工具适配为框架内 Tool。"""

    def __init__(
        self,
        client: Any,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        super().__init__(
            name=name or remote_name, description=description or remote_name
        )
        self.client = client
        self.remote_name = remote_name
        self.input_schema = input_schema or {}
        self.logger = get_logger(__name__)

    def get_parameters(self) -> list[ToolParameter]:
        properties = self.input_schema.get("properties")
        if not isinstance(properties, dict):
            self.logger.debug("🔧 MCP 工具 '%s' 未声明可解析参数", self.name)
            return []

        required_names = self.input_schema.get("required", [])
        if not isinstance(required_names, list):
            required_names = []

        parameters: list[ToolParameter] = []
        for param_name, schema in properties.items():
            if not isinstance(schema, dict):
                schema = {}

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=self._resolve_parameter_type(schema),
                    description=self._build_parameter_description(schema),
                    required=param_name in required_names,
                    default=schema.get("default"),
                )
            )

        self.logger.debug(
            "🔧 MCP 工具 '%s' 已解析 %s 个参数", self.name, len(parameters)
        )
        return parameters

    async def run(self, parameters: dict[str, Any]) -> str:
        self.logger.debug("🔧 调用 MCP 工具 '%s'，参数: %s", self.name, parameters)

        try:
            result = await self.client.call_tool(self.remote_name, parameters)
        except ToolException:
            raise
        except Exception as e:
            self.logger.exception("🔧 MCP 工具 '%s' 调用失败: %s", self.name, e)
            raise ToolException(f"MCP 工具 '{self.name}' 调用失败: {e}") from e

        rendered = self._render_result(result)
        if result.isError:
            self.logger.warning("🔧 MCP 工具 '%s' 返回错误结果", self.name)
            raise ToolException(rendered or f"MCP 工具 '{self.name}' 执行失败")

        self.logger.debug("🔧 MCP 工具 '%s' 调用成功", self.name)
        return rendered

    async def close(self) -> None:
        self.logger.debug("🔧 MCP 工具适配器 '%s' 无需单独关闭共享连接", self.name)

    def _resolve_parameter_type(self, schema: dict[str, Any]) -> str:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            for item in schema_type:
                if item != "null" and isinstance(item, str):
                    schema_type = item
                    break
            else:
                schema_type = "string"

        if isinstance(schema_type, str):
            return schema_type

        if "enum" in schema:
            return "string"
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        if any(key in schema for key in ("oneOf", "anyOf", "allOf")):
            return "string"
        return "string"

    def _build_parameter_description(self, schema: dict[str, Any]) -> str:
        description = str(schema.get("description") or "")

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            choices = ", ".join(str(value) for value in enum_values)
            description = f"{description} 可选值: {choices}".strip()

        if any(key in schema for key in ("oneOf", "anyOf", "allOf")):
            hint = "该参数使用复合 schema，必要时请传入符合服务端要求的 JSON 值。"
            description = f"{description} {hint}".strip()

        return description or "MCP 参数"

    def _render_result(self, result: CallToolResult) -> str:
        text_parts: list[str] = []
        structured_part: str | None = None

        for item in result.content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                text_parts.append(text)
                continue

            dumped = item.model_dump(mode="json")
            text_parts.append(json.dumps(dumped, ensure_ascii=False, indent=2))

        if result.structuredContent is not None:
            structured_part = json.dumps(
                result.structuredContent,
                ensure_ascii=False,
                indent=2,
            )

        if text_parts and structured_part:
            return "\n\n".join(["\n\n".join(text_parts), structured_part])
        if text_parts:
            return "\n\n".join(text_parts)
        if structured_part:
            return structured_part
        return "工具执行成功，无返回内容"
