"""工具调用解析器"""

import json
import re
from typing import Any

from ...tools.registry import ToolRegistry
from ...utils.logger import get_logger


class ToolParser:
    """
    工具调用解析器
    负责从 LLM 返回的文本中提取工具调用请求并解析参数
    """

    TOOL_CALL_PROTOCOL = """
<tool_execution_protocol>
### 工具调用协议 ###
当且仅当需要调用工具时，必须严格输出以下格式的字符串，不要输出任何其他解释性文字：
格式：`[TOOL_CALL:{tool_name}:{parameters_json}]`

### 参数格式说明 ###
- **parameters_json** 必须是一个合法的 **JSON 对象**。
- **字符串**：必须使用双引号 `"` 包裹 (e.g., `{"query": "Python 编程"}`)
- **数字**：直接写数字，不需要引号 (e.g., `{"count": 5}`)
- **特殊字符**：JSON 原生支持逗号、空格等特殊字符，直接写在引号内即可。

### 示例 ###
1. 无参数：`[TOOL_CALL:list_tools:{}]`
2. 搜索工具：`[TOOL_CALL:search:{"query": "LangChain 教程"}]`
3. 计算器工具：`[TOOL_CALL:calculator:{"expression": "100 / 5 + 10"}]`
4. 复杂参数：`[TOOL_CALL:write_file:{"path": "test.py", "content": "print('Hello, World!')"}]`
</tool_execution_protocol>
"""

    _TYPE_MAP: dict[str, Any] = {
        "string": str,
        "str": str,
        "number": (int, float),
        "float": float,
        "integer": int,
        "int": int,
        "boolean": bool,
    }

    _COMMON_ACTION_WORDS = {
        "create",
        "read",
        "update",
        "delete",
        "list",
        "search",
        "summary",
        "stats",
        "clear",
    }

    def __init__(self, tool_registry: ToolRegistry | None = None):
        self.logger = get_logger(__name__)
        self._tool_registry = tool_registry
        self._alias_index = self._build_alias_index(tool_registry)

    def _build_alias_index(
        self, registry: ToolRegistry | None
    ) -> dict[str, dict[str, Any]]:
        """基于已注册工具构建通用别名索引。"""
        alias_index: dict[str, dict[str, Any]] = {}
        if not registry:
            return alias_index

        for tool in registry.get_all_tools():
            try:
                params = tool.get_parameters()
            except Exception as e:
                self.logger.warning("🔍 读取工具参数失败: %s - %s", tool.name, e)
                continue

            param_names = {p.name for p in params}
            if "action" not in param_names:
                continue

            alias_index[tool.name] = {
                "description": tool.description,
                "param_names": param_names,
            }

        self.logger.debug(
            "🔧 ToolParser别名索引构建完成: alias_tool_count=%s",
            len(alias_index),
        )
        return alias_index

    def _refresh_alias_index_if_needed(self, registry: ToolRegistry) -> None:
        if self._alias_index:
            return
        self._alias_index = self._build_alias_index(registry)

    def extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """从文本中提取所有工具调用（自动去重，保留首次出现顺序）"""
        if not text:
            return []

        seen: set[str] = set()
        tool_calls: list[dict[str, Any]] = []
        cursor = 0

        while True:
            start = text.find("[TOOL_CALL:", cursor)
            if start == -1:
                break

            name_start = start + len("[TOOL_CALL:")
            name_end = text.find(":", name_start)
            if name_end == -1:
                break

            tool_name = text[name_start:name_end].strip()
            if not tool_name:
                cursor = name_end + 1
                continue

            params_start = name_end + 1
            parse_result = self._extract_params_block(text, params_start)
            if parse_result is None:
                cursor = params_start
                continue

            raw_params, end_index = parse_result
            key = f"[TOOL_CALL:{tool_name}:{raw_params}]"
            if key not in seen:
                seen.add(key)
                tool_calls.append({"tool_name": tool_name, "raw_params": raw_params})

            cursor = end_index + 1

        return tool_calls

    def _extract_params_block(
        self, text: str, params_start: int
    ) -> tuple[str, int] | None:
        """提取 TOOL_CALL 中参数 JSON，允许字符串内出现 ]。"""
        if params_start >= len(text):
            return None

        first = text[params_start]
        if first != "{":
            closing = text.find("]", params_start)
            if closing == -1:
                return None
            return text[params_start:closing].strip(), closing

        brace_depth = 0
        in_string = False
        escaped = False

        for i in range(params_start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                brace_depth += 1
                continue

            if ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    if i + 1 < len(text) and text[i + 1] == "]":
                        return text[params_start : i + 1], i + 1
                    return None

        return None

    def parse_parameters(self, raw_params: str) -> dict[str, Any]:
        """解析 JSON 参数字符串，解析失败时尝试有限修复。"""
        if not raw_params:
            return {}

        normalized = raw_params.strip()

        bare_action = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)$", normalized)
        if bare_action and bare_action.group(1) in self._COMMON_ACTION_WORDS:
            return {"action": bare_action.group(1)}

        action_prefix = re.match(
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\{.*\})$", normalized
        )
        if action_prefix:
            action, payload = action_prefix.groups()
            try:
                parsed_payload = json.loads(payload)
                if isinstance(parsed_payload, dict) and "action" not in parsed_payload:
                    parsed_payload = {"action": action, **parsed_payload}
                return parsed_payload
            except json.JSONDecodeError:
                pass

        if normalized.startswith("create,"):
            normalized = normalized[len("create,") :].strip()

        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            self.logger.warning(f"🔍 JSON解析失败，尝试修复: {raw_params}")
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", normalized)
                return json.loads(fixed)
            except Exception:
                self.logger.error("🔍 JSON修复失败")
                return {}

    def _normalize_tool_name(self, tool_name: str) -> tuple[str, str | None]:
        match = re.match(
            r"^([a-zA-Z_][a-zA-Z0-9_]*)_([a-zA-Z_][a-zA-Z0-9_]*)$", tool_name
        )
        if not match:
            return tool_name, None

        base_name, suffix = match.groups()
        if base_name not in self._alias_index:
            return tool_name, None

        if suffix not in self._COMMON_ACTION_WORDS:
            return tool_name, None

        self.logger.info(
            "🔧 工具名通用归一化: %s -> %s(action=%s)",
            tool_name,
            base_name,
            suffix,
        )
        return base_name, suffix

    def _apply_forced_action(
        self, params: dict[str, Any], forced_action: str | None
    ) -> dict[str, Any]:
        if not forced_action or "action" in params:
            return params
        return {"action": forced_action, **params}

    def parse_typed_parameters(
        self, tool_name: str, raw_parameters: str, tool_obj: Any
    ) -> dict[str, Any]:
        """
        解析 JSON 参数字符串并按工具定义做类型转换

        Args:
            tool_name: 工具名称（仅用于日志）
            raw_parameters: 原始 JSON 字符串
            tool_obj: 工具对象实例，用于获取参数定义

        Returns:
            转换类型后的参数字典
        """
        parsed = self.parse_parameters(raw_parameters)
        if not parsed:
            return {}

        param_defs: dict[str, Any] = {}
        if tool_obj:
            try:
                param_defs = {p.name: p for p in tool_obj.get_parameters()}
            except Exception as e:
                self.logger.error(f"🔍 获取工具 {tool_name} 参数定义失败: {e}")

        result: dict[str, Any] = {}
        for key, value in parsed.items():
            if key in param_defs:
                result[key] = self._convert_value_type(value, param_defs[key])
            else:
                self.logger.warning(f"🔍 工具 {tool_name} 未定义参数: {key}")
                result[key] = value
        return result

    def prepare_tool_task(
        self, tool_name: str, raw_params: str, registry: Any
    ) -> dict[str, Any]:
        """
        准备工具执行任务（参数解析 + 类型转换）

        Args:
            tool_name: 工具名称
            raw_params: 原始 JSON 参数字符串
            registry: 工具注册表

        Returns:
            ``{"tool_name": ..., "input_data": ...}``（成功）或
            ``{"tool_name": ..., "error": ...}``（失败）
        """
        if not registry:
            return {"tool_name": tool_name, "error": "未配置工具注册表"}

        self._refresh_alias_index_if_needed(registry)
        normalized_tool_name, forced_action = self._normalize_tool_name(tool_name)

        try:
            tool_obj = registry.get_tool(normalized_tool_name)
            if tool_obj:
                params = self.parse_typed_parameters(
                    normalized_tool_name, raw_params, tool_obj
                )
                params = self._apply_forced_action(params, forced_action)
                return {"tool_name": normalized_tool_name, "input_data": params}

            func_obj = registry.get_function(normalized_tool_name)
            if func_obj:
                params = self.parse_parameters(raw_params)
                params = self._apply_forced_action(params, forced_action)
                return {"tool_name": normalized_tool_name, "input_data": params}

            return {"tool_name": normalized_tool_name, "error": "工具未注册"}
        except Exception as e:
            return {"tool_name": normalized_tool_name, "error": str(e)}

    def strip_tool_calls(self, text: str, tool_calls: list[dict[str, Any]]) -> str:
        """从文本中移除所有工具调用标记，返回清理后的文本"""
        result = text
        for call in tool_calls:
            result = result.replace(
                f"[TOOL_CALL:{call['tool_name']}:{call['raw_params']}]", ""
            )
        return result.strip()

    def has_tool_calls(self, text: str) -> bool:
        """检查文本是否包含工具调用"""
        return bool(self.extract_tool_calls(text))

    def _convert_value_type(self, value: Any, param_def: Any) -> Any:
        """按参数定义转换值的类型，转换失败时原值返回"""
        target_type = param_def.type
        expected = self._TYPE_MAP.get(target_type.lower())

        if expected and isinstance(value, expected):
            return value

        try:
            if target_type in ("number", "float"):
                return float(value)
            if target_type in ("integer", "int"):
                return int(value)
            if target_type in ("boolean", "bool"):
                return (
                    value if isinstance(value, bool) else str(value).lower() == "true"
                )
            return str(value)
        except (ValueError, TypeError):
            self.logger.warning(f"🔍 类型转换失败 {target_type}: {value}")
            return value
