"""工具调用解析器"""

import json
import re
from typing import Any

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

    PROTOCOL_PATTERN = r"\[TOOL_CALL:([^:]+):(.*?)\]"
    _PATTERN = re.compile(PROTOCOL_PATTERN, re.DOTALL)

    # 类型映射：协议类型名 → Python 类型（用于 isinstance 检查）
    _TYPE_MAP: dict[str, Any] = {
        "string": str,
        "str": str,
        "number": (int, float),
        "float": float,
        "integer": int,
        "int": int,
        "boolean": bool,
    }

    def __init__(self):
        self.logger = get_logger(__name__)

    def extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """从文本中提取所有工具调用（自动去重，保留首次出现顺序）"""
        if not text:
            return []

        try:
            seen: set[str] = set()
            tool_calls: list[dict[str, Any]] = []
            for name, params_str in self._PATTERN.findall(text):
                name, params_str = name.strip(), params_str.strip()
                key = f"[TOOL_CALL:{name}:{params_str}]"
                if key not in seen:
                    seen.add(key)
                    tool_calls.append({"tool_name": name, "raw_params": params_str})
            return tool_calls
        except Exception as e:
            self.logger.error(f"🔍 正则匹配工具调用失败: {e}")
            return []

    def parse_parameters(self, raw_params: str) -> dict[str, Any]:
        """解析 JSON 参数字符串，解析失败时尝试修复尾部多余逗号"""
        if not raw_params:
            return {}
        try:
            return json.loads(raw_params)
        except json.JSONDecodeError:
            self.logger.warning(f"🔍 JSON解析失败，尝试修复: {raw_params}")
            try:
                return json.loads(re.sub(r",\s*}", "}", raw_params))
            except Exception:
                self.logger.error("🔍 JSON修复失败")
                return {}

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
        try:
            tool_obj = registry.get_tool(tool_name)
            if tool_obj:
                params = self.parse_typed_parameters(tool_name, raw_params, tool_obj)
                return {"tool_name": tool_name, "input_data": params}

            func_obj = registry.get_function(tool_name)
            if func_obj:
                params = self.parse_parameters(raw_params)
                return {"tool_name": tool_name, "input_data": params}

            return {"tool_name": tool_name, "error": "工具未注册"}
        except Exception as e:
            return {"tool_name": tool_name, "error": str(e)}

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
        return bool(self._PATTERN.search(text))

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
