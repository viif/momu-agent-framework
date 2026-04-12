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

    def __init__(self):
        """初始化解析器，配置专属 logger"""
        self.logger = get_logger(__name__)

    def extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        从文本中提取所有工具调用
        Returns:
            List[Dict]: 包含 tool_name 和 raw_params 的列表
        """
        if not text:
            return []

        try:
            matches = re.findall(self.PROTOCOL_PATTERN, text, flags=re.DOTALL)
            tool_calls = []
            seen_originals = set()

            for name, params_str in matches:
                clean_name = name.strip()
                clean_params = params_str.strip()
                original_str = f"[TOOL_CALL:{clean_name}:{clean_params}]"

                if original_str in seen_originals:
                    continue
                seen_originals.add(original_str)

                tool_calls.append({"tool_name": clean_name, "raw_params": clean_params})
            return tool_calls
        except Exception as e:
            self.logger.error(f"🔍 正则匹配工具调用失败: {e}")
            return []

    def parse_parameters(self, raw_params: str) -> dict[str, Any]:
        """
        解析 JSON 参数字符串
        """
        if not raw_params:
            return {}

        try:
            return json.loads(raw_params)
        except json.JSONDecodeError:
            self.logger.warning(f"🔍 JSON解析失败，尝试修复: {raw_params}")
            try:
                cleaned = re.sub(r",\s*}", "}", raw_params)
                return json.loads(cleaned)
            except Exception:
                self.logger.error("🔍 JSON修复失败")
                return {}

    def parse_typed_parameters(
        self, tool_name: str, raw_parameters: str, tool_obj: Any
    ) -> dict[str, Any]:
        """
        解析并转换工具参数类型

        Args:
            tool_name: 工具名称
            raw_parameters: 原始 JSON 字符串
            tool_obj: 工具对象实例，用于获取参数定义

        Returns:
            Dict: 转换类型后的参数字典
        """
        parsed_data = self.parse_parameters(raw_parameters)
        if not parsed_data:
            return {}

        param_definitions = {}
        if tool_obj:
            try:
                for p in tool_obj.get_parameters():
                    param_definitions[p.name] = p
            except Exception as e:
                self.logger.error(f"🔍 获取工具 {tool_name} 参数定义失败: {e}")

        final_params = {}
        for key, value in parsed_data.items():
            if key in param_definitions:
                final_params[key] = self._convert_value_type(
                    value, param_definitions[key]
                )
            else:
                self.logger.warning(f"🔍 工具 {tool_name} 未定义参数: {key}")
                final_params[key] = value

        return final_params

    def _convert_value_type(self, value: Any, param_def: Any) -> Any:
        """根据参数定义转换值的类型"""
        target_type = param_def.type
        type_mapping = {
            "string": str,
            "str": str,
            "number": (int, float),
            "float": float,
            "integer": int,
            "int": int,
            "boolean": bool,
        }
        expected_type = type_mapping.get(target_type.lower())

        if expected_type and isinstance(value, expected_type):
            return value

        try:
            if target_type in ["number", "float"]:
                return float(value)
            elif target_type in ["integer", "int"]:
                return int(value)
            elif target_type in ["boolean", "bool"]:
                if isinstance(value, bool):
                    return value
                return str(value).lower() == "true"
            else:
                return str(value)
        except (ValueError, TypeError):
            self.logger.warning(f"🔍 类型转换失败 {target_type}: {value}")
            return value

    def has_tool_calls(self, text: str) -> bool:
        """检查文本是否包含工具调用"""
        return bool(re.search(self.PROTOCOL_PATTERN, text))
