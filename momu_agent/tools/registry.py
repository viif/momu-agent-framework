"""工具注册表"""

from typing import Any, Callable, Optional

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .base import Tool


class ToolRegistry:
    """
    工具注册表

    提供工具的注册、管理和执行功能。
    支持两种工具注册方式：
    1. Tool对象注册（推荐）
    2. 函数直接注册（简便）
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

        self.logger = get_logger(__name__)

    def register_tool(self, tool: Tool):
        """
        注册Tool对象

        Args:
            tool: Tool实例
        """
        if tool.name in self._tools:
            self.logger.warning(f"🔧 工具 '{tool.name}' 已存在，将被覆盖。")

        self._tools[tool.name] = tool
        self.logger.info(f"🔧 工具 '{tool.name}' 已注册。")

    def register_function(
        self, name: str, description: str, func: Callable[[str], str]
    ):
        """
        直接注册函数作为工具（简便方式）

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._functions:
            self.logger.warning(f"🔧 工具 '{name}' 已存在，将被覆盖。")

        self._functions[name] = {"description": description, "func": func}
        self.logger.info(f"🔧 工具 '{name}' 已注册。")

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            self.logger.info(f"🔧 工具 '{name}' 已注销。")
        elif name in self._functions:
            del self._functions[name]
            self.logger.info(f"🔧 工具 '{name}' 已注销。")
        else:
            self.logger.warning(f"🔧 工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取Tool对象"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        执行工具

        Args:
            name: 工具名称
            input_text: 输入参数

        Returns:
            工具执行结果
        """
        # 优先查找Tool对象
        if name in self._tools:
            tool = self._tools[name]
            try:
                # 若 input_text 已经是解析好的参数字典，直接传入；否则包装为 {"input": ...}
                params = (
                    input_text
                    if isinstance(input_text, dict)
                    else {"input": input_text}
                )
                return tool.run(params)
            except ToolException as e:
                self.logger.error(f"🔧 工具 '{name}' 执行失败: {e}")
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"
            except Exception as e:
                self.logger.exception(f"🔧 工具 '{name}' 发生未知异常: {e}")
                return f"错误：执行工具 '{name}' 时发生未知异常: {str(e)}"

        # 查找函数工具
        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except ToolException as e:
                self.logger.error(f"🔧 工具 '{name}' 执行失败: {e}")
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"
            except Exception as e:
                self.logger.exception(f"🔧 工具 '{name}' 发生未知异常: {e}")
                return f"错误：执行工具 '{name}' 时发生未知异常: {str(e)}"

        else:
            self.logger.error(f"🔧 未找到名为 '{name}' 的工具。")
            return f"错误：未找到名为 '{name}' 的工具。"

    def get_tools_description(self) -> str:
        """
        获取所有可用工具的格式化描述字符串

        Returns:
            工具描述字符串，用于构建提示词
        """
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # 函数工具描述
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def list_tools(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """获取所有Tool对象"""
        return list(self._tools.values())

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._functions.clear()
        self.logger.info("🔧 所有工具已清空。")


# 全局工具注册表
global_registry = ToolRegistry()
