"""工具链管理"""

from typing import Any, Dict, List, Optional

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .registry import ToolRegistry


class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

        self.logger = get_logger(__name__)

    def add_step(
        self, tool_name: str, input_template: str, output_key: Optional[str] = None
    ):
        """
        添加工具执行步骤

        Args:
            tool_name: 工具名称（必须在 Registry 中注册）
            input_template: 输入模板，支持 {context_key} 变量替换
            output_key: 输出结果的键名，用于后续步骤引用
        """
        self.steps.append(
            {
                "tool_name": tool_name,
                "input_template": input_template,
                "output_key": output_key or f"step_{len(self.steps)}_result",
            }
        )

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        执行工具链

        Args:
            registry: 工具注册表
            initial_input: 初始输入字符串
            context: 额外的上下文变量

        Returns:
            最后一步的执行结果
        """
        exec_context = context.copy() if context else {}
        exec_context["input"] = initial_input

        self.logger.info(f"🔧 开始执行工具链: {self.name}")

        try:
            for i, step in enumerate(self.steps, 1):
                tool_name = step["tool_name"]
                input_template = step["input_template"]
                output_key = step["output_key"]

                try:
                    tool_input = input_template.format(**exec_context)
                except KeyError as e:
                    self.logger.error(f"🔧 模板错误: 缺少变量 {e}")
                    raise ToolException(f"模板变量缺失: {e}")

                self.logger.info(f"🔧 步骤 {i}/{len(self.steps)}: 调用 [{tool_name}]")
                self.logger.debug(f"🔧 输入预览: {tool_input[:50]}...")  # 调试用

                try:
                    result = registry.execute_tool(tool_name, tool_input)
                except Exception as e:
                    self.logger.error(f"🔧 工具执行失败 [{tool_name}]: {str(e)}")
                    raise ToolException(f"工具执行失败 [{tool_name}]: {str(e)}")

                exec_context[output_key] = result
                self.logger.info(f"🔧 步骤 {i} 完成")

            final_result = exec_context[self.steps[-1]["output_key"]]
            self.logger.info(f"🔧 工具链 '{self.name}' 执行完成")
            return final_result

        except ToolException:
            raise
        except Exception as e:
            self.logger.error(f"🔧 工具链系统错误: {str(e)}")
            raise ToolException(f"工具链系统错误: {str(e)}")


class ToolChainManager:
    """工具链管理器 - 负责注册和调度工具链"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

        self.logger = get_logger(__name__)

    def register_chain(self, chain: ToolChain):
        """注册工具链"""
        if chain.name in self.chains:
            self.logger.warning(f"🔧 工具链 '{chain.name}' 已存在，正在覆盖")
        self.chains[chain.name] = chain
        self.logger.info(f"🔧 工具链 '{chain.name}' 已注册")

    def execute_chain(
        self, chain_name: str, input_data: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """执行指定的工具链"""
        if chain_name not in self.chains:
            available = ", ".join(self.chains.keys())
            error_msg = f"工具链 '{chain_name}' 不存在. 可用链: {available}"
            self.logger.error(f"🔧 {error_msg}")
            raise ToolException(error_msg)

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """列出所有工具链名称"""
        return list(self.chains.keys())
