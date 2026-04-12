"""工具链管理"""

from dataclasses import dataclass
from typing import Any

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .registry import ToolRegistry


@dataclass
class ChainStep:
    """工具链单步配置"""

    tool_name: str
    input_template: str
    output_key: str


class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: list[ChainStep] = []
        self.logger = get_logger(__name__)

    def add_step(
        self, tool_name: str, input_template: str, output_key: str | None = None
    ) -> None:
        """
        添加工具执行步骤

        Args:
            tool_name: 工具名称（必须在 Registry 中注册）
            input_template: 输入模板，支持 {context_key} 变量替换
            output_key: 输出结果的键名，用于后续步骤引用
        """
        resolved_key = output_key or f"step_{len(self.steps)}_result"
        self.steps.append(ChainStep(tool_name, input_template, resolved_key))

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        顺序执行工具链，返回最后一步的结果

        Args:
            registry: 工具注册表
            initial_input: 初始输入字符串
            context: 额外的上下文变量（不会被修改）
        """
        if not self.steps:
            raise ToolException(f"工具链 '{self.name}' 没有任何步骤")

        exec_context: dict[str, Any] = {**(context or {}), "input": initial_input}
        self.logger.info(f"🔧 开始执行工具链 '{self.name}'，共 {len(self.steps)} 步")

        for i, step in enumerate(self.steps, 1):
            self.logger.info(f"🔧 步骤 {i}/{len(self.steps)}: 调用 [{step.tool_name}]")
            exec_context[step.output_key] = self._execute_step(
                step, exec_context, registry
            )
            self.logger.info(f"🔧 步骤 {i} 完成")

        final_result = exec_context[self.steps[-1].output_key]
        self.logger.info(f"🔧 工具链 '{self.name}' 执行完成")
        return final_result

    def _execute_step(
        self, step: ChainStep, context: dict[str, Any], registry: ToolRegistry
    ) -> str:
        """渲染输入模板并执行工具，返回结果字符串"""
        try:
            tool_input = step.input_template.format(**context)
        except KeyError as e:
            self.logger.error(f"🔧 模板错误: 缺少变量 {e}")
            raise ToolException(f"模板变量缺失: {e}")

        self.logger.debug(f"🔧 [{step.tool_name}] 输入: {tool_input[:50]}")

        try:
            return registry.execute_tool(step.tool_name, tool_input)
        except ToolException:
            raise
        except Exception as e:
            self.logger.error(f"🔧 工具执行失败 [{step.tool_name}]: {e}")
            raise ToolException(f"工具执行失败 [{step.tool_name}]: {e}")


class ToolChainManager:
    """工具链管理器 - 负责注册和调度工具链"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}
        self.logger = get_logger(__name__)

    def register_chain(self, chain: ToolChain) -> None:
        """注册工具链（同名时覆盖）"""
        if chain.name in self.chains:
            self.logger.warning(f"🔧 工具链 '{chain.name}' 已存在，正在覆盖")
        self.chains[chain.name] = chain
        self.logger.info(f"🔧 工具链 '{chain.name}' 已注册")

    def execute_chain(
        self, chain_name: str, input_data: str, context: dict[str, Any] | None = None
    ) -> str:
        """执行指定的工具链"""
        if chain_name not in self.chains:
            available = ", ".join(self.chains.keys()) or "（无）"
            msg = f"工具链 '{chain_name}' 不存在. 可用链: {available}"
            self.logger.error(f"🔧 {msg}")
            raise ToolException(msg)
        return self.chains[chain_name].execute(self.registry, input_data, context)

    def list_chains(self) -> list[str]:
        """列出所有工具链名称"""
        return list(self.chains.keys())
