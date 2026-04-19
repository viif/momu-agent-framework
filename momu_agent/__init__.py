"""
MomuAgent - 灵活、可扩展的多智能体框架
"""

import logging

from .agents.plan_solve_agent import PlanSolveAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.simple_agent import SimpleAgent
from .core.exceptions import MomuAgentException
from .core.llm import LLM
from .core.message import Message
from .tools.builtin.calculator import CalculatorTool, calculate
from .tools.builtin.search import SearchTool, search
from .tools.chain import ToolChain, ToolChainManager
from .tools.executor import ToolExecutor, run_batch_tool, run_parallel_tools
from .tools.registry import ToolRegistry
from .utils.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("urllib3").setLevel(logging.WARNING)

__all__ = [
    "LLM",
    "Config",
    "Message",
    "MomuAgentException",
    "CalculatorTool",
    "calculate",
    "SearchTool",
    "search",
    "ToolChain",
    "ToolChainManager",
    "ToolRegistry",
    "ToolExecutor",
    "run_parallel_tools",
    "run_batch_tool",
    "SimpleAgent",
    "ReActAgent",
    "PlanSolveAgent",
    "ReflectionAgent",
]
