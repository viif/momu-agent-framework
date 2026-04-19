"""MomuAgent 异常体系"""


class MomuAgentException(Exception):
    """MomuAgent 基础异常类"""

    pass


class LLMException(MomuAgentException):
    """LLM 相关异常"""

    pass


class AgentException(MomuAgentException):
    """Agent 相关异常"""

    pass


class ConfigException(MomuAgentException):
    """配置相关异常"""

    pass


class ToolException(MomuAgentException):
    """工具相关异常"""

    pass


class MemoryException(MomuAgentException):
    """记忆相关异常"""

    pass


class StorageException(MomuAgentException):
    """存储相关异常"""

    pass
