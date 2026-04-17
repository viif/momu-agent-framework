"""工具基类"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ..utils.logger import get_logger


class ToolParameter(BaseModel):
    """工具参数定义"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

        self.logger = get_logger(__name__)

    @abstractmethod
    async def run(self, parameters: dict[str, Any]) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """获取工具参数定义"""
        pass

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """验证参数"""
        self.logger.debug(f"🔧 正在验证工具 [{self.name}] 的参数: {parameters}")

        required_params = [p.name for p in self.get_parameters() if p.required]
        missing_params = [param for param in required_params if param not in parameters]

        if missing_params:
            self.logger.error(
                f"🔧 工具 [{self.name}] 参数验证失败，缺少必填参数: {missing_params}"
            )
            return False

        self.logger.debug(f"🔧 工具 [{self.name}] 参数验证通过")
        return True

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.model_dump() for param in self.get_parameters()],
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
