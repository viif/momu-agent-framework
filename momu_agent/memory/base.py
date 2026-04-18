"""记忆系统基础类和配置。

- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统配置
- Memory: 记忆基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """记忆项数据结构。"""

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """记忆系统配置。"""

    # 存储路径
    storage_path: str = "./memory_data"

    # 统计显示用的基础配置
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: list[str] = Field(
        default_factory=lambda: ["text", "image", "audio", "video"]
    )


class Memory(ABC):
    """记忆基类。

    定义所有记忆类型的通用接口和行为。
    """

    def __init__(self, config: MemoryConfig, storage_backend: Any = None) -> None:
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    async def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项。"""

    @abstractmethod
    async def retrieve(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> list[MemoryItem]:
        """检索相关记忆。"""

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆。"""

    @abstractmethod
    async def remove(self, memory_id: str) -> bool:
        """删除记忆。"""

    @abstractmethod
    async def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""

    @abstractmethod
    async def clear(self) -> None:
        """清空所有记忆。"""

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """获取记忆统计信息。"""

    def _generate_id(self) -> str:
        """生成记忆 ID。"""
        return str(uuid4())

    def _calculate_importance(
        self, content: str, base_importance: float = 0.5
    ) -> float:
        """计算记忆重要性。"""
        importance = base_importance

        if len(content) > 100:
            importance += 0.1

        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(type={self.memory_type})"

    def __repr__(self) -> str:
        return self.__str__()
