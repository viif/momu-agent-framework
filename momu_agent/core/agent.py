"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional

from ..utils.logger import get_logger
from .llm import LLM
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        max_history_length: int = 100,
    ):
        """
        初始化 Agent

        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            max_history_length: 最大历史记录长度，超过时会自动移除最早的记录
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_history_length = max_history_length
        self._history: list[Message] = []

        self.logger = get_logger(__name__)

        self.logger.info(f"🤖 Agent '{self.name}' 初始化完成 (模型: {self.llm.model})")
        if self.system_prompt:
            self.logger.debug(f"🤖 系统提示词已加载: {self.system_prompt[:50]}...")

    @abstractmethod
    async def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """
        添加消息到历史记录
        如果历史记录超过最大长度，自动移除最早的记录 (FIFO)
        """
        self._history.append(message)
        self.logger.debug(
            f"🤖 添加消息到历史: [{message.role}] {message.content[:30]}..."
        )
        self._trim_history()

    def _trim_history(self):
        """
        修剪历史记录
        当记录数超过 max_history_length 时，移除最旧的记录
        """
        if len(self._history) > self.max_history_length:
            self.logger.warning(
                f"🤖 历史记录超出限制 ({len(self._history)}/{self.max_history_length})，正在修剪..."
            )
        while len(self._history) > self.max_history_length:
            removed_msg = self._history.pop(0)
            self.logger.debug(f"🤖 移除旧消息: {removed_msg.content[:20]}...")

    def clear_history(self):
        """清空历史记录"""
        self.logger.info(
            f"🤖 清空 Agent '{self.name}' 的历史记录 (当前数量: {len(self._history)})"
        )
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.llm.model})"
