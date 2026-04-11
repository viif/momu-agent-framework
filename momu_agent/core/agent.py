"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional

from .llm import MomuAgentLLM
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: MomuAgentLLM,
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

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """
        添加消息到历史记录
        如果历史记录超过最大长度，自动移除最早的记录 (FIFO)
        """
        self._history.append(message)
        self._trim_history()

    def _trim_history(self):
        """
        修剪历史记录
        当记录数超过 max_history_length 时，移除最旧的记录
        """
        while len(self._history) > self.max_history_length:
            self._history.pop(0)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.llm.model})"
