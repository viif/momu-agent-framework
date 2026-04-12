from unittest.mock import Mock

from momu_agent.core.agent import Agent
from momu_agent.core.message import Message


class MockAgent(Agent):
    """用于测试的 Mock Agent，实现抽象方法"""

    async def run(self, input_text: str, **kwargs) -> str:
        return f"Mock response to: {input_text}"


class TestAgent:
    """测试 Agent 基类"""

    def setup_method(self):
        """在每个测试方法运行前执行"""
        self.mock_llm = Mock()
        self.mock_llm.model = "qwen-test"

    def _create_agent(self, max_history_length=100):
        """辅助方法：创建 Mock Agent"""
        return MockAgent(
            name="TestAgent",
            llm=self.mock_llm,
            system_prompt="You are a helpful assistant.",
            max_history_length=max_history_length,
        )

    def test_init(self):
        """测试 Agent 初始化"""
        agent = self._create_agent()

        assert agent.name == "TestAgent"
        assert agent.llm.model == "qwen-test"
        assert agent.system_prompt == "You are a helpful assistant."
        assert agent.max_history_length == 100
        assert agent._history == []

    def test_add_message(self):
        """测试添加单条消息"""
        agent = self._create_agent()
        msg = Message(role="user", content="Hello")

        agent.add_message(msg)

        assert len(agent._history) == 1
        assert agent._history[0].content == "Hello"

    def test_get_history_returns_copy(self):
        """测试获取历史记录返回的是副本，修改副本不影响原始数据"""
        agent = self._create_agent()
        msg = Message(role="user", content="Hello")
        agent.add_message(msg)

        history_copy = agent.get_history()
        assert history_copy == agent._history

        history_copy.clear()

        assert len(agent._history) == 1

    def test_clear_history(self):
        """测试清空历史记录"""
        agent = self._create_agent()
        agent.add_message(Message(role="user", content="Hello"))
        agent.add_message(Message(role="assistant", content="Hi"))

        assert len(agent._history) == 2

        agent.clear_history()

        assert len(agent._history) == 0

    def test_history_trimming(self):
        """测试当历史记录超过限制时，自动移除最早的记录"""
        # 设置最大长度为 3
        agent = self._create_agent(max_history_length=3)

        # 添加 5 条消息
        for i in range(5):
            agent.add_message(Message(role="user", content=f"Message {i}"))

        # 检查长度是否被限制在 3
        assert len(agent._history) == 3

        # 检查保留的是否是最新的 3 条 (Message 2, 3, 4)
        # 最早的 Message 0 和 Message 1 应该被移除
        assert agent._history[0].content == "Message 2"
        assert agent._history[1].content == "Message 3"
        assert agent._history[2].content == "Message 4"

    def test_history_trimming_edge_case(self):
        """测试边界情况：刚好达到最大长度"""
        agent = self._create_agent(max_history_length=2)

        agent.add_message(Message(role="user", content="Msg 1"))
        agent.add_message(Message(role="user", content="Msg 2"))

        # 刚好达到限制，不应移除
        assert len(agent._history) == 2
        assert agent._history[0].content == "Msg 1"

    def test_str(self):
        """测试 __str__ 方法"""
        agent = self._create_agent()
        assert str(agent) == "Agent(name=TestAgent, model=qwen-test)"
