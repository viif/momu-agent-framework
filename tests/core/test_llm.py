from unittest.mock import MagicMock, patch

import pytest

from momu_agent.core.exceptions import LLMException
from momu_agent.core.llm import MomuAgentLLM


def test_initialization_success():
    """测试 LLM 初始化成功"""
    llm = MomuAgentLLM(
        model="qwen-turbo", api_key="test_key", base_url="http://test_url"
    )
    assert llm.model == "qwen-turbo"
    assert llm.api_key == "test_key"


def test_initialization_missing_params():
    """测试缺少必填参数时抛出异常"""
    with pytest.raises(LLMException):
        MomuAgentLLM(model="", api_key="key", base_url="url")


@patch("momu_agent.core.llm.OpenAI")
def test_invoke(mock_openai_class):
    """测试非流式调用逻辑"""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "我是Mock AI"
    mock_client.chat.completions.create.return_value = mock_response

    llm = MomuAgentLLM(model="test", api_key="key", base_url="url")
    response = llm.invoke([{"role": "user", "content": "hi"}])

    assert response == "我是Mock AI"
    mock_client.chat.completions.create.assert_called_once()


@patch("momu_agent.core.llm.OpenAI")
def test_think_stream(mock_openai_class):
    """测试流式调用逻辑"""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices[0].delta.content = "Hello"

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices[0].delta.content = " World"

    mock_chunk_3 = MagicMock()
    mock_chunk_3.choices[0].delta.content = "!"

    mock_client.chat.completions.create.return_value = [
        mock_chunk_1,
        mock_chunk_2,
        mock_chunk_3,
    ]

    llm = MomuAgentLLM(model="test", api_key="key", base_url="url")
    chunks = list(llm.think([{"role": "user", "content": "hi"}]))

    assert chunks == ["Hello", " World", "!"]
