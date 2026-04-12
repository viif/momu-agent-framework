from typing import List

import pytest

from momu_agent.agents.parser.tool_parser import ToolParser
from momu_agent.tools.base import ToolParameter


class MockSearchTool:
    """模拟一个搜索工具，用于测试参数定义"""

    def __init__(self):
        self.params = [
            ToolParameter(
                name="query", type="string", description="搜索关键词", required=True
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="最大结果数",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="detailed",
                type="boolean",
                description="是否详细模式",
                required=False,
                default=False,
            ),
        ]

    def get_parameters(self) -> List[ToolParameter]:
        return self.params


class MockCalculatorTool:
    """模拟一个计算器工具"""

    def __init__(self):
        self.params = [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="number",
                description="超时时间",
                required=False,
                default=10.0,
            ),
        ]

    def get_parameters(self) -> List[ToolParameter]:
        return self.params


@pytest.fixture
def parser():
    """创建一个 ToolParser 实例供测试使用"""
    return ToolParser()


@pytest.fixture
def search_tool():
    """创建一个模拟的搜索工具实例"""
    return MockSearchTool()


@pytest.fixture
def calculator_tool():
    """创建一个模拟的计算器工具实例"""
    return MockCalculatorTool()


def test_extract_tool_calls_single(parser):
    """测试提取单个工具调用"""
    text = '这是文本 [TOOL_CALL:search:{"query": "Python"}] 结尾'
    result = parser.extract_tool_calls(text)

    assert len(result) == 1
    assert result[0]["tool_name"] == "search"
    assert result[0]["raw_params"] == '{"query": "Python"}'


def test_extract_tool_calls_multiple(parser):
    """测试提取多个工具调用"""
    text = """
    [TOOL_CALL:search:{"query": "天气"}]
    [TOOL_CALL:calculator:{"expression": "1+1"}]
    """
    result = parser.extract_tool_calls(text)

    assert len(result) == 2
    assert result[0]["tool_name"] == "search"
    assert result[1]["tool_name"] == "calculator"


def test_extract_tool_calls_duplicate_deduplication(parser):
    """测试去重逻辑：相同的调用只应出现一次"""
    text = "[TOOL_CALL:test:{}] [TOOL_CALL:test:{}]"
    result = parser.extract_tool_calls(text)

    assert len(result) == 1
    assert result[0]["tool_name"] == "test"


def test_extract_tool_calls_no_match(parser):
    """测试没有匹配到工具调用的情况"""
    text = "这是一段普通文本，没有工具调用。"
    result = parser.extract_tool_calls(text)
    assert len(result) == 0


def test_parse_parameters_valid_json(parser):
    """测试标准 JSON 解析"""
    raw = '{"query": "FastAPI", "count": 10}'
    result = parser.parse_parameters(raw)

    assert result["query"] == "FastAPI"
    assert result["count"] == 10
    assert isinstance(result["count"], int)


def test_parse_parameters_invalid_json_fix(parser):
    """测试无效 JSON 的自动修复（尾随逗号）"""
    raw = '{"query": "Error", "tags": ["a", "b"],}'
    result = parser.parse_parameters(raw)

    assert "query" in result
    assert result["tags"] == ["a", "b"]


def test_parse_parameters_empty_string(parser):
    """测试空字符串输入"""
    result = parser.parse_parameters("")
    assert result == {}


def test_parse_typed_parameters_string_to_int(parser, search_tool):
    """测试将字符串参数转换为整数"""
    raw = '{"max_results": "5"}'
    result = parser.parse_typed_parameters("search", raw, search_tool)

    assert result["max_results"] == 5
    assert isinstance(result["max_results"], int)


def test_parse_typed_parameters_string_to_bool(parser, search_tool):
    """测试字符串到布尔值的转换"""
    raw_true = '{"detailed": "true"}'
    result_true = parser.parse_typed_parameters("search", raw_true, search_tool)
    assert result_true["detailed"] is True

    raw_false = '{"detailed": "false"}'
    result_false = parser.parse_typed_parameters("search", raw_false, search_tool)
    assert result_false["detailed"] is False


def test_parse_typed_parameters_native_types(parser, search_tool):
    """测试原生类型（布尔值、数字）的保持"""
    raw = '{"detailed": true, "max_results": 3}'
    result = parser.parse_typed_parameters("search", raw, search_tool)

    assert result["detailed"] is True
    assert result["max_results"] == 3


def test_parse_typed_parameters_undefined_param(parser, search_tool):
    """测试传入工具未定义的参数（应保留但警告）"""
    raw = '{"query": "Known", "undefined_param": "Unknown Value"}'
    result = parser.parse_typed_parameters("search", raw, search_tool)

    assert "query" in result
    assert "undefined_param" in result
    assert result["undefined_param"] == "Unknown Value"


def test_parse_typed_parameters_type_conversion_error(parser, calculator_tool, caplog):
    """测试类型转换失败的情况（例如将非数字转为数字）"""
    raw = '{"timeout": "abc"}'
    result = parser.parse_typed_parameters("calculator", raw, calculator_tool)

    assert "类型转换失败" in caplog.text
    assert result["timeout"] == "abc"


def test_has_tool_calls(parser):
    """测试快速检测工具调用的方法"""
    text_with = "Hello [TOOL_CALL:test:{}] World"
    text_without = "Hello World"

    assert parser.has_tool_calls(text_with) is True
    assert parser.has_tool_calls(text_without) is False
