from typing import List, cast

import pytest

from momu_agent.agents.parser.tool_parser import ToolParser
from momu_agent.tools.base import ToolParameter
from momu_agent.tools.registry import ToolRegistry


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


class MockRecordTool:
    """模拟一个支持 action 的通用记录工具"""

    def __init__(self):
        self.name = "record"
        self.description = "通用记录工具，支持创建、读取、更新、删除、列表、搜索与摘要"
        self.params = [
            ToolParameter(
                name="action", type="string", description="动作", required=True
            ),
            ToolParameter(
                name="title", type="string", description="标题", required=False
            ),
            ToolParameter(
                name="content", type="string", description="内容", required=False
            ),
        ]

    def get_parameters(self) -> List[ToolParameter]:
        return self.params


class MockNoActionTool:
    """模拟一个不支持 action 的工具"""

    def __init__(self):
        self.name = "fetch"
        self.description = "抓取工具"
        self.params = [
            ToolParameter(name="url", type="string", description="地址", required=True)
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


def test_prepare_tool_task_with_function_tool(parser):
    """测试 prepare_tool_task 支持函数工具分支"""

    class MockRegistry:
        def get_tool(self, name):
            return None

        def get_function(self, name):
            return (lambda text: f"ok:{text}") if name == "func_tool" else None

        def get_all_tools(self):
            return []

    task = parser.prepare_tool_task("func_tool", '{"input": "hello"}', MockRegistry())

    assert task["tool_name"] == "func_tool"
    assert task["input_data"] == {"input": "hello"}


def test_prepare_tool_task_not_registered(parser):
    """测试 prepare_tool_task 在工具未注册时返回 error"""

    class MockRegistry:
        def get_tool(self, name):
            return None

        def get_function(self, name):
            return None

        def get_all_tools(self):
            return []

    task = parser.prepare_tool_task("missing", "{}", MockRegistry())

    assert task["tool_name"] == "missing"
    assert task["error"] == "工具未注册"


def test_extract_tool_calls_supports_list_in_json(parser):
    """测试 JSON 数组参数不会被 ] 提前截断"""
    text = '[TOOL_CALL:note:{"action": "create", "tags": ["demo", "examples"]}]'
    result = parser.extract_tool_calls(text)

    assert len(result) == 1
    assert result[0]["tool_name"] == "note"
    assert (
        result[0]["raw_params"] == '{"action": "create", "tags": ["demo", "examples"]}'
    )


def test_parse_parameters_supports_bare_action_word(parser):
    """测试兼容纯动作词参数"""
    assert parser.parse_parameters("list") == {"action": "list"}
    assert parser.parse_parameters("summary") == {"action": "summary"}
    assert parser.parse_parameters("stats") == {"action": "stats"}
    assert parser.parse_parameters("clear") == {"action": "clear"}


def test_prepare_tool_task_generic_alias_without_action_field():
    """测试通用别名：tool_suffix 在缺少 action 时注入 action。"""

    class MockRegistry:
        def __init__(self):
            self.record_tool = MockRecordTool()

        def get_tool(self, name):
            if name == "record":
                return self.record_tool
            return None

        def get_function(self, name):
            return None

        def get_all_tools(self):
            return [self.record_tool]

    parser = ToolParser(cast(ToolRegistry, MockRegistry()))
    task = parser.prepare_tool_task(
        "record_create", '{"title": "A", "content": "B"}', MockRegistry()
    )

    assert task["tool_name"] == "record"
    assert task["input_data"]["action"] == "create"
    assert task["input_data"]["title"] == "A"


def test_prepare_tool_task_generic_alias_keep_existing_action():
    """测试通用别名：若参数已有 action，则保留参数中的 action。"""

    class MockRegistry:
        def __init__(self):
            self.record_tool = MockRecordTool()

        def get_tool(self, name):
            if name == "record":
                return self.record_tool
            return None

        def get_function(self, name):
            return None

        def get_all_tools(self):
            return [self.record_tool]

    parser = ToolParser(cast(ToolRegistry, MockRegistry()))
    task = parser.prepare_tool_task(
        "record_list", '{"action": "search", "title": "A"}', MockRegistry()
    )

    assert task["tool_name"] == "record"
    assert task["input_data"]["action"] == "search"


def test_prepare_tool_task_no_alias_when_base_has_no_action_param():
    """测试 base 工具不含 action 参数时不做别名归一化。"""

    class MockRegistry:
        def __init__(self):
            self.fetch_tool = MockNoActionTool()

        def get_tool(self, name):
            if name == "fetch":
                return self.fetch_tool
            return None

        def get_function(self, name):
            return None

        def get_all_tools(self):
            return [self.fetch_tool]

    parser = ToolParser(cast(ToolRegistry, MockRegistry()))
    task = parser.prepare_tool_task(
        "fetch_list", '{"url": "https://a.com"}', MockRegistry()
    )

    assert task["tool_name"] == "fetch_list"
    assert task["error"] == "工具未注册"


def test_prepare_tool_task_no_alias_when_suffix_not_common_action():
    """测试后缀不在通用动作词集合时不做别名归一化。"""

    class MockRegistry:
        def __init__(self):
            self.record_tool = MockRecordTool()

        def get_tool(self, name):
            if name == "record":
                return self.record_tool
            return None

        def get_function(self, name):
            return None

        def get_all_tools(self):
            return [self.record_tool]

    parser = ToolParser(cast(ToolRegistry, MockRegistry()))
    task = parser.prepare_tool_task("record_custom", '{"title": "A"}', MockRegistry())

    assert task["tool_name"] == "record_custom"
    assert task["error"] == "工具未注册"
