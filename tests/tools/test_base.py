import pytest

from momu_agent.tools.base import Tool, ToolParameter


class MockTool(Tool):
    """用于测试的具体工具实现"""

    def __init__(self):
        super().__init__(name="mock_tool", description="这是一个用于测试的工具")

    async def run(self, parameters: dict[str, object]) -> str:
        return f"MockTool executed with: {parameters}"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query", type="string", description="搜索关键词", required=True
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回数量限制",
                required=False,
                default=10,
            ),
        ]


class TestTool:
    """测试 Tool 基类"""

    @pytest.fixture
    def mock_tool(self):
        return MockTool()

    def test_init(self, mock_tool):
        assert mock_tool.name == "mock_tool"
        assert mock_tool.description == "这是一个用于测试的工具"
        assert mock_tool.logger is not None

    def test_validate_parameters_success(self, mock_tool):
        params = {"query": "python"}
        assert mock_tool.validate_parameters(params) is True

    def test_validate_parameters_with_optional(self, mock_tool):
        params = {"query": "python", "limit": 5}
        assert mock_tool.validate_parameters(params) is True

    def test_validate_parameters_missing_required(self, mock_tool):
        params = {"limit": 5}
        assert mock_tool.validate_parameters(params) is False

    def test_validate_parameters_empty(self, mock_tool):
        assert mock_tool.validate_parameters({}) is False

    @pytest.mark.asyncio
    async def test_close_default_noop(self, mock_tool):
        assert await mock_tool.close() is None

    def test_to_dict(self, mock_tool):
        result = mock_tool.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "mock_tool"
        assert result["description"] == "这是一个用于测试的工具"

        params = result["parameters"]
        assert len(params) == 2

        assert params[0]["name"] == "query"
        assert params[0]["required"] is True
        assert params[0]["default"] is None

        assert params[1]["name"] == "limit"
        assert params[1]["required"] is False
        assert params[1]["default"] == 10

    def test_str(self, mock_tool):
        assert str(mock_tool) == "Tool(name=mock_tool)"

    def test_repr(self, mock_tool):
        assert repr(mock_tool) == "Tool(name=mock_tool)"
