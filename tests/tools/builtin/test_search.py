from unittest.mock import Mock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools import SearchTool


def _build_import_side_effect(
    tavily_client: Mock | None = None,
    serpapi_client: Mock | None = None,
    tavily_import_error: bool = False,
    serpapi_import_error: bool = False,
):
    tavily_client = tavily_client or Mock()
    serpapi_client = serpapi_client or Mock()

    def _import_module(name: str):
        if name == "tavily":
            if tavily_import_error:
                raise ImportError("No module named 'tavily'")
            module = Mock()
            module.TavilyClient = Mock(return_value=tavily_client)
            return module

        if name == "serpapi":
            if serpapi_import_error:
                raise ImportError("No module named 'serpapi'")
            module = Mock()
            module.Client = Mock(return_value=serpapi_client)
            return module

        raise ImportError(f"No module named '{name}'")

    return _import_module


class TestSearchTool:
    """测试 SearchTool 类"""

    def test_init_no_keys_raises_exception(self):
        """测试在没有 API Key 时初始化应抛出异常"""
        with pytest.raises(ToolException) as exc_info:
            SearchTool(backend="hybrid")

        assert "初始化失败" in str(exc_info.value)

    def test_init_specific_backend_missing_key(self):
        """测试指定后端但缺少对应 Key 时抛出异常"""
        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(serpapi_client=Mock())

            with pytest.raises(ToolException) as exc_info:
                SearchTool(backend="tavily", serpapi_key="fake_key")

        assert "配置错误" in str(exc_info.value)
        assert "tavily" in str(exc_info.value)

    def test_init_invalid_backend(self):
        """测试不支持的后端类型"""
        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(tavily_client=Mock())

            with pytest.raises(ToolException) as exc_info:
                SearchTool(backend="bing", tavily_api_key="fake")

        assert "不支持的搜索后端" in str(exc_info.value)

    def test_init_success(self):
        """测试成功初始化"""
        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(tavily_client=Mock())

            tool = SearchTool(backend="tavily", tavily_api_key="fake_key")

        assert tool.backend == "tavily"
        assert "tavily" in tool.available_backends

    def test_init_import_failure_marks_backend_unavailable(self):
        """测试导入失败时将对应后端标记为不可用"""
        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(
                tavily_import_error=True,
                serpapi_client=Mock(),
            )

            tool = SearchTool(
                backend="hybrid", tavily_api_key="fake", serpapi_key="fake"
            )

        assert "tavily" not in tool.available_backends
        assert "serpapi" in tool.available_backends

    @pytest.mark.asyncio
    async def test_run_empty_query(self):
        """测试空查询抛出异常"""
        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(tavily_client=Mock())
            tool = SearchTool(backend="tavily", tavily_api_key="fake")

        with pytest.raises(ToolException):
            await tool.run({"query": ""})

    @pytest.mark.asyncio
    async def test_search_tavily_success(self):
        """测试 Tavily 搜索成功"""
        mock_client_instance = Mock()
        mock_response = {
            "answer": "42",
            "results": [
                {
                    "title": "Life Answer",
                    "content": "The answer to everything",
                    "url": "http://example.com",
                }
            ],
        }
        mock_client_instance.search.return_value = mock_response

        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(
                tavily_client=mock_client_instance
            )
            tool = SearchTool(backend="tavily", tavily_api_key="fake")
            result = await tool.run({"query": "meaning of life"})

        assert "42" in result
        assert "Life Answer" in result
        mock_client_instance.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_serpapi_success(self):
        """测试 SerpApi 搜索成功"""
        mock_client_instance = Mock()
        mock_results = {
            "answer_box": {"answer": "2 + 2 = 4"},
            "organic_results": [
                {"title": "Math", "snippet": "Basic math", "link": "http://math.com"}
            ],
        }
        mock_client_instance.search.return_value = mock_results

        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(
                serpapi_client=mock_client_instance
            )
            tool = SearchTool(backend="serpapi", serpapi_key="fake")
            result = await tool.run({"query": "calculate 2+2"})

        assert "2 + 2 = 4" in result
        assert "Math" in result
        mock_client_instance.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_prefers_tavily(self):
        """测试混合模式优先使用 Tavily"""
        tavily_client_instance = Mock()
        serpapi_client_instance = Mock()
        tavily_client_instance.search.return_value = {
            "answer": "Tavily Result",
            "results": [],
        }

        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(
                tavily_client=tavily_client_instance,
                serpapi_client=serpapi_client_instance,
            )
            tool = SearchTool(
                backend="hybrid", tavily_api_key="fake", serpapi_key="fake"
            )
            result = await tool.run({"query": "test"})

        assert "Tavily Result" in result
        tavily_client_instance.search.assert_called_once()
        serpapi_client_instance.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_fallback_to_serpapi(self):
        """测试混合模式 Tavily 失败时回退到 SerpApi"""
        tavily_client_instance = Mock()
        tavily_client_instance.search.side_effect = Exception("API Error")

        serpapi_client_instance = Mock()
        serpapi_client_instance.search.return_value = {
            "organic_results": [
                {"title": "Fallback", "snippet": "SerpApi Result", "link": "..."}
            ]
        }

        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(
                tavily_client=tavily_client_instance,
                serpapi_client=serpapi_client_instance,
            )
            tool = SearchTool(
                backend="hybrid", tavily_api_key="fake", serpapi_key="fake"
            )
            result = await tool.run({"query": "test"})

        assert "Fallback" in result
        tavily_client_instance.search.assert_called_once()
        serpapi_client_instance.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_tavily_unavailable(self):
        """测试混合模式在没有 Tavily Key 时直接使用 SerpApi"""
        serpapi_client_instance = Mock()
        serpapi_client_instance.search.return_value = {"organic_results": []}

        with patch(
            "momu_agent.tools.builtin.search.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = _build_import_side_effect(
                serpapi_client=serpapi_client_instance
            )
            tool = SearchTool(backend="hybrid", serpapi_key="fake")
            await tool.run({"query": "test"})

        serpapi_client_instance.search.assert_called_once()
