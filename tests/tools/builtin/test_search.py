from unittest.mock import Mock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools import SearchTool


class TestSearchTool:
    """测试 SearchTool 类"""

    def test_init_no_keys_raises_exception(self):
        """测试在没有 API Key 时初始化应抛出异常"""
        with pytest.raises(ToolException) as exc_info:
            SearchTool(backend="hybrid")

        assert "初始化失败" in str(exc_info.value)

    def test_init_specific_backend_missing_key(self):
        """测试指定后端但缺少对应 Key 时抛出异常"""
        with pytest.raises(ToolException) as exc_info:
            SearchTool(backend="tavily", serpapi_key="fake_key")

        assert "配置错误" in str(exc_info.value)
        assert "tavily" in str(exc_info.value)

    def test_init_invalid_backend(self):
        """测试不支持的后端类型"""
        with pytest.raises(ToolException) as exc_info:
            SearchTool(backend="bing", tavily_api_key="fake")

        assert "不支持的搜索后端" in str(exc_info.value)

    def test_init_success(self):
        """测试成功初始化"""
        tool = SearchTool(backend="tavily", tavily_api_key="fake_key")
        assert tool.backend == "tavily"
        assert "tavily" in tool.available_backends

    def test_run_empty_query(self):
        """测试空查询抛出异常"""
        tool = SearchTool(backend="tavily", tavily_api_key="fake")

        with pytest.raises(ToolException):
            tool.run({"query": ""})

    @patch("momu_agent.tools.builtin.search.TavilyClient")
    def test_search_tavily_success(self, mock_tavily_class):
        """测试 Tavily 搜索成功"""
        mock_client_instance = Mock()
        mock_tavily_class.return_value = mock_client_instance

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

        tool = SearchTool(backend="tavily", tavily_api_key="fake")
        result = tool.run({"query": "meaning of life"})

        assert "42" in result
        assert "Life Answer" in result
        mock_client_instance.search.assert_called_once()

    @patch("momu_agent.tools.builtin.search.GoogleSearchClient")
    def test_search_serpapi_success(self, mock_serp_class):
        """测试 SerpApi 搜索成功"""
        mock_client_instance = Mock()
        mock_serp_class.return_value = mock_client_instance

        mock_results = {
            "answer_box": {"answer": "2 + 2 = 4"},
            "organic_results": [
                {"title": "Math", "snippet": "Basic math", "link": "http://math.com"}
            ],
        }
        mock_client_instance.search.return_value = mock_results

        tool = SearchTool(backend="serpapi", serpapi_key="fake")
        result = tool.run({"query": "calculate 2+2"})

        assert "2 + 2 = 4" in result
        assert "Math" in result
        mock_client_instance.search.assert_called_once()

    @patch("momu_agent.tools.builtin.search.TavilyClient")
    def test_hybrid_prefers_tavily(self, mock_tavily_class):
        """测试混合模式优先使用 Tavily"""
        mock_client_instance = Mock()
        mock_tavily_class.return_value = mock_client_instance
        mock_client_instance.search.return_value = {
            "answer": "Tavily Result",
            "results": [],
        }

        tool = SearchTool(backend="hybrid", tavily_api_key="fake", serpapi_key="fake")
        result = tool.run({"query": "test"})

        assert "Tavily Result" in result
        mock_client_instance.search.assert_called_once()

    @patch("momu_agent.tools.builtin.search.GoogleSearchClient")
    @patch("momu_agent.tools.builtin.search.TavilyClient")
    def test_hybrid_fallback_to_serpapi(self, mock_tavily_class, mock_serp_class):
        """测试混合模式 Tavily 失败时回退到 SerpApi"""
        mock_tavily_instance = Mock()
        mock_tavily_class.return_value = mock_tavily_instance
        mock_tavily_instance.search.side_effect = Exception("API Error")

        mock_serp_instance = Mock()
        mock_serp_class.return_value = mock_serp_instance
        mock_serp_instance.search.return_value = {
            "organic_results": [
                {"title": "Fallback", "snippet": "SerpApi Result", "link": "..."}
            ]
        }

        tool = SearchTool(backend="hybrid", tavily_api_key="fake", serpapi_key="fake")
        result = tool.run({"query": "test"})

        assert "Fallback" in result
        mock_tavily_instance.search.assert_called_once()
        mock_serp_instance.search.assert_called_once()

    @patch("momu_agent.tools.builtin.search.GoogleSearchClient")
    def test_hybrid_tavily_unavailable(self, mock_serp_class):
        """测试混合模式在没有 Tavily Key 时直接使用 SerpApi"""
        mock_serp_instance = Mock()
        mock_serp_class.return_value = mock_serp_instance
        mock_serp_instance.search.return_value = {"organic_results": []}

        tool = SearchTool(backend="hybrid", serpapi_key="fake")
        tool.run({"query": "test"})

        mock_serp_instance.search.assert_called_once()
