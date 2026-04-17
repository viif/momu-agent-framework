"""搜索工具"""

import importlib
from typing import Any

from ...core.exceptions import ToolException
from ...utils.logger import get_logger
from ..base import Tool, ToolParameter


class SearchTool(Tool):
    """
    智能混合搜索工具

    支持多种搜索引擎后端，智能选择最佳搜索源：
    1. 混合模式 (hybrid) - 智能选择TAVILY或SERPAPI
    2. Tavily API (tavily) - 专业AI搜索
    3. SerpApi (serpapi) - 传统Google搜索
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_api_key: str | None = None,
        serpapi_key: str | None = None,
    ):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
        )

        self.logger = get_logger(__name__)

        self.backend = backend
        self.tavily_api_key = tavily_api_key
        self.serpapi_key = serpapi_key
        self.available_backends = []

        self.tavily_client: Any | None = None
        self.serpapi_client: Any | None = None

        self._setup_backends()

    def _setup_backends(self):
        """设置搜索后端并初始化客户端，如果没有可用后端则抛出异常"""
        has_valid_backend = False

        if self.tavily_api_key:
            try:
                tavily_module = importlib.import_module("tavily")
                tavily_client_cls = getattr(tavily_module, "TavilyClient")
                self.tavily_client = tavily_client_cls(api_key=self.tavily_api_key)
                self.available_backends.append("tavily")
                has_valid_backend = True
                self.logger.debug("🔧 Tavily 客户端已初始化")
            except ImportError:
                self.logger.warning("🔧 Tavily 依赖未安装，该后端不可用")
            except Exception as e:
                self.logger.warning(f"🔧 Tavily 客户端初始化失败，该后端不可用: {e}")
        else:
            self.logger.warning("🔧 Tavily API Key 未设置")

        if self.serpapi_key:
            try:
                serpapi_module = importlib.import_module("serpapi")
                serpapi_client_cls = getattr(serpapi_module, "Client")
                self.serpapi_client = serpapi_client_cls(api_key=self.serpapi_key)
                self.available_backends.append("serpapi")
                has_valid_backend = True
                self.logger.debug("🔧 SerpApi 客户端已初始化")
            except ImportError:
                self.logger.warning("🔧 SerpApi 依赖未安装，该后端不可用")
            except Exception as e:
                self.logger.warning(f"🔧 SerpApi 客户端初始化失败，该后端不可用: {e}")
        else:
            self.logger.warning("🔧 SerpApi API Key 未设置")

        if not has_valid_backend:
            error_msg = "初始化失败：未找到可用的搜索后端。请确保已安装依赖库，并在初始化时传入有效的 API Key。"
            self.logger.error(f"🔧 {error_msg}")
            raise ToolException(error_msg)

        # 确定最终使用的后端
        if self.backend == "hybrid":
            self.logger.info(
                f"🔧 混合搜索模式已启用，可用后端: {', '.join(self.available_backends)}"
            )
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            raise ToolException(
                "配置错误：后端设置为 'tavily'，但 Tavily 不可用（检查 Key 和依赖）"
            )
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            raise ToolException(
                "配置错误：后端设置为 'serpapi'，但 SerpApi 不可用（检查 Key 和依赖）"
            )
        elif self.backend not in ["tavily", "serpapi", "hybrid"]:
            raise ToolException(
                f"配置错误：不支持的搜索后端 '{self.backend}'。请使用 'tavily', 'serpapi', 或 'hybrid'"
            )

    def run(self, parameters: dict[str, Any]) -> str:
        """
        执行搜索

        Args:
            parameters: 包含input参数的字典

        Returns:
            搜索结果

        Raises:
            ToolException: 当搜索执行失败时抛出
        """
        query = parameters.get("query", parameters.get("input", "")).strip()
        if not query:
            raise ToolException("搜索查询不能为空")

        self.logger.info(f"🔧 正在执行搜索: {query}")

        try:
            if self.backend == "hybrid":
                return self._search_hybrid(query)
            elif self.backend == "tavily":
                return self._search_tavily(query)
            elif self.backend == "serpapi":
                return self._search_serpapi(query)
            else:
                raise ToolException(f"未知的后端类型: {self.backend}")
        except Exception as e:
            raise ToolException(f"搜索执行时发生错误: {str(e)}")

    def _search_hybrid(self, query: str) -> str:
        """混合搜索 - 智能选择最佳搜索源"""
        # 优先使用Tavily（AI优化的搜索）
        if "tavily" in self.available_backends:
            try:
                self.logger.info("🔧 使用Tavily进行AI优化搜索")
                return self._search_tavily(query)
            except Exception as e:
                self.logger.warning(f"🔧 Tavily搜索失败: {e}")
                # 如果Tavily失败，尝试SerpApi
                if "serpapi" in self.available_backends:
                    self.logger.info("🔧 切换到SerpApi搜索")
                    return self._search_serpapi(query)

        # 如果Tavily不可用，使用SerpApi
        elif "serpapi" in self.available_backends:
            self.logger.info("🔧 使用SerpApi进行Google搜索")
            return self._search_serpapi(query)

        raise ToolException("内部错误：混合搜索模式下没有可用的后端")

    def _search_tavily(self, query: str) -> str:
        """使用Tavily搜索"""
        if not self.tavily_client:
            raise ToolException("Tavily 客户端未初始化")

        response = self.tavily_client.search(
            query=query, search_depth="basic", include_answer=True, max_results=3
        )

        result = f"🎯 Tavily AI搜索结果：{response.get('answer', '未找到直接答案')}\n\n"

        for i, item in enumerate(response.get("results", [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    来源: {item.get('url', '')}\n\n"

        return result

    def _search_serpapi(self, query: str) -> str:
        """使用SerpApi搜索"""
        if not self.serpapi_client:
            raise ToolException("SerpApi 客户端未初始化")

        try:
            results = self.serpapi_client.search(
                q=query, engine="google", gl="cn", hl="zh-cn"
            )
        except Exception as e:
            raise ToolException(f"SerpApi 请求失败: {str(e)}")

        result_text = "🔍 SerpApi Google搜索结果：\n\n"

        if "answer_box" in results and "answer" in results["answer_box"]:
            result_text += f"💡 直接答案：{results['answer_box']['answer']}\n\n"

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            result_text += (
                f"📖 知识图谱：{results['knowledge_graph']['description']}\n\n"
            )

        if "organic_results" in results and results["organic_results"]:
            result_text += "🔗 相关结果：\n"
            for i, res in enumerate(results["organic_results"][:3], 1):
                result_text += f"[{i}] {res.get('title', '')}\n"
                result_text += f"    {res.get('snippet', '')}\n"
                result_text += f"    来源: {res.get('link', '')}\n\n"
            return result_text

        return f"对不起，没有找到关于 '{query}' 的信息。"

    def get_parameters(self) -> list[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="query", type="string", description="搜索查询关键词", required=True
            )
        ]


# 便捷函数
def search(query: str, backend: str = "hybrid") -> str:
    """
    便捷的搜索函数

    Args:
        query: 搜索查询关键词
        backend: 搜索后端 ("hybrid", "tavily", "serpapi")

    Returns:
        搜索结果

    Raises:
        ToolException: 如果配置无效或搜索失败
    """
    tool = SearchTool(backend=backend)
    return tool.run({"input": query})


# 专用搜索函数
def search_tavily(query: str) -> str:
    """使用Tavily进行AI优化搜索"""
    tool = SearchTool(backend="tavily")
    return tool.run({"input": query})


def search_serpapi(query: str) -> str:
    """使用SerpApi进行Google搜索"""
    tool = SearchTool(backend="serpapi")
    return tool.run({"input": query})


def search_hybrid(query: str) -> str:
    """智能混合搜索，自动选择最佳搜索源"""
    tool = SearchTool(backend="hybrid")
    return tool.run({"input": query})
