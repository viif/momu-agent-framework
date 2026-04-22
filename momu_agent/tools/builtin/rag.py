"""RAG 工具"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.exceptions import ToolException
from ...core.llm import LLM
from ...rag.document import DocumentProcessor, create_document
from ...rag.pipeline import create_rag_pipeline, index_chunks, merge_snippets
from ...utils.config import Config
from ...utils.logger import get_logger
from ..base import Tool, ToolParameter


class RAGTool(Tool):
    """基于内置 RAG pipeline 的检索增强生成工具。"""

    def __init__(
        self,
        namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        max_chars: int = 3000,
        llm: LLM | None = None,
    ):
        super().__init__(
            name="rag",
            description=(
                "基于本地知识库的 RAG 工具，支持文档入库、检索、问答和统计。"
                "检索策略遵循“由简入繁”原则：默认优先使用基础检索；"
                "仅在基础检索结果不足或查询复杂模糊时，再尝试启用多查询扩展（MQE）或假设文档嵌入（HyDE）策略。"
            ),
        )
        self.logger = get_logger(__name__)
        self.namespace = namespace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_chars = max_chars
        self._llm = llm
        self._pipeline_cache: dict[tuple[str, int, int, int, bool], dict[str, Any]] = {}

    async def run(self, parameters: dict[str, Any]) -> str:
        """根据 action 分发并执行对应 RAG 操作。"""
        action = str(parameters.get("action", "")).strip()
        if not action:
            raise ToolException("必须提供 action 参数")

        # 按 action 分发到对应的 RAG 流程。
        if action == "add_document":
            return await self._add_document(parameters)
        if action == "add_text":
            return await self._add_text(parameters)
        if action == "search":
            return await self._search(parameters)
        if action == "ask":
            return await self._ask(parameters)
        if action == "stats":
            return await self._stats(parameters)

        raise ToolException(
            "不支持的 action: "
            f"{action}。可用 action: add_document, add_text, search, ask, stats"
        )

    def get_parameters(self) -> list[ToolParameter]:
        """返回工具参数定义。"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add_document、add_text、search、ask、stats",
                required=True,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="待导入的文件路径",
                required=False,
            ),
            ToolParameter(
                name="text",
                type="string",
                description="待导入的文本内容",
                required=False,
            ),
            ToolParameter(
                name="document_id",
                type="string",
                description="文本导入时使用的文档 ID",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索查询",
                required=False,
            ),
            ToolParameter(
                name="question",
                type="string",
                description="问答问题，search 时也可作为 query 别名",
                required=False,
            ),
            ToolParameter(
                name="namespace",
                type="string",
                description="RAG 命名空间",
                required=False,
                default="default",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="score_threshold",
                type="float",
                description="最小相似度阈值",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="enable_mqe",
                type="boolean",
                description="是否启用多查询扩展（MQE）",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="mqe_expansions",
                type="integer",
                description="MQE 生成改写查询数量",
                required=False,
                default=2,
            ),
            ToolParameter(
                name="enable_hyde",
                type="boolean",
                description="是否启用假设文档嵌入（HyDE）",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="candidate_pool_multiplier",
                type="integer",
                description="扩展检索候选池倍率",
                required=False,
                default=4,
            ),
            ToolParameter(
                name="chunk_size",
                type="integer",
                description="文本分块大小",
                required=False,
                default=1000,
            ),
            ToolParameter(
                name="chunk_overlap",
                type="integer",
                description="文本分块重叠长度",
                required=False,
                default=200,
            ),
            ToolParameter(
                name="max_chars",
                type="integer",
                description="问答时注入上下文的最大字符数",
                required=False,
                default=3000,
            ),
        ]

    async def _add_document(self, parameters: dict[str, Any]) -> str:
        """导入文件并完成入库。"""
        # 校验输入并组装入库参数。
        file_path = str(parameters.get("file_path", "")).strip()
        if not file_path:
            raise ToolException("add_document 需要提供 file_path")

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise ToolException(f"文件不存在: {file_path}")

        namespace = self._resolve_namespace(parameters)
        chunk_size = self._get_positive_int(
            parameters.get("chunk_size"), self.chunk_size, "chunk_size"
        )
        chunk_overlap = self._get_non_negative_int(
            parameters.get("chunk_overlap"), self.chunk_overlap, "chunk_overlap"
        )
        pipeline = self._create_pipeline(
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 调用 pipeline 完成切分、向量化与索引写入。
        try:
            chunk_count = await pipeline["add_documents"]([str(path)])
        except Exception as e:
            raise ToolException(f"添加文档失败: {e}") from e

        return (
            f"已添加文档: {path.name}\n命名空间: {namespace}\n分块数量: {chunk_count}"
        )

    async def _add_text(self, parameters: dict[str, Any]) -> str:
        """导入文本并完成入库。"""
        # 将内联文本标准化为文档并按切分策略入库。
        text = str(parameters.get("text", "")).strip()
        if not text:
            raise ToolException("add_text 需要提供非空 text")

        namespace = self._resolve_namespace(parameters)
        chunk_size = self._get_positive_int(
            parameters.get("chunk_size"), self.chunk_size, "chunk_size"
        )
        chunk_overlap = self._get_non_negative_int(
            parameters.get("chunk_overlap"), self.chunk_overlap, "chunk_overlap"
        )
        document_id = str(parameters.get("document_id", "")).strip() or None

        try:
            document = create_document(
                text,
                source=document_id or "inline_text",
                type="inline_text",
                rag_namespace=namespace,
            )
            if document_id:
                document.doc_id = document_id

            processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = processor.process_document(document)
            items = [
                {
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": {
                        **chunk.metadata,
                        "source": document_id or "inline_text",
                        "rag_namespace": namespace,
                    },
                }
                for chunk in chunks
                if chunk.content.strip()
            ]
            # 写入向量存储并返回实际入库分块数。
            chunk_count = await index_chunks(items, namespace=namespace)
        except Exception as e:
            raise ToolException(f"添加文本失败: {e}") from e

        return (
            f"已添加文本: {document.doc_id}\n"
            f"命名空间: {namespace}\n"
            f"分块数量: {chunk_count}"
        )

    async def _search(self, parameters: dict[str, Any]) -> str:
        """执行检索并返回格式化结果。"""
        # 参数归一化后执行向量检索，并格式化为可读结果。
        query = self._get_query(parameters)
        if not query:
            raise ToolException("search 需要提供 query 或 question")

        namespace = self._resolve_namespace(parameters)
        limit = self._get_positive_int(parameters.get("limit"), self.top_k, "limit")
        score_threshold = self._get_optional_float(
            parameters.get("score_threshold"), "score_threshold"
        )
        enable_mqe = self._get_bool(parameters.get("enable_mqe"), False, "enable_mqe")
        mqe_expansions = self._get_positive_int(
            parameters.get("mqe_expansions"), 2, "mqe_expansions"
        )
        enable_hyde = self._get_bool(
            parameters.get("enable_hyde"), False, "enable_hyde"
        )
        candidate_pool_multiplier = self._get_positive_int(
            parameters.get("candidate_pool_multiplier"),
            4,
            "candidate_pool_multiplier",
        )
        pipeline = self._create_pipeline(namespace=namespace, top_k=limit)

        try:
            results = await pipeline["search"](
                query,
                limit=limit,
                score_threshold=score_threshold,
                enable_mqe=enable_mqe,
                mqe_expansions=mqe_expansions,
                enable_hyde=enable_hyde,
                candidate_pool_multiplier=candidate_pool_multiplier,
            )
        except Exception as e:
            raise ToolException(f"搜索失败: {e}") from e

        return self._format_search_results(query, results)

    async def _ask(self, parameters: dict[str, Any]) -> str:
        """基于检索上下文生成回答。"""
        # 先检索相关片段，再将压缩上下文交给 LLM 生成回答。
        question = self._get_query(parameters)
        if not question:
            raise ToolException("ask 需要提供 question 或 query")

        namespace = self._resolve_namespace(parameters)
        limit = self._get_positive_int(parameters.get("limit"), self.top_k, "limit")
        max_chars = self._get_positive_int(
            parameters.get("max_chars"), self.max_chars, "max_chars"
        )
        score_threshold = self._get_optional_float(
            parameters.get("score_threshold"), "score_threshold"
        )
        enable_mqe = self._get_bool(parameters.get("enable_mqe"), False, "enable_mqe")
        mqe_expansions = self._get_positive_int(
            parameters.get("mqe_expansions"), 2, "mqe_expansions"
        )
        enable_hyde = self._get_bool(
            parameters.get("enable_hyde"), False, "enable_hyde"
        )
        candidate_pool_multiplier = self._get_positive_int(
            parameters.get("candidate_pool_multiplier"),
            4,
            "candidate_pool_multiplier",
        )
        pipeline = self._create_pipeline(namespace=namespace, top_k=limit)

        try:
            results = await pipeline["search"](
                question,
                limit=limit,
                score_threshold=score_threshold,
                enable_mqe=enable_mqe,
                mqe_expansions=mqe_expansions,
                enable_hyde=enable_hyde,
                candidate_pool_multiplier=candidate_pool_multiplier,
            )
        except Exception as e:
            raise ToolException(f"问答检索失败: {e}") from e

        if not results:
            return (
                f"知识库中没有找到与“{question}”相关的内容。"
                "可以尝试换个关键词，或先添加相关文档。"
            )

        # 合并候选片段，控制上下文长度，降低无关噪声。
        context = merge_snippets(results, max_chars=max_chars)
        llm = self._get_llm()

        try:
            answer = await llm.invoke(self._build_messages(question, context))
        except Exception as e:
            raise ToolException(f"生成回答失败: {e}") from e

        if not answer.strip():
            raise ToolException("生成回答失败: 模型返回内容为空")

        return self._format_answer(answer.strip(), results)

    async def _stats(self, parameters: dict[str, Any]) -> str:
        """返回指定命名空间的存储统计信息。"""
        namespace = self._resolve_namespace(parameters)
        pipeline = self._create_pipeline(namespace=namespace)

        try:
            stats = await pipeline["get_stats"]()
        except Exception as e:
            raise ToolException(f"获取统计失败: {e}") from e

        return self._format_stats(stats)

    async def close(self) -> None:
        """关闭当前工具已创建的所有 RAG pipeline 资源。"""
        for key, pipeline in list(self._pipeline_cache.items()):
            close_fn = pipeline.get("close")
            if close_fn is None:
                continue
            try:
                result = close_fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                self.logger.warning(f"关闭 RAG pipeline {key} 失败: {e}")

    def _create_pipeline(
        self,
        *,
        namespace: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """创建带默认参数的 RAG pipeline。"""
        resolved_chunk_size = chunk_size or self.chunk_size
        resolved_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        )
        resolved_top_k = top_k or self.top_k
        cache_key = (
            namespace,
            resolved_chunk_size,
            resolved_chunk_overlap,
            resolved_top_k,
            self._llm is not None,
        )
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        pipeline = create_rag_pipeline(
            namespace=namespace,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            top_k=resolved_top_k,
            llm=self._llm,
        )
        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def _get_llm(self) -> LLM:
        """获取或初始化 LLM 实例。"""
        if self._llm is not None:
            return self._llm

        try:
            config = Config.from_env()
            self._llm = LLM(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=int(config.timeout),
            )
            return self._llm
        except Exception as e:
            raise ToolException(f"初始化 LLM 失败: {e}") from e

    def _build_messages(self, question: str, context: str) -> list[dict[str, str]]:
        """构造问答请求消息。"""
        return [
            {
                "role": "system",
                "content": (
                    "你是一个严格基于检索上下文回答问题的知识助手。"
                    "如果上下文不足以支持回答，请明确说明。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"上下文：\n{context}\n\n"
                    "请基于上下文给出准确、简洁的回答。"
                ),
            },
        ]

    def _format_search_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> str:
        if not results:
            return f"未找到与“{query}”相关的内容。"

        lines = [f"检索到 {len(results)} 条结果："]
        for index, item in enumerate(results, start=1):
            metadata = dict(item.get("metadata") or {})
            source = str(metadata.get("source") or "unknown")
            score = float(item.get("score") or 0.0)
            content = self._summarize_content(
                str(item.get("content") or metadata.get("content") or "")
            )
            lines.append(f"[{index}] {Path(source).name} (score={score:.3f})")
            if content:
                lines.append(f"    {content}")

        return "\n".join(lines)

    def _format_answer(
        self,
        answer: str,
        results: list[dict[str, Any]],
    ) -> str:
        lines = [answer]
        seen: set[str] = set()
        references: list[str] = []

        for item in results:
            metadata = dict(item.get("metadata") or {})
            source = str(metadata.get("source") or "unknown")
            name = Path(source).name
            if name in seen:
                continue
            seen.add(name)
            references.append(f"- {name} (score={float(item.get('score') or 0.0):.3f})")
            if len(references) >= 5:
                break

        if references:
            lines.append("\n参考来源：")
            lines.extend(references)

        return "\n".join(lines)

    def _format_stats(self, stats: dict[str, Any]) -> str:
        namespace = str(stats.get("namespace") or self.namespace)
        vector = dict(stats.get("vector") or {})
        document = dict(stats.get("document") or {})

        lines = [
            "RAG 统计信息：",
            f"命名空间: {namespace}",
            f"向量存储: {vector.get('store_type', 'unknown')}",
            f"向量数量: {vector.get('points_count', vector.get('vectors_count', 0))}",
            f"文档存储: {document.get('store_type', 'unknown')}",
            f"记忆数量: {document.get('memories_count', 0)}",
        ]

        memory_types = document.get("memory_types")
        if memory_types:
            lines.append(f"记忆类型分布: {memory_types}")

        return "\n".join(lines)

    def _resolve_namespace(self, parameters: dict[str, Any]) -> str:
        namespace = str(parameters.get("namespace") or self.namespace).strip()
        return namespace or self.namespace

    def _get_query(self, parameters: dict[str, Any]) -> str:
        query = parameters.get("query") or parameters.get("question") or ""
        return str(query).strip()

    def _get_bool(self, value: Any, default: bool, name: str) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        raise ToolException(f"{name} 必须是布尔值")

    def _get_positive_int(self, value: Any, default: int, name: str) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as e:
            raise ToolException(f"{name} 必须是正整数") from e
        if parsed <= 0:
            raise ToolException(f"{name} 必须是正整数")
        return parsed

    def _get_non_negative_int(self, value: Any, default: int, name: str) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as e:
            raise ToolException(f"{name} 必须是非负整数") from e
        if parsed < 0:
            raise ToolException(f"{name} 必须是非负整数")
        return parsed

    def _get_optional_float(self, value: Any, name: str) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise ToolException(f"{name} 必须是数字") from e

    def _summarize_content(self, content: str, max_length: int = 200) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3] + "..."


async def rag_add_document(
    file_path: str,
    namespace: str = "default",
    **kwargs: Any,
) -> str:
    tool = RAGTool()
    return await tool.run(
        {
            "action": "add_document",
            "file_path": file_path,
            "namespace": namespace,
            **kwargs,
        }
    )


async def rag_add_text(
    text: str,
    namespace: str = "default",
    document_id: str | None = None,
    **kwargs: Any,
) -> str:
    tool = RAGTool()
    payload: dict[str, Any] = {
        "action": "add_text",
        "text": text,
        "namespace": namespace,
        **kwargs,
    }
    if document_id is not None:
        payload["document_id"] = document_id
    return await tool.run(payload)


async def rag_search(
    query: str,
    namespace: str = "default",
    **kwargs: Any,
) -> str:
    tool = RAGTool()
    return await tool.run(
        {
            "action": "search",
            "query": query,
            "namespace": namespace,
            **kwargs,
        }
    )


async def rag_ask(
    question: str,
    namespace: str = "default",
    **kwargs: Any,
) -> str:
    tool = RAGTool()
    return await tool.run(
        {
            "action": "ask",
            "question": question,
            "namespace": namespace,
            **kwargs,
        }
    )


async def rag_get_stats(namespace: str = "default", **kwargs: Any) -> str:
    tool = RAGTool()
    return await tool.run(
        {
            "action": "stats",
            "namespace": namespace,
            **kwargs,
        }
    )
