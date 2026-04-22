from unittest.mock import AsyncMock, Mock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools import RAGTool
from momu_agent.tools.builtin.rag import (
    rag_add_document,
    rag_add_text,
    rag_ask,
    rag_get_stats,
    rag_search,
)


class TestRAGTool:
    @pytest.fixture
    def rag_tool(self):
        return RAGTool()

    @pytest.mark.asyncio
    async def test_add_document_success(self, rag_tool, tmp_path):
        file_path = tmp_path / "knowledge.txt"
        file_path.write_text("hello rag", encoding="utf-8")

        pipeline = {"add_documents": AsyncMock(return_value=3)}
        with patch(
            "momu_agent.tools.builtin.rag.create_rag_pipeline",
            return_value=pipeline,
        ) as mock_create_pipeline:
            result = await rag_tool.run(
                {
                    "action": "add_document",
                    "file_path": str(file_path),
                    "namespace": "docs",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                }
            )

        assert "已添加文档: knowledge.txt" in result
        assert "命名空间: docs" in result
        assert "分块数量: 3" in result
        mock_create_pipeline.assert_called_once_with(
            namespace="docs",
            chunk_size=512,
            chunk_overlap=64,
            top_k=5,
            llm=None,
        )
        pipeline["add_documents"].assert_awaited_once_with([str(file_path)])

    @pytest.mark.asyncio
    async def test_add_document_missing_file_path_raises(self, rag_tool):
        with pytest.raises(ToolException, match="file_path"):
            await rag_tool.run({"action": "add_document"})

    @pytest.mark.asyncio
    async def test_add_text_success(self, rag_tool):
        mock_document = Mock(doc_id="doc-1")
        mock_chunk = Mock(
            chunk_id="chunk-1",
            content="first chunk",
            metadata={"chunk_index": 0},
        )

        with (
            patch(
                "momu_agent.tools.builtin.rag.create_document",
                return_value=mock_document,
            ) as mock_create_document,
            patch(
                "momu_agent.tools.builtin.rag.DocumentProcessor"
            ) as mock_processor_cls,
            patch(
                "momu_agent.tools.builtin.rag.index_chunks",
                new=AsyncMock(return_value=1),
            ) as mock_index_chunks,
        ):
            mock_processor = Mock()
            mock_processor.process_document.return_value = [mock_chunk]
            mock_processor_cls.return_value = mock_processor

            result = await rag_tool.run(
                {
                    "action": "add_text",
                    "text": "inline content",
                    "namespace": "notes",
                    "document_id": "doc-1",
                    "chunk_size": 256,
                    "chunk_overlap": 32,
                }
            )

        assert "已添加文本: doc-1" in result
        assert "命名空间: notes" in result
        assert "分块数量: 1" in result
        mock_create_document.assert_called_once_with(
            "inline content",
            source="doc-1",
            type="inline_text",
            rag_namespace="notes",
        )
        mock_processor_cls.assert_called_once_with(chunk_size=256, chunk_overlap=32)
        mock_index_chunks.assert_awaited_once_with(
            [
                {
                    "id": "chunk-1",
                    "content": "first chunk",
                    "metadata": {
                        "chunk_index": 0,
                        "source": "doc-1",
                        "rag_namespace": "notes",
                    },
                }
            ],
            namespace="notes",
        )

    @pytest.mark.asyncio
    async def test_add_text_empty_text_raises(self, rag_tool):
        with pytest.raises(ToolException, match="非空 text"):
            await rag_tool.run({"action": "add_text", "text": "   "})

    @pytest.mark.asyncio
    async def test_search_success_formats_results(self, rag_tool):
        pipeline = {
            "search": AsyncMock(
                return_value=[
                    {
                        "score": 0.91,
                        "content": "Python agent framework overview",
                        "metadata": {"source": "docs/intro.md"},
                    }
                ]
            )
        }
        with patch(
            "momu_agent.tools.builtin.rag.create_rag_pipeline",
            return_value=pipeline,
        ):
            result = await rag_tool.run(
                {
                    "action": "search",
                    "query": "python agent",
                    "namespace": "docs",
                    "limit": 3,
                    "score_threshold": 0.2,
                }
            )

        assert "检索到 1 条结果" in result
        assert "intro.md (score=0.910)" in result
        assert "Python agent framework overview" in result
        pipeline["search"].assert_awaited_once_with(
            "python agent",
            limit=3,
            score_threshold=0.2,
            enable_mqe=False,
            mqe_expansions=2,
            enable_hyde=False,
            candidate_pool_multiplier=4,
        )

    @pytest.mark.asyncio
    async def test_search_empty_query_raises(self, rag_tool):
        with pytest.raises(ToolException, match="query 或 question"):
            await rag_tool.run({"action": "search", "query": ""})

    @pytest.mark.asyncio
    async def test_ask_success_uses_retrieved_context(self, rag_tool):
        pipeline = {
            "search": AsyncMock(
                return_value=[
                    {
                        "score": 0.88,
                        "content": "MomuAgent supports async tools.",
                        "metadata": {"source": "docs/async.md"},
                    },
                    {
                        "score": 0.76,
                        "content": "ToolRegistry executes Tool.run asynchronously.",
                        "metadata": {"source": "docs/registry.md"},
                    },
                ]
            )
        }
        mock_llm = Mock()
        mock_llm.invoke = AsyncMock(return_value="它通过异步 Tool.run 接口执行工具。")

        with (
            patch(
                "momu_agent.tools.builtin.rag.create_rag_pipeline",
                return_value=pipeline,
            ),
            patch.object(RAGTool, "_get_llm", return_value=mock_llm),
        ):
            result = await rag_tool.run(
                {
                    "action": "ask",
                    "question": "工具是如何执行的？",
                    "namespace": "docs",
                    "limit": 2,
                    "max_chars": 120,
                }
            )

        assert "它通过异步 Tool.run 接口执行工具。" in result
        assert "参考来源：" in result
        assert "async.md (score=0.880)" in result
        assert "registry.md (score=0.760)" in result
        mock_llm.invoke.assert_awaited_once()
        messages = mock_llm.invoke.await_args.args[0]
        assert messages[1]["role"] == "user"
        assert "工具是如何执行的？" in messages[1]["content"]
        assert "MomuAgent supports async tools." in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_ask_no_hits_returns_hint_without_llm_call(self, rag_tool):
        pipeline = {"search": AsyncMock(return_value=[])}
        mock_llm = Mock()
        mock_llm.invoke = AsyncMock(return_value="should not be used")

        with (
            patch(
                "momu_agent.tools.builtin.rag.create_rag_pipeline",
                return_value=pipeline,
            ),
            patch.object(RAGTool, "_get_llm", return_value=mock_llm),
        ):
            result = await rag_tool.run(
                {"action": "ask", "question": "unknown question"}
            )

        assert "没有找到" in result
        mock_llm.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_stats_formats_pipeline_stats(self, rag_tool):
        pipeline = {
            "get_stats": AsyncMock(
                return_value={
                    "namespace": "docs",
                    "vector": {"store_type": "fake-vector", "points_count": 12},
                    "document": {
                        "store_type": "fake-doc",
                        "memories_count": 9,
                        "memory_types": {"rag_chunk": 9},
                    },
                }
            )
        }
        with patch(
            "momu_agent.tools.builtin.rag.create_rag_pipeline",
            return_value=pipeline,
        ):
            result = await rag_tool.run({"action": "stats", "namespace": "docs"})

        assert "RAG 统计信息" in result
        assert "命名空间: docs" in result
        assert "向量数量: 12" in result
        assert "记忆数量: 9" in result
        pipeline["get_stats"].assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, rag_tool):
        with pytest.raises(ToolException, match="不支持的 action"):
            await rag_tool.run({"action": "delete_all"})

    @pytest.mark.asyncio
    async def test_pipeline_error_is_wrapped_as_tool_exception(
        self, rag_tool, tmp_path
    ):
        file_path = tmp_path / "knowledge.txt"
        file_path.write_text("hello rag", encoding="utf-8")

        pipeline = {"add_documents": AsyncMock(side_effect=RuntimeError("boom"))}
        with patch(
            "momu_agent.tools.builtin.rag.create_rag_pipeline",
            return_value=pipeline,
        ):
            with pytest.raises(ToolException, match="添加文档失败: boom"):
                await rag_tool.run(
                    {"action": "add_document", "file_path": str(file_path)}
                )

    @pytest.mark.asyncio
    async def test_close_calls_all_cached_pipeline_close(self, rag_tool):
        close_a = AsyncMock(return_value=None)
        close_b = AsyncMock(return_value=None)
        rag_tool._pipeline_cache = {
            ("docs", 1000, 200, 5): {"close": close_a},
            ("notes", 512, 64, 3): {"close": close_b},
        }

        await rag_tool.close()

        close_a.assert_awaited_once_with()
        close_b.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_close_continues_when_single_pipeline_close_fails(self, rag_tool):
        bad_close = AsyncMock(side_effect=RuntimeError("boom"))
        good_close = AsyncMock(return_value=None)
        rag_tool._pipeline_cache = {
            ("docs", 1000, 200, 5): {"close": bad_close},
            ("notes", 1000, 200, 5): {"close": good_close},
        }

        with patch.object(rag_tool.logger, "warning") as mock_warning:
            await rag_tool.close()

        bad_close.assert_awaited_once_with()
        good_close.assert_awaited_once_with()
        mock_warning.assert_called_once()


class TestRAGConvenienceFunctions:
    @pytest.mark.asyncio
    async def test_rag_add_document_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.rag.RAGTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await rag_add_document("doc.txt", namespace="docs")

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with(
            {"action": "add_document", "file_path": "doc.txt", "namespace": "docs"}
        )

    @pytest.mark.asyncio
    async def test_rag_add_text_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.rag.RAGTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await rag_add_text("hello", namespace="notes", document_id="doc-1")

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with(
            {
                "action": "add_text",
                "text": "hello",
                "namespace": "notes",
                "document_id": "doc-1",
            }
        )

    @pytest.mark.asyncio
    async def test_rag_search_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.rag.RAGTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await rag_search("python", namespace="docs", limit=3)

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with(
            {
                "action": "search",
                "query": "python",
                "namespace": "docs",
                "limit": 3,
            }
        )

    @pytest.mark.asyncio
    async def test_rag_ask_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.rag.RAGTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await rag_ask("what is rag", namespace="docs", max_chars=500)

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with(
            {
                "action": "ask",
                "question": "what is rag",
                "namespace": "docs",
                "max_chars": 500,
            }
        )

    @pytest.mark.asyncio
    async def test_rag_get_stats_calls_tool_run(self):
        with patch("momu_agent.tools.builtin.rag.RAGTool") as mock_tool_cls:
            mock_tool = Mock()
            mock_tool.run = AsyncMock(return_value="ok")
            mock_tool_cls.return_value = mock_tool

            result = await rag_get_stats(namespace="docs")

        assert result == "ok"
        mock_tool.run.assert_awaited_once_with({"action": "stats", "namespace": "docs"})
