import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.builtin.note import NoteTool


class TestNoteTool:
    @pytest.fixture
    def note_tool(self, tmp_path):
        return NoteTool(workspace=str(tmp_path / "notes"))

    @pytest.mark.asyncio
    async def test_create_success_writes_md_and_index(self, note_tool):
        result = await note_tool.run(
            {
                "action": "create",
                "title": "项目进展",
                "content": "已完成需求分析",
                "note_type": "task_state",
                "tags": ["phase1", "milestone"],
            }
        )

        assert "笔记创建成功" in result
        assert "ID: note_" in result

        list_result = await note_tool.run({"action": "list"})
        assert "[task_state] 项目进展" in list_result

    @pytest.mark.asyncio
    async def test_create_missing_title_or_content_raises(self, note_tool):
        with pytest.raises(ToolException, match="title 必须是非空字符串"):
            await note_tool.run({"action": "create", "content": "x"})

        with pytest.raises(ToolException, match="content 必须是非空字符串"):
            await note_tool.run({"action": "create", "title": "x"})

    @pytest.mark.asyncio
    async def test_read_success_returns_note_detail(self, note_tool):
        create_result = await note_tool.run(
            {
                "action": "create",
                "title": "设计结论",
                "content": "采用异步工具执行",
                "note_type": "conclusion",
            }
        )
        note_id = self._extract_note_id(create_result)

        read_result = await note_tool.run({"action": "read", "note_id": note_id})
        assert "笔记详情" in read_result
        assert "标题: 设计结论" in read_result
        assert "采用异步工具执行" in read_result

    @pytest.mark.asyncio
    async def test_read_missing_or_not_found_raises(self, note_tool):
        with pytest.raises(ToolException, match="note_id 必须是非空字符串"):
            await note_tool.run({"action": "read"})

        with pytest.raises(ToolException, match="未找到笔记"):
            await note_tool.run({"action": "read", "note_id": "note_not_exist"})

    @pytest.mark.asyncio
    async def test_update_success_updates_content(self, note_tool):
        create_result = await note_tool.run(
            {
                "action": "create",
                "title": "待办",
                "content": "先完成 A",
                "tags": ["todo"],
            }
        )
        note_id = self._extract_note_id(create_result)

        update_result = await note_tool.run(
            {
                "action": "update",
                "note_id": note_id,
                "title": "待办更新",
                "content": "先完成 B",
                "tags": ["todo", "updated"],
            }
        )
        assert f"笔记更新成功: {note_id}" == update_result

        read_result = await note_tool.run({"action": "read", "note_id": note_id})
        assert "标题: 待办更新" in read_result
        assert "先完成 B" in read_result
        assert "todo, updated" in read_result

    @pytest.mark.asyncio
    async def test_update_missing_note_id_or_fields_raises(self, note_tool):
        with pytest.raises(ToolException, match="note_id 必须是非空字符串"):
            await note_tool.run({"action": "update", "content": "x"})

        create_result = await note_tool.run(
            {
                "action": "create",
                "title": "x",
                "content": "y",
            }
        )
        note_id = self._extract_note_id(create_result)

        with pytest.raises(ToolException, match="至少需要提供一个可更新字段"):
            await note_tool.run({"action": "update", "note_id": note_id})

    @pytest.mark.asyncio
    async def test_delete_success_then_read_raises(self, note_tool):
        create_result = await note_tool.run(
            {
                "action": "create",
                "title": "临时笔记",
                "content": "待删除",
            }
        )
        note_id = self._extract_note_id(create_result)

        delete_result = await note_tool.run({"action": "delete", "note_id": note_id})
        assert delete_result == f"笔记已删除: {note_id}"

        with pytest.raises(ToolException, match="未找到笔记"):
            await note_tool.run({"action": "read", "note_id": note_id})

    @pytest.mark.asyncio
    async def test_list_supports_type_and_limit(self, note_tool):
        await note_tool.run(
            {
                "action": "create",
                "title": "结论1",
                "content": "内容1",
                "note_type": "conclusion",
            }
        )
        await note_tool.run(
            {
                "action": "create",
                "title": "任务1",
                "content": "内容2",
                "note_type": "task_state",
            }
        )

        result = await note_tool.run(
            {
                "action": "list",
                "note_type": "conclusion",
                "limit": 1,
            }
        )

        assert "[conclusion] 结论1" in result
        assert "任务1" not in result

    @pytest.mark.asyncio
    async def test_search_matches_title_content(self, note_tool):
        await note_tool.run(
            {
                "action": "create",
                "title": "RAG 方案",
                "content": "使用向量检索",
                "tags": ["retrieval"],
            }
        )

        by_title = await note_tool.run(
            {
                "action": "search",
                "query": "rag",
            }
        )
        assert "搜索结果" in by_title
        assert "RAG 方案" in by_title

        by_content = await note_tool.run(
            {
                "action": "search",
                "query": "向量检索",
            }
        )
        assert "RAG 方案" in by_content

    @pytest.mark.asyncio
    async def test_summary_returns_total_and_type_counts(self, note_tool):
        await note_tool.run(
            {
                "action": "create",
                "title": "A",
                "content": "a",
                "note_type": "conclusion",
            }
        )
        await note_tool.run(
            {
                "action": "create",
                "title": "B",
                "content": "b",
                "note_type": "task_state",
            }
        )

        result = await note_tool.run({"action": "summary"})
        assert "笔记摘要" in result
        assert "总笔记数: 2" in result
        assert "- conclusion: 1" in result
        assert "- task_state: 1" in result

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, note_tool):
        with pytest.raises(ToolException, match="不支持的 action"):
            await note_tool.run({"action": "unsupported"})

    @pytest.mark.asyncio
    async def test_limit_or_tags_invalid_raises(self, note_tool):
        with pytest.raises(ToolException, match="limit 必须是正整数"):
            await note_tool.run({"action": "list", "limit": 0})

        with pytest.raises(ToolException, match="tags 必须是字符串列表"):
            await note_tool.run(
                {
                    "action": "create",
                    "title": "x",
                    "content": "y",
                    "tags": "invalid",
                }
            )

    @staticmethod
    def _extract_note_id(result: str) -> str:
        for line in result.splitlines():
            if line.startswith("ID: "):
                return line.replace("ID: ", "").strip()
        raise AssertionError("创建结果中未找到 ID")
