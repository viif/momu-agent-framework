import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.builtin.skills import SkillsTool


def write_skill(root, name, description, body):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


class TestSkillsTool:
    @pytest.fixture
    def skills_tool(self, tmp_path):
        write_skill(tmp_path, "pdf", "处理 PDF 文件", "请处理 $ARGUMENTS")
        return SkillsTool(skills_dir=str(tmp_path))

    @pytest.mark.asyncio
    async def test_list_action_returns_skill_descriptions(self, skills_tool):
        result = await skills_tool.run({"action": "list"})

        assert result == "- pdf: 处理 PDF 文件"

    @pytest.mark.asyncio
    async def test_get_action_returns_loaded_skill_content(self, skills_tool):
        result = await skills_tool.run(
            {"action": "get", "name": "pdf", "args": "季度报告.pdf"}
        )

        assert '<skill-loaded name="pdf">' in result
        assert "请处理 季度报告.pdf" in result
        assert "技能已加载：pdf" in result
        assert "描述：处理 PDF 文件" in result

    @pytest.mark.asyncio
    async def test_get_action_includes_resources_hint(self, tmp_path):
        skill_dir = write_skill(tmp_path, "python", "Python 开发", "执行脚本")
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "run.py").write_text("print('ok')", encoding="utf-8")

        tool = SkillsTool(skills_dir=str(tmp_path))
        result = await tool.run({"action": "get", "name": "python"})

        assert "**可用资源**" in result
        assert "run.py" in result

    @pytest.mark.asyncio
    async def test_reload_action_refreshes_loader_cache(self, tmp_path):
        skill_dir = write_skill(tmp_path, "writer", "初始描述", "初始内容")
        tool = SkillsTool(skills_dir=str(tmp_path))

        first_result = await tool.run({"action": "get", "name": "writer"})
        assert "初始内容" in first_result

        (skill_dir / "SKILL.md").write_text(
            "---\nname: writer\ndescription: 更新描述\n---\n更新内容\n",
            encoding="utf-8",
        )

        cached_result = await tool.run({"action": "get", "name": "writer"})
        assert "初始内容" in cached_result

        reload_result = await tool.run({"action": "reload"})
        assert "技能已重新加载，共 1 个技能" == reload_result

        refreshed_result = await tool.run({"action": "get", "name": "writer"})
        assert "更新内容" in refreshed_result
        assert "更新描述" in refreshed_result

    @pytest.mark.asyncio
    async def test_missing_action_raises(self, skills_tool):
        with pytest.raises(ToolException, match="必须提供 action 参数"):
            await skills_tool.run({})

    @pytest.mark.asyncio
    async def test_missing_name_for_get_raises(self, skills_tool):
        with pytest.raises(ToolException, match="get 操作必须提供 name 参数"):
            await skills_tool.run({"action": "get"})

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, skills_tool):
        with pytest.raises(ToolException, match="不支持的 action"):
            await skills_tool.run({"action": "unsupported"})

    @pytest.mark.asyncio
    async def test_missing_skill_raises(self, skills_tool):
        with pytest.raises(ToolException, match="技能 'missing' 不存在"):
            await skills_tool.run({"action": "get", "name": "missing"})

    def test_get_parameters(self, skills_tool):
        parameters = skills_tool.get_parameters()

        assert [parameter.name for parameter in parameters] == [
            "action",
            "name",
            "args",
        ]
        assert parameters[0].required is True
        assert parameters[1].required is False
        assert parameters[2].default == ""
