from pathlib import Path

from momu_agent.skills import SkillLoader


def write_skill(root: Path, name: str, description: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


class TestSkillLoader:
    def test_empty_dir_returns_empty_state(self, tmp_path):
        loader = SkillLoader(tmp_path)

        assert loader.list_skills() == []
        assert loader.get_descriptions() == "暂无可用技能"

    def test_scan_and_get_skill_success(self, tmp_path):
        write_skill(tmp_path, "pdf", "处理 PDF 文件", "请分析 PDF 内容")

        loader = SkillLoader(tmp_path)

        assert loader.list_skills() == ["pdf"]
        assert loader.get_descriptions() == "- pdf: 处理 PDF 文件"

        skill = loader.get_skill("pdf")
        assert skill is not None
        assert skill.name == "pdf"
        assert skill.description == "处理 PDF 文件"
        assert skill.body == "请分析 PDF 内容"
        assert skill.path == tmp_path / "pdf" / "SKILL.md"
        assert skill.dir == tmp_path / "pdf"

    def test_resource_properties_return_files(self, tmp_path):
        skill_dir = write_skill(tmp_path, "python", "Python 开发", "执行 $ARGUMENTS")
        script_file = skill_dir / "scripts" / "run.py"
        example_file = skill_dir / "examples" / "demo.md"
        reference_file = skill_dir / "references" / "guide.txt"

        script_file.parent.mkdir(parents=True, exist_ok=True)
        example_file.parent.mkdir(parents=True, exist_ok=True)
        reference_file.parent.mkdir(parents=True, exist_ok=True)

        script_file.write_text("print('ok')", encoding="utf-8")
        example_file.write_text("demo", encoding="utf-8")
        reference_file.write_text("guide", encoding="utf-8")

        loader = SkillLoader(tmp_path)
        skill = loader.get_skill("python")

        assert skill is not None
        assert skill.scripts == [script_file]
        assert skill.examples == [example_file]
        assert skill.references == [reference_file]

    def test_get_skill_returns_none_when_missing(self, tmp_path):
        loader = SkillLoader(tmp_path)

        assert loader.get_skill("missing") is None

    def test_reload_refreshes_metadata_and_skill_cache(self, tmp_path):
        skill_dir = write_skill(tmp_path, "writer", "初始描述", "初始内容")
        loader = SkillLoader(tmp_path)

        first_skill = loader.get_skill("writer")
        assert first_skill is not None
        assert first_skill.description == "初始描述"
        assert first_skill.body == "初始内容"

        (skill_dir / "SKILL.md").write_text(
            "---\nname: writer\ndescription: 更新描述\n---\n更新内容\n",
            encoding="utf-8",
        )

        cached_skill = loader.get_skill("writer")
        assert cached_skill is first_skill
        assert cached_skill.description == "初始描述"
        assert cached_skill.body == "初始内容"

        loader.reload()
        reloaded_skill = loader.get_skill("writer")
        assert reloaded_skill is not None
        assert reloaded_skill is not first_skill
        assert reloaded_skill.description == "更新描述"
        assert reloaded_skill.body == "更新内容"

    def test_invalid_frontmatter_skill_is_skipped(self, tmp_path):
        skill_dir = tmp_path / "broken"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: broken\n---\n缺少 description\n",
            encoding="utf-8",
        )

        loader = SkillLoader(tmp_path)

        assert loader.list_skills() == []
        assert loader.get_skill("broken") is None
