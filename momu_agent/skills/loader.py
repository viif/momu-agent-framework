"""Skills 加载器"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    """技能数据。"""

    name: str
    description: str
    body: str
    path: Path
    dir: Path

    @property
    def scripts(self) -> list[Path]:
        """获取 scripts/ 目录下的所有文件。"""
        scripts_dir = self.dir / "scripts"
        if not scripts_dir.exists():
            return []
        return [file for file in scripts_dir.rglob("*") if file.is_file()]

    @property
    def examples(self) -> list[Path]:
        """获取 examples/ 目录下的所有文件。"""
        examples_dir = self.dir / "examples"
        if not examples_dir.exists():
            return []
        return [file for file in examples_dir.rglob("*") if file.is_file()]

    @property
    def references(self) -> list[Path]:
        """获取 references/ 目录下的所有文件。"""
        references_dir = self.dir / "references"
        if not references_dir.exists():
            return []
        return [file for file in references_dir.rglob("*") if file.is_file()]


class SkillLoader:
    """技能加载器。"""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.skills_cache: dict[str, Skill] = {}
        self.metadata_cache: dict[str, dict[str, Any]] = {}

        self._scan_skills()

    def _scan_skills(self) -> None:
        """扫描 skills 目录并加载元数据。"""
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            metadata = self._parse_frontmatter_only(skill_md)
            if not metadata:
                continue

            name = str(metadata.get("name", skill_dir.name))
            self.metadata_cache[name] = {
                "name": name,
                "description": str(metadata.get("description", "")),
                "path": skill_md,
                "dir": skill_dir,
            }

    def _parse_frontmatter_only(self, path: Path) -> dict[str, Any] | None:
        """仅解析 YAML frontmatter。"""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return None

        yaml_str = match.group(1)
        try:
            metadata = yaml.safe_load(yaml_str) or {}
        except yaml.YAMLError:
            return None

        if "name" not in metadata or "description" not in metadata:
            return None

        return metadata

    def get_descriptions(self) -> str:
        """获取所有技能描述。"""
        if not self.metadata_cache:
            return "暂无可用技能"

        return "\n".join(
            f"- {name}: {skill['description']}"
            for name, skill in self.metadata_cache.items()
        )

    def get_skill(self, name: str) -> Skill | None:
        """按需加载完整技能。"""
        if name in self.skills_cache:
            return self.skills_cache[name]

        if name not in self.metadata_cache:
            return None

        metadata = self.metadata_cache[name]
        try:
            content = metadata["path"].read_text(encoding="utf-8")
        except Exception:
            return None

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter, body = match.groups()
        try:
            parsed_metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            return None

        skill = Skill(
            name=str(parsed_metadata.get("name", name)),
            description=str(parsed_metadata.get("description", "")),
            body=body.strip(),
            path=metadata["path"],
            dir=metadata["dir"],
        )
        self.skills_cache[name] = skill
        return skill

    def list_skills(self) -> list[str]:
        """列出所有可用技能名称。"""
        return list(self.metadata_cache.keys())

    def reload(self) -> None:
        """重新扫描技能目录。"""
        self.skills_cache.clear()
        self.metadata_cache.clear()
        self._scan_skills()
