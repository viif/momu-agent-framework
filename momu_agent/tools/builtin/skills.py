"""SkillsTool - 技能加载工具"""

from __future__ import annotations

from typing import Any

from ...core.exceptions import ToolException
from ...skills import Skill, SkillLoader
from ..base import Tool, ToolParameter


class SkillsTool(Tool):
    """加载并返回技能内容。"""

    def __init__(
        self,
        skills_dir: str = "./skills",
        loader: SkillLoader | None = None,
    ) -> None:
        self.skill_loader = loader or SkillLoader(skills_dir)
        descriptions = self.skill_loader.get_descriptions()

        super().__init__(
            name="skills",
            description=(
                "加载本地技能内容。参数：action（必填，list/get/reload）；"
                "name（get 时必填，技能名称）；args（可选，用于替换技能中的 $ARGUMENTS）。"
                f"可用技能：\n{descriptions}"
            ),
        )

    async def run(self, parameters: dict[str, Any]) -> str:
        action = str(parameters.get("action", "")).strip()
        if not action:
            raise ToolException("必须提供 action 参数")

        if action == "list":
            return self.skill_loader.get_descriptions()
        if action == "get":
            return self._get_skill_content(parameters)
        if action == "reload":
            self.skill_loader.reload()
            return f"技能已重新加载，共 {len(self.skill_loader.list_skills())} 个技能"

        raise ToolException(
            f"不支持的 action: {action}。可用 action: list, get, reload"
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: list(列出), get(获取), reload(重载)",
                required=True,
            ),
            ToolParameter(
                name="name",
                type="string",
                description="技能名称（get 时必填）",
                required=False,
            ),
            ToolParameter(
                name="args",
                type="string",
                description="可选参数，将替换技能内容中的 $ARGUMENTS",
                required=False,
                default="",
            ),
        ]

    def _get_skill_content(self, parameters: dict[str, Any]) -> str:
        name = str(parameters.get("name", "")).strip()
        if not name:
            raise ToolException("get 操作必须提供 name 参数")

        args = str(parameters.get("args", ""))
        skill = self.skill_loader.get_skill(name)
        if not skill:
            available = ", ".join(self.skill_loader.list_skills()) or "无"
            raise ToolException(f"技能 '{name}' 不存在。可用技能：{available}")

        content = skill.body.replace("$ARGUMENTS", args)
        resources_hint = self._get_resources_hint(skill)

        return (
            f'<skill-loaded name="{skill.name}">\n'
            f"{content}"
            f"{resources_hint}\n"
            f"</skill-loaded>\n\n"
            f"技能已加载：{skill.name}\n"
            f"描述：{skill.description}\n"
            "请严格遵循上述技能说明来完成任务。"
        )

    def _get_resources_hint(self, skill: Skill) -> str:
        resources: list[str] = []

        for folder, label in [
            ("scripts", "脚本"),
            ("references", "参考文档"),
            ("examples", "示例"),
        ]:
            folder_path = skill.dir / folder
            if not folder_path.exists():
                continue

            files = [file for file in folder_path.rglob("*") if file.is_file()]
            if not files:
                continue

            file_list = ", ".join(file.name for file in files[:5])
            if len(files) > 5:
                file_list += f" 等 {len(files)} 个文件"
            resources.append(f"  - {label}：{file_list}")

        if not resources:
            return ""

        return "\n\n**可用资源**：\n" + "\n".join(resources)
