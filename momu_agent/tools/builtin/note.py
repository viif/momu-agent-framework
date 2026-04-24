"""NoteTool - 结构化笔记工具

为Agent提供结构化笔记能力，支持：
- 创建/读取/更新/删除笔记
- 按类型组织（任务状态、结论、阻塞项、行动计划等）
- 持久化存储（Markdown格式，带YAML前置元数据）
- 搜索与过滤

使用场景：
- 长时程任务的状态跟踪
- 关键结论与依赖记录
- 待办事项与行动计划
- 项目知识沉淀

笔记格式示例：
```markdown
---
id: note_20250118_120000_0
title: 项目进展
type: task_state
tags: [milestone, phase1]
created_at: 2025-01-18T12:00:00
updated_at: 2025-01-18T12:00:00
---

# 项目进展

已完成需求分析，下一步：设计方案

## 关键里程碑
- [x] 需求收集
- [ ] 方案设计
```
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.exceptions import ToolException
from ...utils.logger import get_logger
from ..base import Tool, ToolParameter


class NoteTool(Tool):
    """结构化笔记工具。"""

    NOTE_TYPES = {
        "task_state",
        "conclusion",
        "blocker",
        "action",
        "reference",
        "general",
    }

    def __init__(
        self,
        workspace: str = "./notes",
        auto_backup: bool = True,
        max_notes: int = 1000,
    ) -> None:
        super().__init__(
            name="note",
            description="结构化笔记工具，支持创建、读取、更新、删除、列表、搜索与摘要",
        )
        self.logger = get_logger(__name__)
        self.workspace = Path(workspace)
        self.auto_backup = auto_backup
        self.max_notes = max_notes
        self.index_file = self.workspace / "notes_index.json"

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.notes_index: dict[str, Any] = self._new_index()
        self._load_index()

    async def run(self, parameters: dict[str, Any]) -> str:
        action = str(parameters.get("action", "")).strip()
        if not action:
            raise ToolException("必须提供 action 参数")

        self.logger.info(
            "🔧 NoteTool执行: action=%s keys=%s",
            action,
            sorted(parameters.keys()),
        )

        if action == "create":
            return self._create_note(parameters)
        if action == "read":
            return self._read_note(parameters)
        if action == "update":
            return self._update_note(parameters)
        if action == "delete":
            return self._delete_note(parameters)
        if action == "list":
            return self._list_notes(parameters)
        if action == "search":
            return self._search_notes(parameters)
        if action == "summary":
            return self._get_summary()

        raise ToolException(
            "不支持的 action: "
            f"{action}。可用 action: create, read, update, delete, list, search, summary"
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "操作类型: create(创建), read(读取), update(更新), "
                    "delete(删除), list(列表), search(搜索), summary(摘要)"
                ),
                required=True,
            ),
            ToolParameter(
                name="title",
                type="string",
                description="笔记标题（create/update 时可用）",
                required=False,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="笔记内容（create/update 时可用）",
                required=False,
            ),
            ToolParameter(
                name="note_type",
                type="string",
                description=(
                    "笔记类型: task_state(任务状态), conclusion(结论), "
                    "blocker(阻塞项), action(行动计划), reference(参考), general(通用)"
                ),
                required=False,
                default="general",
            ),
            ToolParameter(
                name="tags",
                type="array",
                description="标签列表（可选）",
                required=False,
            ),
            ToolParameter(
                name="note_id",
                type="string",
                description="笔记 ID（read/update/delete 时必需）",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词（search 时必需）",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量限制（默认 10）",
                required=False,
                default=10,
            ),
        ]

    def _create_note(self, params: dict[str, Any]) -> str:
        title = self._as_non_empty_str(params.get("title"), "title")
        content = self._as_non_empty_str(params.get("content"), "content")
        note_type = self._as_note_type(
            params.get("note_type"), "note_type", default="general"
        )
        tags = self._as_tags(params.get("tags"))

        self.logger.debug(
            "🔧 创建笔记请求: title=%s type=%s content_len=%s tags_count=%s",
            title,
            note_type,
            len(content),
            len(tags),
        )

        if len(self.notes_index["notes"]) >= self.max_notes:
            raise ToolException(f"笔记数量已达上限 ({self.max_notes})")

        note_id = self._generate_note_id()
        while self._get_note_path(note_id).exists():
            note_id = self._generate_note_id()

        now = datetime.now().isoformat()
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "type": note_type,
            "tags": tags,
            "created_at": now,
            "updated_at": now,
            "metadata": {
                "word_count": len(content),
                "status": "active",
            },
        }

        note_path = self._get_note_path(note_id)
        note_path.write_text(self._note_to_markdown(note), encoding="utf-8")

        self.notes_index["notes"].append(
            {
                "id": note_id,
                "title": title,
                "type": note_type,
                "tags": tags,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._save_index()
        self.logger.info(
            "🔧 笔记创建成功: id=%s type=%s index_total=%s",
            note_id,
            note_type,
            len(self.notes_index["notes"]),
        )

        return f"笔记创建成功\nID: {note_id}\n标题: {title}\n类型: {note_type}"

    def _read_note(self, params: dict[str, Any]) -> str:
        note_id = self._as_non_empty_str(params.get("note_id"), "note_id")
        note_path = self._get_note_path(note_id)

        if not note_path.exists():
            raise ToolException(f"未找到笔记: {note_id}")

        note = self._markdown_to_note(note_path.read_text(encoding="utf-8"))
        self.logger.debug("🔧 读取笔记成功: id=%s type=%s", note_id, note.get("type"))
        return self._format_note(note)

    def _update_note(self, params: dict[str, Any]) -> str:
        note_id = self._as_non_empty_str(params.get("note_id"), "note_id")
        note_path = self._get_note_path(note_id)

        if not note_path.exists():
            raise ToolException(f"未找到笔记: {note_id}")

        if not any(key in params for key in ["title", "content", "note_type", "tags"]):
            raise ToolException(
                "update 至少需要提供一个可更新字段: title/content/note_type/tags"
            )

        note = self._markdown_to_note(note_path.read_text(encoding="utf-8"))

        if "title" in params:
            note["title"] = self._as_non_empty_str(params.get("title"), "title")
        if "content" in params:
            content = self._as_non_empty_str(params.get("content"), "content")
            note["content"] = content
            note["metadata"]["word_count"] = len(content)
        if "note_type" in params:
            note["type"] = self._as_note_type(params.get("note_type"), "note_type")
        if "tags" in params:
            note["tags"] = self._as_tags(params.get("tags"))

        note["updated_at"] = datetime.now().isoformat()

        note_path.write_text(self._note_to_markdown(note), encoding="utf-8")

        for idx_note in self.notes_index["notes"]:
            if idx_note["id"] == note_id:
                idx_note["title"] = note["title"]
                idx_note["type"] = note["type"]
                idx_note["tags"] = note["tags"]
                idx_note["updated_at"] = note["updated_at"]
                break

        self._save_index()
        self.logger.info("🔧 笔记更新成功: id=%s", note_id)
        return f"笔记更新成功: {note_id}"

    def _delete_note(self, params: dict[str, Any]) -> str:
        note_id = self._as_non_empty_str(params.get("note_id"), "note_id")
        note_path = self._get_note_path(note_id)

        if not note_path.exists():
            raise ToolException(f"未找到笔记: {note_id}")

        note_path.unlink()
        self.notes_index["notes"] = [
            note for note in self.notes_index["notes"] if note["id"] != note_id
        ]
        self._save_index()
        self.logger.info(
            "🔧 笔记删除成功: id=%s index_total=%s",
            note_id,
            len(self.notes_index["notes"]),
        )

        return f"笔记已删除: {note_id}"

    def _list_notes(self, params: dict[str, Any]) -> str:
        note_type = None
        if "note_type" in params:
            note_type = self._as_note_type(params.get("note_type"), "note_type")
        limit = self._as_limit(params.get("limit"), default=10)

        notes = self.notes_index["notes"]
        if note_type:
            notes = [note for note in notes if note["type"] == note_type]

        notes = notes[:limit]
        self.logger.debug(
            "🔧 列出笔记: note_type=%s limit=%s returned=%s",
            note_type,
            limit,
            len(notes),
        )
        if not notes:
            return "暂无笔记"

        lines = [f"笔记列表（共 {len(notes)} 条）", ""]
        for note in notes:
            lines.append(f"• [{note['type']}] {note['title']}")
            lines.append(f"  ID: {note['id']}")
            if note.get("tags"):
                lines.append(f"  标签: {', '.join(note['tags'])}")
            lines.append(f"  创建时间: {note['created_at']}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _search_notes(self, params: dict[str, Any]) -> str:
        query = self._as_non_empty_str(params.get("query"), "query").lower()
        limit = self._as_limit(params.get("limit"), default=10)
        note_type = None
        if "note_type" in params:
            note_type = self._as_note_type(params.get("note_type"), "note_type")

        matched_notes: list[dict[str, Any]] = []
        for idx_note in self.notes_index["notes"]:
            if note_type and idx_note["type"] != note_type:
                continue

            note_path = self._get_note_path(idx_note["id"])
            if not note_path.exists():
                continue

            try:
                note = self._markdown_to_note(note_path.read_text(encoding="utf-8"))
            except ToolException:
                self.logger.warning("🔧 解析笔记失败: %s", idx_note["id"])
                continue

            if (
                query in note["title"].lower()
                or query in note["content"].lower()
                or any(query in tag.lower() for tag in note.get("tags", []))
            ):
                matched_notes.append(note)

        matched_notes = matched_notes[:limit]
        self.logger.debug(
            "🔧 搜索笔记: query=%s note_type=%s limit=%s matched=%s",
            query,
            note_type,
            limit,
            len(matched_notes),
        )
        if not matched_notes:
            return f"未找到匹配 '{query}' 的笔记"

        lines = [f"搜索结果（共 {len(matched_notes)} 条）", ""]
        for note in matched_notes:
            lines.append(self._format_note(note, compact=True))
            lines.append("")

        return "\n".join(lines).rstrip()

    def _get_summary(self) -> str:
        total = len(self.notes_index["notes"])
        type_counts: dict[str, int] = {}
        for note in self.notes_index["notes"]:
            note_type = note["type"]
            type_counts[note_type] = type_counts.get(note_type, 0) + 1

        self.logger.debug("🔧 笔记摘要: total=%s type_counts=%s", total, type_counts)

        lines = ["笔记摘要", "", f"总笔记数: {total}", "", "按类型统计:"]
        for note_type in sorted(type_counts):
            lines.append(f"- {note_type}: {type_counts[note_type]}")

        return "\n".join(lines)

    def _load_index(self) -> None:
        if not self.index_file.exists():
            self.notes_index = self._new_index()
            self._save_index()
            self.logger.info("🔧 初始化笔记索引: %s", self.index_file)
            return

        try:
            loaded = json.loads(self.index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.logger.warning(
                "🔧 索引文件损坏或读取失败，重建索引: %s", self.index_file
            )
            self.notes_index = self._new_index()
            self._save_index()
            return

        notes = loaded.get("notes") if isinstance(loaded, dict) else None
        metadata = loaded.get("metadata") if isinstance(loaded, dict) else None
        if not isinstance(notes, list) or not isinstance(metadata, dict):
            self.logger.warning("🔧 索引结构无效，重建索引: %s", self.index_file)
            self.notes_index = self._new_index()
            self._save_index()
            return

        self.notes_index = {"notes": notes, "metadata": metadata}
        self.logger.debug("🔧 加载笔记索引成功: total=%s", len(notes))

    def _save_index(self) -> None:
        now = datetime.now().isoformat()
        metadata = self.notes_index.setdefault("metadata", {})
        metadata.setdefault("created_at", now)
        metadata["updated_at"] = now
        metadata["total_notes"] = len(self.notes_index.get("notes", []))

        self.index_file.write_text(
            json.dumps(self.notes_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.debug("🔧 写入笔记索引: total=%s", metadata["total_notes"])

    def _new_index(self) -> dict[str, Any]:
        now = datetime.now().isoformat()
        return {
            "notes": [],
            "metadata": {
                "created_at": now,
                "updated_at": now,
                "total_notes": 0,
            },
        }

    def _generate_note_id(self) -> str:
        return f"note_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _get_note_path(self, note_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", note_id):
            raise ToolException("note_id 格式非法")
        return self.workspace / f"{note_id}.md"

    def _note_to_markdown(self, note: dict[str, Any]) -> str:
        frontmatter = [
            "---",
            f"id: {note['id']}",
            f"title: {note['title']}",
            f"type: {note['type']}",
            f"tags: {json.dumps(note.get('tags', []), ensure_ascii=False)}",
            f"created_at: {note['created_at']}",
            f"updated_at: {note['updated_at']}",
            "---",
            "",
        ]
        body = f"# {note['title']}\n\n{note['content']}"
        return "\n".join(frontmatter) + body

    def _markdown_to_note(self, markdown_text: str) -> dict[str, Any]:
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n", markdown_text, re.DOTALL
        )
        if not frontmatter_match:
            raise ToolException("笔记文件格式无效")

        metadata_text = frontmatter_match.group(1)
        content_text = markdown_text[frontmatter_match.end() :].strip()

        note: dict[str, Any] = {}
        for raw_line in metadata_text.split("\n"):
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "tags":
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ToolException("笔记文件格式无效") from exc
                if not isinstance(parsed, list):
                    raise ToolException("笔记文件格式无效")
                note[key] = [str(tag) for tag in parsed]
            else:
                note[key] = value

        lines = content_text.split("\n") if content_text else []
        if lines and lines[0].startswith("# "):
            content_text = "\n".join(lines[1:]).strip()

        for required_key in ["id", "title", "type", "created_at", "updated_at"]:
            if required_key not in note:
                raise ToolException("笔记文件格式无效")

        note.setdefault("tags", [])
        note["content"] = content_text
        note["metadata"] = {
            "word_count": len(content_text),
            "status": "active",
        }
        return note

    def _format_note(self, note: dict[str, Any], compact: bool = False) -> str:
        if compact:
            preview = note["content"][:100]
            if len(note["content"]) > 100:
                preview += "..."
            return (
                f"[{note['type']}] {note['title']}\nID: {note['id']}\n内容: {preview}"
            )

        lines = [
            "笔记详情",
            "",
            f"ID: {note['id']}",
            f"标题: {note['title']}",
            f"类型: {note['type']}",
        ]
        if note.get("tags"):
            lines.append(f"标签: {', '.join(note['tags'])}")
        lines.extend(
            [
                f"创建时间: {note['created_at']}",
                f"更新时间: {note['updated_at']}",
                "",
                "内容:",
                note["content"],
            ]
        )
        return "\n".join(lines)

    def _as_non_empty_str(self, value: Any, field_name: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ToolException(f"{field_name} 必须是非空字符串")
        return text

    def _as_note_type(
        self, value: Any, field_name: str, default: str | None = None
    ) -> str:
        if value is None:
            if default is None:
                raise ToolException(f"{field_name} 必须是字符串")
            resolved = default
        else:
            resolved = str(value).strip()
        if resolved not in self.NOTE_TYPES:
            valid_types = ", ".join(sorted(self.NOTE_TYPES))
            raise ToolException(f"{field_name} 必须是以下之一: {valid_types}")
        return resolved

    def _as_tags(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ToolException("tags 必须是字符串列表")

        tags: list[str] = []
        for item in value:
            tag = str(item).strip()
            if not tag:
                raise ToolException("tags 不能为空字符串")
            tags.append(tag)
        return tags

    def _as_limit(self, value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolException("limit 必须是正整数") from exc
        if resolved <= 0:
            raise ToolException("limit 必须是正整数")
        return resolved
