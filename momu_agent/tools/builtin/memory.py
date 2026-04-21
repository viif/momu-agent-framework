"""记忆工具"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ...core.exceptions import ToolException
from ...memory import MemoryConfig, MemoryManager
from ...utils.logger import get_logger
from ..base import Tool, ToolParameter


class MemoryTool(Tool):
    """基于内置记忆系统的统一工具。"""

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig | None = None,
        memory_types: list[str] | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        """初始化记忆工具并配置底层记忆管理器。"""
        super().__init__(
            name="memory",
            description="基于内置记忆系统的记忆工具，支持添加、检索、更新、删除和统计。",
        )
        self.logger = get_logger(__name__)
        enabled_types = memory_types or ["working", "episodic", "semantic"]
        self.memory_manager = memory_manager or MemoryManager(
            config=memory_config or MemoryConfig(),
            user_id=user_id,
            enable_working="working" in enabled_types,
            enable_episodic="episodic" in enabled_types,
            enable_semantic="semantic" in enabled_types,
        )

    async def run(self, parameters: dict[str, Any]) -> str:
        """根据 action 分发并执行对应记忆操作。"""
        action = str(parameters.get("action", "")).strip()
        if not action:
            raise ToolException("必须提供 action 参数")

        # 显式分支保证 action 与处理方法一一对应，便于排查与扩展。
        if action == "add":
            return await self._add(parameters)
        if action == "search":
            return await self._search(parameters)
        if action == "update":
            return await self._update(parameters)
        if action == "remove":
            return await self._remove(parameters)
        if action == "stats":
            return await self._stats(parameters)
        if action == "forget":
            return await self._forget(parameters)
        if action == "clear":
            return await self._clear(parameters)

        raise ToolException(
            f"不支持的 action: {action}。可用 action: add, search, update, remove, stats, forget, clear"
        )

    def get_parameters(self) -> list[ToolParameter]:
        """定义并返回记忆工具可用参数列表。"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add、search、update、remove、stats、forget、clear",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="记忆内容",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索查询",
                required=False,
            ),
            ToolParameter(
                name="memory_id",
                type="string",
                description="记忆 ID",
                required=False,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="单个记忆类型",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="memory_types",
                type="array",
                description="多个记忆类型",
                required=False,
            ),
            ToolParameter(
                name="importance",
                type="float",
                description="记忆重要性",
                required=False,
            ),
            ToolParameter(
                name="metadata",
                type="object",
                description="记忆元数据",
                required=False,
            ),
            ToolParameter(
                name="auto_classify",
                type="boolean",
                description="是否自动分类记忆类型",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="检索结果数量限制",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="user_id",
                type="string",
                description="检索时使用的用户 ID",
                required=False,
            ),
            ToolParameter(
                name="session_id",
                type="string",
                description="情景记忆 session_id 过滤条件",
                required=False,
            ),
            ToolParameter(
                name="importance_threshold",
                type="float",
                description="最小重要性阈值",
                required=False,
                default=0.0,
            ),
            ToolParameter(
                name="score_threshold",
                type="float",
                description="最小相关性分数阈值",
                required=False,
            ),
            ToolParameter(
                name="start_time",
                type="string",
                description="起始时间 ISO 字符串",
                required=False,
            ),
            ToolParameter(
                name="end_time",
                type="string",
                description="结束时间 ISO 字符串",
                required=False,
            ),
            ToolParameter(
                name="strategy",
                type="string",
                description="遗忘策略",
                required=False,
                default="importance_based",
            ),
            ToolParameter(
                name="threshold",
                type="float",
                description="遗忘阈值",
                required=False,
                default=0.1,
            ),
            ToolParameter(
                name="max_age_days",
                type="integer",
                description="最大保留天数",
                required=False,
                default=30,
            ),
        ]

    async def _add(self, parameters: dict[str, Any]) -> str:
        """添加一条记忆并返回写入结果。"""
        content = str(parameters.get("content", "")).strip()
        if not content:
            raise ToolException("add 需要提供非空 content")

        memory_type = str(parameters.get("memory_type", "working")).strip() or "working"
        importance = self._as_float(parameters.get("importance"), "importance")
        memory_id = await self.memory_manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=self._as_metadata(parameters.get("metadata")),
            auto_classify=bool(parameters.get("auto_classify", True)),
        )
        resolved_type = (
            memory_type if not parameters.get("auto_classify", True) else "自动分类"
        )
        return f"已添加记忆: {memory_id}\n记忆类型: {resolved_type}"

    async def _search(self, parameters: dict[str, Any]) -> str:
        """按条件检索记忆并格式化返回结果。"""
        query = str(parameters.get("query", "")).strip()
        if not query:
            raise ToolException("search 需要提供非空 query")

        memory_types = parameters.get("memory_types")
        single_type = parameters.get("memory_type")
        # 兼容仅传 memory_type 的调用方式，内部统一为列表处理。
        if memory_types is None and single_type:
            memory_types = [str(single_type)]
        if memory_types is not None and not isinstance(memory_types, list):
            raise ToolException("memory_types 必须是列表")

        start_time = self._parse_datetime(parameters.get("start_time"))
        end_time = self._parse_datetime(parameters.get("end_time"))
        # 时间与阈值过滤统一下沉到 MemoryManager，工具层只做参数解析与校验。
        results = await self.memory_manager.retrieve_memories(
            query=query,
            memory_types=memory_types,
            limit=self._get_positive_int(parameters.get("limit"), 5, "limit"),
            importance_threshold=float(parameters.get("importance_threshold", 0.0)),
            score_threshold=self._as_float(
                parameters.get("score_threshold"), "score_threshold"
            ),
            user_id=self._as_optional_str(parameters.get("user_id")),
            session_id=self._as_optional_str(parameters.get("session_id")),
            start_time=start_time,
            end_time=end_time,
        )
        if not results:
            return f"未检索到与 '{query}' 相关的记忆"

        lines = [f"检索到 {len(results)} 条记忆："]
        # 输出中保留 score 与 id，便于后续人工筛选和定向 update/remove。
        for index, memory in enumerate(results, start=1):
            score = float(memory.metadata.get("relevance_score", memory.importance))
            preview = (
                memory.content[:80] + "..."
                if len(memory.content) > 80
                else memory.content
            )
            lines.append(
                f"{index}. [{memory.memory_type}] {preview} (score={score:.3f}, id={memory.id})"
            )
        return "\n".join(lines)

    async def _update(self, parameters: dict[str, Any]) -> str:
        """更新指定记忆的内容、重要性或元数据。"""
        memory_id = str(parameters.get("memory_id", "")).strip()
        if not memory_id:
            raise ToolException("update 需要提供 memory_id")

        success = await self.memory_manager.update_memory(
            memory_id=memory_id,
            content=self._as_optional_str(parameters.get("content")),
            importance=self._as_float(parameters.get("importance"), "importance"),
            metadata=self._as_metadata(parameters.get("metadata")),
        )
        if not success:
            raise ToolException(f"未找到要更新的记忆: {memory_id}")
        return f"已更新记忆: {memory_id}"

    async def _remove(self, parameters: dict[str, Any]) -> str:
        """删除指定 ID 的记忆。"""
        memory_id = str(parameters.get("memory_id", "")).strip()
        if not memory_id:
            raise ToolException("remove 需要提供 memory_id")

        success = await self.memory_manager.remove_memory(memory_id)
        if not success:
            raise ToolException(f"未找到要删除的记忆: {memory_id}")
        return f"已删除记忆: {memory_id}"

    async def _stats(self, parameters: dict[str, Any]) -> str:
        """返回记忆系统的聚合统计信息。"""
        stats = await self.memory_manager.get_memory_stats()
        lines = [
            "记忆系统统计",
            f"用户: {stats['user_id']}",
            f"启用类型: {', '.join(stats['enabled_types'])}",
            f"总记忆数: {stats['total_memories']}",
        ]
        for memory_type, detail in stats["memories_by_type"].items():
            count = detail.get(
                "count", detail.get("total_count", detail.get("memories_count", 0))
            )
            lines.append(f"- {memory_type}: {count}")
        return "\n".join(lines)

    async def _forget(self, parameters: dict[str, Any]) -> str:
        """按策略执行遗忘并返回处理数量。"""
        count = await self.memory_manager.forget_memories(
            strategy=str(parameters.get("strategy", "importance_based")),
            threshold=float(parameters.get("threshold", 0.1)),
            max_age_days=self._get_positive_int(
                parameters.get("max_age_days"), 30, "max_age_days"
            ),
        )
        return f"已遗忘 {count} 条记忆"

    async def _clear(self, parameters: dict[str, Any]) -> str:
        """清空当前记忆管理器中的所有记忆。"""
        await self.memory_manager.clear_all_memories()
        return "已清空所有记忆"

    def _get_positive_int(self, value: Any, default: int, field_name: str) -> int:
        """解析正整数参数，不合法时抛出工具异常。"""
        if value is None:
            return default
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolException(f"{field_name} 必须是正整数") from exc
        if resolved <= 0:
            raise ToolException(f"{field_name} 必须是正整数")
        return resolved

    def _as_float(self, value: Any, field_name: str) -> float | None:
        """解析浮点参数，空值返回 None。"""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ToolException(f"{field_name} 必须是数字") from exc

    def _as_metadata(self, value: Any) -> dict[str, Any] | None:
        """校验并复制 metadata 参数。"""
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ToolException("metadata 必须是对象")
        return dict(value)

    def _as_optional_str(self, value: Any) -> str | None:
        """将输入转换为去空白字符串，空串转为 None。"""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _parse_datetime(self, value: Any) -> datetime | None:
        """解析 ISO 时间字符串为 datetime。"""
        text = self._as_optional_str(value)
        if text is None:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError as exc:
            raise ToolException("时间参数必须是合法的 ISO 时间字符串") from exc


async def memory_add(content: str, **kwargs: Any) -> str:
    """便捷函数：添加记忆。"""
    # 便捷函数每次创建独立工具实例，适合脚本侧直接调用。
    tool = MemoryTool()
    return await tool.run({"action": "add", "content": content, **kwargs})


async def memory_search(query: str, **kwargs: Any) -> str:
    """便捷函数：检索记忆。"""
    tool = MemoryTool()
    return await tool.run({"action": "search", "query": query, **kwargs})


async def memory_get_stats(**kwargs: Any) -> str:
    """便捷函数：获取记忆统计。"""
    tool = MemoryTool()
    return await tool.run({"action": "stats", **kwargs})
