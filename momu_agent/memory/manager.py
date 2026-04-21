"""记忆管理器 - 记忆核心层的统一管理接口"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from ..utils.logger import get_logger
from .base import MemoryConfig, MemoryItem
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working import WorkingMemory

ManagedMemory = WorkingMemory | EpisodicMemory | SemanticMemory


class MemoryManager:
    """统一协调多种记忆类型的异步管理器。"""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        working_memory: WorkingMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        semantic_memory: SemanticMemory | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.logger = get_logger(__name__)
        self.memory_types: dict[str, ManagedMemory] = {}

        if working_memory is not None or enable_working:
            self.memory_types["working"] = working_memory or WorkingMemory(self.config)

        if episodic_memory is not None or enable_episodic:
            self.memory_types["episodic"] = episodic_memory or EpisodicMemory(
                self.config
            )

        if semantic_memory is not None or enable_semantic:
            self.memory_types["semantic"] = semantic_memory or SemanticMemory(
                self.config
            )

        self.logger.info(
            f"MemoryManager 初始化完成，启用记忆类型: {list(self.memory_types.keys())}"
        )

    async def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        auto_classify: bool = True,
    ) -> str:
        metadata = dict(metadata or {})
        if auto_classify:
            memory_type = self._classify_memory_type(content, metadata)

        if memory_type not in self.memory_types:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

        resolved_importance = importance
        if resolved_importance is None:
            resolved_importance = self._calculate_importance(content, metadata)

        memory_item = MemoryItem(
            id=str(uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=resolved_importance,
            metadata=metadata,
        )
        return await self.memory_types[memory_type].add(memory_item)

    async def retrieve_memories(
        self,
        query: str,
        memory_types: list[str] | None = None,
        limit: int = 10,
        importance_threshold: float = 0.0,
        score_threshold: float | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        time_range: tuple[datetime | None, datetime | None] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MemoryItem]:
        selected_types = memory_types or list(self.memory_types.keys())
        selected_types = [
            memory_type
            for memory_type in selected_types
            if memory_type in self.memory_types
        ]
        if not selected_types or limit <= 0:
            return []

        if time_range is not None:
            range_start, range_end = time_range
            start_time = start_time or range_start
            end_time = end_time or range_end

        all_results: list[MemoryItem] = []
        resolved_user_id = user_id or self.user_id
        for memory_type in selected_types:
            type_results = await self.memory_types[memory_type].retrieve(
                query=query,
                limit=limit,
                user_id=resolved_user_id,
                session_id=session_id,
                importance_threshold=importance_threshold,
                score_threshold=score_threshold,
                time_range=time_range,
                start_time=start_time,
                end_time=end_time,
            )
            all_results.extend(type_results)

        unique_results: dict[str, MemoryItem] = {}
        for memory in all_results:
            if memory.id not in unique_results:
                unique_results[memory.id] = memory

        sorted_results = sorted(
            unique_results.values(),
            key=lambda memory: (
                float(memory.metadata.get("relevance_score", memory.importance)),
                memory.importance,
                memory.timestamp,
            ),
            reverse=True,
        )
        return sorted_results[:limit]

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        for memory in self.memory_types.values():
            if await memory.has_memory(memory_id):
                return await memory.update(memory_id, content, importance, metadata)
        return False

    async def remove_memory(self, memory_id: str) -> bool:
        for memory in self.memory_types.values():
            if await memory.has_memory(memory_id):
                return await memory.remove(memory_id)
        return False

    async def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        total_forgotten = 0
        for memory in self.memory_types.values():
            if hasattr(memory, "forget"):
                total_forgotten += await memory.forget(
                    strategy=strategy,
                    threshold=threshold,
                    max_age_days=max_age_days,
                )
        return total_forgotten

    async def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        if from_type != "working":
            raise ValueError("当前只支持从 working 记忆整合")
        if from_type not in self.memory_types or to_type not in self.memory_types:
            return 0

        source_memory = self.memory_types[from_type]
        target_memory = self.memory_types[to_type]
        if not isinstance(source_memory, WorkingMemory):
            return 0

        candidates = [
            memory
            for memory in await source_memory.get_all()
            if memory.importance >= importance_threshold
        ]

        consolidated = 0
        for memory in candidates:
            if await source_memory.remove(memory.id):
                memory.memory_type = to_type
                memory.importance = min(1.0, memory.importance * 1.1)
                await target_memory.add(memory)
                consolidated += 1

        return consolidated

    async def get_memory_stats(self) -> dict[str, Any]:
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor,
            },
        }

        for memory_type, memory in self.memory_types.items():
            type_stats = await memory.get_stats()
            stats["memories_by_type"][memory_type] = type_stats
            stats["total_memories"] += int(
                type_stats.get(
                    "count",
                    type_stats.get("total_count", type_stats.get("memories_count", 0)),
                )
            )

        return stats

    async def clear_all_memories(self) -> None:
        for memory in self.memory_types.values():
            await memory.clear()

    def _classify_memory_type(
        self,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        if metadata:
            explicit_type = metadata.get("memory_type") or metadata.get("type")
            if explicit_type in self.memory_types:
                return explicit_type
            if metadata.get("concepts") or metadata.get("entities"):
                return "semantic"
            if metadata.get("session_id"):
                return "episodic"

        if self._is_episodic_content(content):
            return "episodic"
        if self._is_semantic_content(content):
            return "semantic"
        return "working"

    def _is_episodic_content(self, content: str) -> bool:
        episodic_keywords = ["昨天", "今天", "明天", "上次", "发生", "经历"]
        return any(keyword in content for keyword in episodic_keywords)

    def _is_semantic_content(self, content: str) -> bool:
        semantic_keywords = ["定义", "概念", "规则", "知识", "原理", "方法"]
        return any(keyword in content for keyword in semantic_keywords)

    def _calculate_importance(
        self,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> float:
        importance = 0.5
        if len(content) > 100:
            importance += 0.1
        if any(
            keyword in content
            for keyword in ["重要", "关键", "必须", "注意", "警告", "错误"]
        ):
            importance += 0.2

        if metadata:
            priority = metadata.get("priority")
            if priority == "high":
                importance += 0.3
            elif priority == "low":
                importance -= 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        return f"MemoryManager(user={self.user_id}, types={list(self.memory_types.keys())})"
