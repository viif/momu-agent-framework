"""工作记忆实现。

- 短期上下文管理
- 容量和时间限制
- 优先级管理
- 自动清理机制
"""

from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from typing import Any

from .base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """工作记忆实现。

    特点：
    - 容量有限（通常10-20条记忆）
    - 时效性强（会话级别）
    - 优先级管理
    - 自动清理过期记忆
    """

    def __init__(self, config: MemoryConfig, storage_backend: Any = None) -> None:
        super().__init__(config, storage_backend)

        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        self.memories: list[MemoryItem] = []
        self.memory_heap: list[tuple[float, datetime, MemoryItem]] = []

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆。"""
        self._expire_old_memories()
        priority = self._calculate_priority(memory_item)

        heapq.heappush(
            self.memory_heap, (-priority, memory_item.timestamp, memory_item)
        )
        self.memories.append(memory_item)

        self.current_tokens += len(memory_item.content.split())
        self._enforce_capacity_limits()

        return memory_item.id

    def retrieve(
        self, query: str, limit: int = 5, user_id: str | None = None, **_: Any
    ) -> list[MemoryItem]:
        """检索工作记忆。"""
        self._expire_old_memories()
        if not self.memories:
            return []

        active_memories = [
            memory
            for memory in self.memories
            if not memory.metadata.get("forgotten", False)
        ]

        filtered_memories = active_memories
        if user_id:
            filtered_memories = [
                memory for memory in active_memories if memory.user_id == user_id
            ]

        if not filtered_memories:
            return []

        vector_scores: dict[str, float] = {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            documents = [memory.content for memory in filtered_memories]
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            doc_vectors = vectorizer.fit_transform(documents)
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            for index, memory in enumerate(filtered_memories):
                vector_scores[memory.id] = float(similarities[index])
        except Exception:
            vector_scores = {}

        query_lower = query.lower()
        scored_memories: list[tuple[float, MemoryItem]] = []

        for memory in filtered_memories:
            content_lower = memory.content.lower()
            vector_score = vector_scores.get(memory.id, 0.0)

            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                union = query_words.union(content_words)
                intersection = query_words.intersection(content_words)
                if union and intersection:
                    keyword_score = len(intersection) / len(union) * 0.8

            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score

            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance *= time_decay

            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))

        scored_memories.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新工作记忆。"""
        for memory in self.memories:
            if memory.id != memory_id:
                continue

            old_tokens = len(memory.content.split())

            if content is not None:
                memory.content = content
                new_tokens = len(content.split())
                self.current_tokens = self.current_tokens - old_tokens + new_tokens

            if importance is not None:
                memory.importance = importance

            if metadata is not None:
                memory.metadata.update(metadata)

            self._update_heap_priority()
            self._enforce_capacity_limits()
            return True

        return False

    def remove(self, memory_id: str) -> bool:
        """删除工作记忆。"""
        for index, memory in enumerate(self.memories):
            if memory.id != memory_id:
                continue

            removed_memory = self.memories.pop(index)
            self.current_tokens -= len(removed_memory.content.split())
            self.current_tokens = max(0, self.current_tokens)
            self._update_heap_priority()
            return True

        return False

    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        return any(memory.id == memory_id for memory in self.memories)

    def clear(self) -> None:
        """清空所有工作记忆。"""
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0

    def get_stats(self) -> dict[str, Any]:
        """获取工作记忆统计信息。"""
        self._expire_old_memories()
        active_memories = self.memories

        return {
            "count": len(active_memories),
            "forgotten_count": 0,
            "total_count": len(self.memories),
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (
                datetime.now() - self.session_start
            ).total_seconds()
            / 60,
            "avg_importance": (
                sum(memory.importance for memory in active_memories)
                / len(active_memories)
                if active_memories
                else 0.0
            ),
            "capacity_usage": (
                len(active_memories) / self.max_capacity
                if self.max_capacity > 0
                else 0.0
            ),
            "token_usage": (
                self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
            ),
            "memory_type": "working",
        }

    def get_recent(self, limit: int = 10) -> list[MemoryItem]:
        """获取最近的记忆。"""
        sorted_memories = sorted(
            self.memories,
            key=lambda memory: memory.timestamp,
            reverse=True,
        )
        return sorted_memories[:limit]

    def get_important(self, limit: int = 10) -> list[MemoryItem]:
        """获取重要记忆。"""
        sorted_memories = sorted(
            self.memories,
            key=lambda memory: memory.importance,
            reverse=True,
        )
        return sorted_memories[:limit]

    def get_all(self) -> list[MemoryItem]:
        """获取所有记忆。"""
        return self.memories.copy()

    def get_context_summary(self, max_length: int = 500) -> str:
        """获取上下文摘要。"""
        if not self.memories:
            return "No working memories available."

        sorted_memories = sorted(
            self.memories,
            key=lambda memory: (memory.importance, memory.timestamp),
            reverse=True,
        )

        summary_parts: list[str] = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
                continue

            remaining = max_length - current_length
            if remaining > 50:
                summary_parts.append(content[:remaining] + "...")
            break

        return "Working Memory Context:\n" + "\n".join(summary_parts)

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 1,
    ) -> int:
        """工作记忆遗忘机制。"""
        forgotten_count = 0
        current_time = datetime.now()
        to_remove: list[str] = []

        cutoff_ttl = current_time - timedelta(minutes=self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)

        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)
        elif strategy == "time_based":
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)
        elif strategy == "capacity_based" and len(self.memories) > self.max_capacity:
            sorted_memories = sorted(
                self.memories,
                key=lambda memory: self._calculate_priority(memory),
            )
            excess_count = len(self.memories) - self.max_capacity
            for memory in sorted_memories[:excess_count]:
                to_remove.append(memory.id)

        for memory_id in dict.fromkeys(to_remove):
            if self.remove(memory_id):
                forgotten_count += 1

        return forgotten_count

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """计算记忆优先级。"""
        priority = memory.importance
        time_decay = self._calculate_time_decay(memory.timestamp)
        priority *= time_decay
        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """计算时间衰减因子。"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = self.config.decay_factor ** (hours_passed / 6)
        return max(0.1, decay_factor)

    def _enforce_capacity_limits(self) -> None:
        """强制执行容量限制。"""
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()

        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _expire_old_memories(self) -> None:
        """按 TTL 清理过期记忆，并同步更新堆与 token 计数。"""
        if not self.memories:
            return

        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept: list[MemoryItem] = []
        removed_token_sum = 0

        for memory in self.memories:
            if memory.timestamp >= cutoff_time:
                kept.append(memory)
            else:
                removed_token_sum += len(memory.content.split())

        if len(kept) == len(self.memories):
            return

        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)
        self.memory_heap = []
        for memory in self.memories:
            priority = self._calculate_priority(memory)
            heapq.heappush(self.memory_heap, (-priority, memory.timestamp, memory))

    def _remove_lowest_priority_memory(self) -> None:
        """删除优先级最低的记忆。"""
        if not self.memories:
            return

        lowest_memory = min(self.memories, key=self._calculate_priority)
        self.remove(lowest_memory.id)

    def _update_heap_priority(self) -> None:
        """更新堆中记忆的优先级。"""
        self.memory_heap = []
        for item in self.memories:
            priority = self._calculate_priority(item)
            heapq.heappush(self.memory_heap, (-priority, item.timestamp, item))
