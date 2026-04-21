"""工作记忆实现

- 短期上下文管理
- 容量和时间限制
- 优先级管理
- 自动清理机制
"""

from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from typing import Any

from ..core.exceptions import MemoryException
from ..utils.logger import get_logger
from .base import Memory, MemoryConfig, MemoryItem


class WorkingMemory(Memory):
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
        # 堆只保存当前记忆的优先级快照；更新后会整体重建，避免维护复杂的 decrease-key。
        self.memory_heap: list[tuple[float, datetime, MemoryItem]] = []

        self.logger = get_logger(__name__)
        self.logger.debug(
            f"🧠 WorkingMemory 初始化完成 (容量: {self.max_capacity}, "
            f"最大 token: {self.max_tokens}, TTL: {self.max_age_minutes} 分钟)"
        )

    async def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆。"""
        try:
            # 写入前先清理过期项，避免过期数据参与容量和 token 计算。
            self._expire_old_memories()
            priority = self._calculate_priority(memory_item)

            heapq.heappush(
                self.memory_heap, (-priority, memory_item.timestamp, memory_item)
            )
            # 列表用于遍历/筛选，堆用于优先级排序；两者需保持同步。
            self.memories.append(memory_item)

            # 这里用词数近似 token，和容量控制保持一致。
            self.current_tokens += len(memory_item.content.split())
            await self._enforce_capacity_limits()

            self.logger.debug(
                f"🧠 添加工作记忆 [{memory_item.id}]，当前数量: {len(self.memories)}，"
                f"token: {self.current_tokens}/{self.max_tokens}"
            )
            return memory_item.id
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 添加工作记忆失败: {e}")
            raise MemoryException(f"添加工作记忆失败: {e}") from e

    async def retrieve(
        self, query: str, limit: int = 5, user_id: str | None = None, **_: Any
    ) -> list[MemoryItem]:
        """检索工作记忆。"""
        try:
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
                # 优先使用向量相似度；依赖不可用时回退到关键词匹配。
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
                    # 语义匹配可用时，以语义为主、关键词为辅做融合。
                    base_relevance = vector_score * 0.7 + keyword_score * 0.3
                else:
                    # 语义匹配不可用时，仅使用关键词得分。
                    base_relevance = keyword_score

                # 时间越久衰减越强，避免旧记忆长期占据前列。
                time_decay = self._calculate_time_decay(memory.timestamp)
                base_relevance *= time_decay

                # 在相关性基础上再乘以重要性权重。
                importance_weight = 0.8 + (memory.importance * 0.4)
                final_score = base_relevance * importance_weight

                if final_score > 0:
                    scored_memories.append((final_score, memory))

            scored_memories.sort(key=lambda item: item[0], reverse=True)
            results = [memory for _, memory in scored_memories[:limit]]
            self.logger.debug(
                f"🧠 检索工作记忆，命中 {len(results)} 条（查询: {query!r}）"
            )
            return results
        except MemoryException:
            raise
        except Exception as e:
            self.logger.error(f"🧠 检索工作记忆失败: {e}")
            raise MemoryException(f"检索工作记忆失败: {e}") from e

    async def update(
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

            # 内容/重要性变化都会影响排序，需刷新堆并重新校验容量限制。
            self._update_heap_priority()
            await self._enforce_capacity_limits()
            self.logger.debug(f"🧠 更新工作记忆 [{memory_id}] 成功")
            return True

        self.logger.warning(f"🧠 更新工作记忆失败，未找到 [{memory_id}]")
        return False

    async def remove(self, memory_id: str) -> bool:
        """删除工作记忆。"""
        for index, memory in enumerate(self.memories):
            if memory.id != memory_id:
                continue

            removed_memory = self.memories.pop(index)
            self.current_tokens -= len(removed_memory.content.split())
            self.current_tokens = max(0, self.current_tokens)
            # 删除后重建堆，避免残留条目影响后续优先级淘汰。
            self._update_heap_priority()
            self.logger.debug(
                f"🧠 删除工作记忆 [{memory_id}]，剩余: {len(self.memories)} 条"
            )
            return True

        return False

    async def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        return any(memory.id == memory_id for memory in self.memories)

    async def clear(self) -> None:
        """清空所有工作记忆。"""
        count = len(self.memories)
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0
        self.logger.info(f"🧠 清空工作记忆，共移除 {count} 条")

    async def get_stats(self) -> dict[str, Any]:
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

    async def get_recent(self, limit: int = 10) -> list[MemoryItem]:
        """获取最近的记忆。"""
        sorted_memories = sorted(
            self.memories,
            key=lambda memory: memory.timestamp,
            reverse=True,
        )
        return sorted_memories[:limit]

    async def get_important(self, limit: int = 10) -> list[MemoryItem]:
        """获取重要记忆。"""
        sorted_memories = sorted(
            self.memories,
            key=lambda memory: memory.importance,
            reverse=True,
        )
        return sorted_memories[:limit]

    async def get_all(self) -> list[MemoryItem]:
        """获取所有记忆。"""
        return self.memories.copy()

    async def get_context_summary(self, max_length: int = 500) -> str:
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

    async def forget(
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
        # 先统一收集 TTL 过期项，确保任意策略下都不会保留超时数据。
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)

        if strategy == "importance_based":
            # 删除低于重要性阈值的记忆。
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)
        elif strategy == "time_based":
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            # 删除早于给定时间窗口的记忆。
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)
        elif strategy == "capacity_based" and len(self.memories) > self.max_capacity:
            sorted_memories = sorted(
                self.memories,
                key=lambda memory: self._calculate_priority(memory),
            )
            excess_count = len(self.memories) - self.max_capacity
            # 仅删除超出容量的最低优先级部分。
            for memory in sorted_memories[:excess_count]:
                to_remove.append(memory.id)

        # 去重后再删除，避免同一记忆被多种策略重复计数。
        for memory_id in dict.fromkeys(to_remove):
            if await self.remove(memory_id):
                forgotten_count += 1

        if forgotten_count:
            self.logger.info(
                f"🧠 遗忘机制（{strategy}）移除了 {forgotten_count} 条记忆"
            )
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

    async def _enforce_capacity_limits(self) -> None:
        """强制执行容量限制。"""
        # 先按条目数裁剪，再按 token 裁剪，最终满足双重约束。
        while len(self.memories) > self.max_capacity:
            self.logger.warning(
                f"🧠 工作记忆超出容量限制 ({len(self.memories)}/{self.max_capacity})，移除最低优先级记忆"
            )
            await self._remove_lowest_priority_memory()

        while self.current_tokens > self.max_tokens:
            self.logger.warning(
                f"🧠 工作记忆超出 token 限制 ({self.current_tokens}/{self.max_tokens})，移除最低优先级记忆"
            )
            await self._remove_lowest_priority_memory()

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

        expired_count = len(self.memories) - len(kept)
        self.logger.debug(
            f"🧠 TTL 清理过期记忆 {expired_count} 条（TTL: {self.max_age_minutes} 分钟）"
        )
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)
        # 清理后直接重建堆，保证后续容量淘汰仍按最新优先级判断。
        self.memory_heap = []
        for memory in self.memories:
            priority = self._calculate_priority(memory)
            heapq.heappush(self.memory_heap, (-priority, memory.timestamp, memory))

    async def _remove_lowest_priority_memory(self) -> None:
        """删除优先级最低的记忆。"""
        if not self.memories:
            return

        # 实时重算最小优先级，避免依赖可能滞后的堆快照。
        lowest_memory = min(self.memories, key=self._calculate_priority)
        await self.remove(lowest_memory.id)

    def _update_heap_priority(self) -> None:
        """更新堆中记忆的优先级。"""
        # 堆不做增量调整，统一重建以保证实现简单且状态一致。
        self.memory_heap = []
        for item in self.memories:
            priority = self._calculate_priority(item)
            heapq.heappush(self.memory_heap, (-priority, item.timestamp, item))
