"""ContextBuilder - GSSC流水线实现

实现 Gather-Select-Structure-Compress 上下文构建流程：
1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
2. Select: 基于优先级、相关性、多样性筛选
3. Structure: 组织成结构化上下文模板
4. Compress: 在预算内压缩与规范化
"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import tiktoken

from ..core.llm import LLM
from ..core.message import Message
from ..tools import MemoryTool, RAGTool
from ..utils.logger import get_logger

# 设定缓存目录
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.getcwd(), ".model_cache")


@dataclass
class ContextPacket:
    """上下文信息包"""

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0

    def __post_init__(self):
        """自动计算token数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文构建配置"""

    max_tokens: int = 8000  # 总预算
    reserve_ratio: float = 0.15  # 生成余量（10-20%）
    min_relevance: float = 0.3  # 最小相关性阈值
    enable_mmr: bool = True  # 启用最大边际相关性（多样性）
    mmr_lambda: float = 0.7  # MMR平衡参数（0=纯多样性, 1=纯相关性）
    system_prompt_template: str = ""  # 系统提示模板
    enable_compression: bool = True  # 启用压缩

    def get_available_tokens(self) -> int:
        """获取可用token预算（扣除余量）"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """上下文构建器 - GSSC流水线

    用法示例：
    ```python
    builder = ContextBuilder(
        llm=llm,
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )

    context = await builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """

    def __init__(
        self,
        llm: LLM,
        memory_tool: MemoryTool | None = None,
        rag_tool: RAGTool | None = None,
        config: ContextConfig | None = None,
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._llm = llm
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self.logger = get_logger(__name__)

    async def build(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        additional_packets: list[ContextPacket] | None = None,
    ) -> str:
        """构建完整上下文"""
        history = conversation_history or []
        extras = additional_packets or []
        self.logger.debug(
            "🧩 ContextBuilder开始构建: query_len=%s history_count=%s has_system=%s extra_count=%s",
            len(user_query),
            len(history),
            bool(system_instructions),
            len(extras),
        )

        packets = await self._gather(
            user_query=user_query,
            conversation_history=history,
            system_instructions=system_instructions,
            additional_packets=extras,
        )
        self.logger.debug("🧩 Gather完成: packets=%s", len(packets))

        selected_packets = self._select(packets, user_query)
        self.logger.debug("🧩 Select完成: selected_packets=%s", len(selected_packets))

        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions,
        )
        self.logger.debug(
            "🧩 Structure完成: structured_tokens=%s", count_tokens(structured_context)
        )

        compressed = await self._compress(structured_context)
        self.logger.info(
            "🧩 ContextBuilder构建完成: final_tokens=%s", count_tokens(compressed)
        )
        return compressed

    async def _gather(
        self,
        user_query: str,
        conversation_history: list[Message],
        system_instructions: str | None,
        additional_packets: list[ContextPacket],
    ) -> list[ContextPacket]:
        """Gather: 收集候选信息"""
        packets: list[ContextPacket] = []
        self.logger.debug(
            "🧩 Gather开始: history_count=%s has_system=%s extra_count=%s",
            len(conversation_history),
            bool(system_instructions),
            len(additional_packets),
        )

        # P0: 系统指令（强约束）
        if system_instructions:
            packet = ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"},
            )
            packets.append(packet)
            self.logger.debug("🧩 Gather加入系统指令: tokens=%s", packet.token_count)

        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                self.logger.debug("🧩 Gather开始检索MemoryTool: task_state")
                state_results = await self.memory_tool.run(
                    {
                        "action": "search",
                        "query": "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                        "importance_threshold": 0.7,
                        "limit": 5,
                    }
                )
                if self._has_retrieval_content(state_results):
                    packet = ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"},
                    )
                    packets.append(packet)
                    self.logger.debug(
                        "🧩 Gather加入task_state: tokens=%s", packet.token_count
                    )
                else:
                    self.logger.debug("🧩 MemoryTool task_state检索无有效内容")

                self.logger.debug("🧩 Gather开始检索MemoryTool: related_memory")
                related_results = await self.memory_tool.run(
                    {
                        "action": "search",
                        "query": user_query,
                        "limit": 5,
                    }
                )
                if self._has_retrieval_content(related_results):
                    packet = ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"},
                    )
                    packets.append(packet)
                    self.logger.debug(
                        "🧩 Gather加入related_memory: tokens=%s", packet.token_count
                    )
                else:
                    self.logger.debug("🧩 MemoryTool related_memory检索无有效内容")
            except Exception as e:
                self.logger.warning("🧩 记忆检索失败: %s", e)

        # P2: 从RAG中获取事实证据
        if self.rag_tool:
            try:
                self.logger.debug("🧩 Gather开始检索RAGTool")
                rag_results = await self.rag_tool.run(
                    {
                        "action": "search",
                        "query": user_query,
                        "limit": 5,
                    }
                )
                if self._has_retrieval_content(rag_results):
                    packet = ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"},
                    )
                    packets.append(packet)
                    self.logger.debug(
                        "🧩 Gather加入knowledge_base: tokens=%s", packet.token_count
                    )
                else:
                    self.logger.debug("🧩 RAGTool检索无有效内容")
            except Exception as e:
                self.logger.warning("🧩 RAG检索失败: %s", e)

        # P3: 对话历史（辅助材料）
        if conversation_history:
            # 仅保留用户/助手消息，避免 tool 结果膨胀上下文
            user_assistant_history = [
                msg for msg in conversation_history if msg.role in {"user", "assistant"}
            ]
            recent_history = user_assistant_history[-6:]
            history_text = "\n".join(
                [self._format_history_message(msg) for msg in recent_history]
            )
            packet = ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)},
            )
            packets.append(packet)
            self.logger.debug(
                "🧩 Gather加入history: count=%s tokens=%s",
                len(recent_history),
                packet.token_count,
            )

        # 添加额外包
        if additional_packets:
            packets.extend(additional_packets)
            self.logger.debug("🧩 Gather加入额外包: count=%s", len(additional_packets))

        self.logger.info(
            "🧩 Gather完成: packets=%s total_tokens=%s",
            len(packets),
            sum(packet.token_count for packet in packets),
        )
        return packets

    @staticmethod
    def _extract_user_query_from_history(content: str) -> str | None:
        marker = "用户问题："
        if marker not in content:
            return None

        extracted = content.split(marker, maxsplit=1)[1]
        next_section_index = extracted.find("\n[")
        if next_section_index != -1:
            extracted = extracted[:next_section_index]

        extracted = extracted.strip()
        return extracted or None

    def _format_history_message(self, msg: Message) -> str:
        content = msg.content
        if msg.role == "user":
            content = self._extract_user_query_from_history(msg.content) or msg.content
        return f"[{msg.role}] {content[:500]}"

    def _select(
        self, packets: list[ContextPacket], user_query: str
    ) -> list[ContextPacket]:
        """Select: 基于分数与预算的筛选"""
        self.logger.debug("🧩 Select开始: packet_count=%s", len(packets))

        # 1) 计算相关性（关键词重叠）
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if query_tokens:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0

        # 2) 计算新近性（指数衰减）
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600
            return math.exp(-delta / tau)

        # 3) 计算复合分：0.7*相关性 + 0.3*新近性
        scored_packets: list[tuple[float, ContextPacket]] = []
        for packet in packets:
            rec = recency_score(packet.timestamp)
            score = 0.7 * packet.relevance_score + 0.3 * rec
            scored_packets.append((score, packet))

        # 4) 系统指令单独拿出，固定纳入
        system_packets = [
            packet
            for _, packet in scored_packets
            if packet.metadata.get("type") == "instructions"
        ]
        remaining = [
            packet
            for _, packet in sorted(
                scored_packets, key=lambda item: item[0], reverse=True
            )
            if packet.metadata.get("type") != "instructions"
        ]

        # 5) 依据 min_relevance 过滤（对非系统包），history 保底保留
        filtered = [
            packet
            for packet in remaining
            if packet.metadata.get("type") == "history"
            or packet.relevance_score >= self.config.min_relevance
        ]

        self.logger.debug(
            "🧩 Select打分结果: system_packets=%s remaining=%s filtered=%s min_relevance=%.2f",
            len(system_packets),
            len(remaining),
            len(filtered),
            self.config.min_relevance,
        )

        # 6) 按预算填充
        available_tokens = self.config.get_available_tokens()
        selected: list[ContextPacket] = []
        used_tokens = 0

        # 先放入系统指令（不排序）
        for packet in system_packets:
            if used_tokens + packet.token_count <= available_tokens:
                selected.append(packet)
                used_tokens += packet.token_count
            else:
                self.logger.debug(
                    "🧩 Select跳过系统包(预算不足): type=%s packet_tokens=%s used=%s budget=%s",
                    packet.metadata.get("type"),
                    packet.token_count,
                    used_tokens,
                    available_tokens,
                )

        # 再按分数加入其余包
        for packet in filtered:
            if used_tokens + packet.token_count > available_tokens:
                self.logger.debug(
                    "🧩 Select跳过候选包(预算不足): type=%s packet_tokens=%s used=%s budget=%s relevance=%.3f",
                    packet.metadata.get("type"),
                    packet.token_count,
                    used_tokens,
                    available_tokens,
                    packet.relevance_score,
                )
                continue
            selected.append(packet)
            used_tokens += packet.token_count

        self.logger.info(
            "🧩 Select完成: selected=%s used_tokens=%s budget=%s",
            len(selected),
            used_tokens,
            available_tokens,
        )
        return selected

    def _structure(
        self,
        selected_packets: list[ContextPacket],
        user_query: str,
        system_instructions: str | None,
    ) -> str:
        """Structure: 组织成结构化上下文模板"""
        sections: list[str] = []

        # [Role & Policies] - 系统指令
        p0_packets = [
            packet
            for packet in selected_packets
            if packet.metadata.get("type") == "instructions"
        ]
        if p0_packets:
            sections.append(
                "[Role & Policies]\n" + "\n".join(p.content for p in p0_packets)
            )

        # [Task] - 当前任务
        sections.append(f"[Task]\n用户问题：{user_query}")

        # [State] - 任务状态
        p1_packets = [
            packet
            for packet in selected_packets
            if packet.metadata.get("type") == "task_state"
        ]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题：\n" + "\n".join(
                packet.content for packet in p1_packets
            )
            sections.append(state_section)

        # [Evidence] - 事实证据
        p2_packets = [
            packet
            for packet in selected_packets
            if packet.metadata.get("type")
            in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_lines = ["[Evidence]", "事实与引用："]
            for packet in p2_packets:
                evidence_lines.append("")
                evidence_lines.append(packet.content)
            sections.append("\n".join(evidence_lines))

        # [Context] - 辅助材料（历史等）
        p3_packets = [
            packet
            for packet in selected_packets
            if packet.metadata.get("type") == "history"
        ]
        if p3_packets:
            context_section = "[Context]\n对话历史与背景：\n" + "\n".join(
                packet.content for packet in p3_packets
            )
            sections.append(context_section)

        # [Output] - 输出约束
        sections.append(
            "[Output]\n"
            "请按以下格式回答：\n"
            "1. 结论（简洁明确）\n"
            "2. 依据（列出支撑证据及来源）\n"
            "3. 风险与假设（如有）\n"
            "4. 下一步行动建议（如适用）"
        )

        structured = "\n\n".join(sections)
        self.logger.debug(
            "🧩 Structure分段统计: role=%s state=%s evidence=%s context=%s total_sections=%s",
            len(p0_packets),
            len(p1_packets),
            len(p2_packets),
            len(p3_packets),
            len(sections),
        )
        return structured

    async def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            self.logger.debug("🧩 Compress跳过: enable_compression=False")
            return context

        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        self.logger.debug(
            "🧩 Compress检查预算: current_tokens=%s available_tokens=%s",
            current_tokens,
            available_tokens,
        )

        if current_tokens <= available_tokens:
            self.logger.debug("🧩 Compress无需压缩: 已在预算内")
            return context

        self.logger.warning(
            "🧩 上下文超预算 (%s > %s)，尝试LLM高保真摘要",
            current_tokens,
            available_tokens,
        )

        try:
            compressed = await self._llm.invoke(
                self._build_compression_messages(context, available_tokens)
            )
            if compressed and compressed.strip():
                compressed = compressed.strip()
                compressed_tokens = count_tokens(compressed)
                self.logger.debug("🧩 LLM摘要完成: tokens=%s", compressed_tokens)
                if compressed_tokens <= available_tokens:
                    self.logger.info("🧩 Compress使用LLM摘要成功")
                    return compressed
                self.logger.warning("🧩 LLM摘要仍超预算，回退截断策略")
            else:
                self.logger.warning("🧩 LLM摘要为空，回退截断策略")
        except Exception as e:
            self.logger.warning("🧩 LLM摘要失败，回退截断策略: %s", e)

        truncated = self._truncate_to_budget(context, available_tokens)
        self.logger.info(
            "🧩 Compress使用截断策略完成: tokens=%s", count_tokens(truncated)
        )
        return truncated

    def _build_compression_messages(
        self, context: str, available_tokens: int
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是上下文压缩助手。请在不改变关键事实、约束、任务目标、"
                    "证据来源和输出要求的前提下，对输入上下文做高保真摘要。"
                    "严格保留核心结构标题（如 [Task]、[State]、[Evidence] 等）"
                    "并删除冗余内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请将以下上下文压缩到不超过 {available_tokens} token。\n"
                    "要求：\n"
                    "1. 保留关键约束与事实\n"
                    "2. 保留关键信息来源\n"
                    "3. 维持结构化段落\n"
                    "4. 仅输出压缩后的上下文，不要附加解释\n\n"
                    f"原始上下文：\n{context}"
                ),
            },
        ]

    def _truncate_to_budget(self, context: str, available_tokens: int) -> str:
        lines = context.split("\n")
        compressed_lines: list[str] = []
        used_tokens = 0

        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens

        self.logger.debug(
            "🧩 截断完成: kept_lines=%s used_tokens=%s budget=%s",
            len(compressed_lines),
            used_tokens,
            available_tokens,
        )
        return "\n".join(compressed_lines)

    @staticmethod
    def _has_retrieval_content(result: str) -> bool:
        if not result or not result.strip():
            return False
        invalid_markers = ("未找到", "未检索到", "检索到 0 条", "错误")
        return not any(marker in result for marker in invalid_markers)


def count_tokens(text: str) -> int:
    """计算文本token数（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4
