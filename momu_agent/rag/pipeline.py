"""RAG 检索与索引流程"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.llm import LLM
from ..storage.document import DocumentStore, SQLiteDocumentStore
from ..storage.vector import ChromaVectorStore, VectorStore
from ..utils.embedding import EmbeddingModel, get_dimension, get_text_embedder
from .document import DocumentProcessor, load_text_file


def load_and_chunk_texts(
    file_paths: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    processor: DocumentProcessor | None = None,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    """加载文件并转换为可索引分块。"""
    # 优先复用外部传入的处理器，便于在 pipeline 外统一配置切分策略。
    doc_processor = processor or DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    items: list[dict[str, Any]] = []
    # 基于 chunk 内容去重，避免重复文本被重复索引。
    seen_hashes: set[str] = set()

    for file_path in file_paths:
        document = load_text_file(file_path)
        chunks = doc_processor.process_document(document)
        source = str(Path(file_path))

        for chunk in chunks:
            content = chunk.content.strip()
            if not content:
                continue

            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # 将来源、分片信息和 namespace 合并到 metadata，供后续检索过滤。
            metadata = {
                "source": source,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content_hash": content_hash,
                "rag_namespace": namespace,
            }
            metadata.update(chunk.metadata)

            items.append(
                {
                    "id": chunk.chunk_id,
                    "content": content,
                    "metadata": metadata,
                }
            )

    return items


def _normalize_vector(vector: Any, target_dim: int) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if not isinstance(vector, list):
        vector = list(vector)

    numeric = [float(v) for v in vector]
    if len(numeric) == target_dim:
        return numeric
    if len(numeric) > target_dim:
        return numeric[:target_dim]
    return numeric + [0.0] * (target_dim - len(numeric))


async def index_chunks(
    chunks: list[dict[str, Any]],
    *,
    store: VectorStore | None = None,
    document_store: DocumentStore | None = None,
    embedder: EmbeddingModel | None = None,
    batch_size: int = 32,
    namespace: str = "default",
) -> int:
    """将分块写入文档库和向量库。"""
    if not chunks:
        return 0

    vector_store = store or ChromaVectorStore()
    doc_store = document_store or SQLiteDocumentStore()
    text_embedder = embedder or get_text_embedder()

    expected_dim = int(get_dimension())

    all_vectors: list[list[float]] = []
    all_metadata: list[dict[str, Any]] = []
    all_ids: list[str] = []

    # 分批 embedding，减少一次性编码造成的内存和延迟压力。
    for i in range(0, len(chunks), max(1, batch_size)):
        batch = chunks[i : i + max(1, batch_size)]
        contents = [item.get("content", "") for item in batch]
        embeddings = text_embedder.encode(contents)
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        for item, vec in zip(batch, embeddings):
            chunk_id = str(
                item.get("id") or hashlib.md5(str(item).encode()).hexdigest()
            )
            content = str(item.get("content") or "")
            metadata = dict(item.get("metadata") or {})
            # 统一补齐 RAG 存储字段，保证向量库和文档库可用同一套元数据。
            metadata.update(
                {
                    "memory_id": chunk_id,
                    "memory_type": "rag_chunk",
                    "is_rag_data": True,
                    "rag_namespace": namespace,
                    "content": content,
                    "timestamp": int(time.time()),
                    "importance": float(metadata.get("importance", 0.5)),
                }
            )

            normalized = _normalize_vector(vec, expected_dim)
            # 先写入文档存储，作为向量检索失败时的兜底数据源。
            await doc_store.add_memory(
                memory_id=chunk_id,
                user_id=str(metadata.get("user_id", "rag_user")),
                content=content,
                memory_type="rag_chunk",
                timestamp=int(metadata.get("timestamp", int(time.time()))),
                importance=float(metadata.get("importance", 0.5)),
                properties=metadata,
            )

            all_vectors.append(normalized)
            all_metadata.append(metadata)
            all_ids.append(chunk_id)

    await vector_store.add_vectors(
        vectors=all_vectors, metadata=all_metadata, ids=all_ids
    )
    return len(all_ids)


def embed_query(
    query: str,
    *,
    embedder: EmbeddingModel | None = None,
    expected_dim: int | None = None,
) -> list[float]:
    """将查询文本编码为统一维度向量。"""
    text_embedder = embedder or get_text_embedder()
    dim = int(expected_dim or get_dimension())
    vector = text_embedder.encode(query)
    return _normalize_vector(vector, dim)


# MQE：通过提示词让 LLM 生成多个语义相关但表达不同的查询。
def _build_mqe_messages(query: str, n: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是检索改写助手。请生成语义相关但表达不同的查询，"
                "提升向量检索的召回率。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"原始问题：{query}\n"
                f"请给出 {n} 条不同改写，每行一条，不要编号，不要解释。"
            ),
        },
    ]


async def _prompt_mqe(query: str, n: int, llm: LLM | None = None) -> list[str]:
    # 扩展能力是可选增强：LLM 不可用时直接降级为不扩展。
    if llm is None or not query.strip() or n <= 0:
        return []

    try:
        content = await llm.invoke(_build_mqe_messages(query, n))
    except Exception:
        # 生成失败不影响主检索链路，调用方会回退到原始 query。
        return []

    queries: list[str] = []
    for line in content.splitlines():
        # 兼容常见模型输出格式（项目符号、数字编号）并抽取纯查询文本。
        candidate = line.strip().lstrip("-*• ").strip()
        if not candidate:
            continue
        dot_parts = candidate.split(".", 1)
        if len(dot_parts) == 2 and dot_parts[0].strip().isdigit():
            candidate = dot_parts[1].strip()
        comma_parts = candidate.split("、", 1)
        if len(comma_parts) == 2 and comma_parts[0].strip().isdigit():
            candidate = comma_parts[1].strip()
        if candidate:
            queries.append(candidate)

    # 去重后截断，稳定控制扩展查询数量上限。
    return _dedupe_queries(queries)[:n]


# HyDE：先生成“假设答案段落”，再把该段落当作检索查询以补足语义线索。
def _build_hyde_messages(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是检索增强助手。请根据问题生成一段客观、简洁、"
                "信息密度高的假设文档，用于向量检索。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{query}\n请直接输出一段假设文档内容，不要分点，不要解释。"
            ),
        },
    ]


async def _prompt_hyde(query: str, llm: LLM | None = None) -> str | None:
    # 与 MQE 一样，HyDE 失败只关闭增强，不中断主路径。
    if llm is None or not query.strip():
        return None

    try:
        content = await llm.invoke(_build_hyde_messages(query))
    except Exception:
        return None

    candidate = content.strip()
    return candidate or None


def _dedupe_queries(queries: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)

    return result


async def _search_single_query(
    query: str,
    *,
    top_k: int,
    score_threshold: float | None,
    namespace: str,
    vector_store: VectorStore,
    doc_store: DocumentStore,
    embedder: EmbeddingModel | None,
    where: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    query_vector = embed_query(query, embedder=embedder)
    # 默认按 memory_type + namespace 约束检索范围，避免跨场景污染结果。
    conditions: dict[str, Any] = {
        "memory_type": "rag_chunk",
        "rag_namespace": namespace,
    }
    if where:
        conditions.update(where)

    try:
        # 主路径：走向量检索，获取语义相近结果。
        return await vector_store.search_similar(
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            where=conditions,
        )
    except Exception:
        # 兜底路径：向量检索异常时退化为文档库 + 关键词打分。
        docs = await doc_store.search_memories(
            memory_type="rag_chunk", limit=max(top_k * 4, 20)
        )
        results: list[dict[str, Any]] = []
        for doc in docs:
            props = dict(doc.get("properties") or {})
            if props.get("rag_namespace") != namespace:
                continue
            content = str(doc.get("content") or "")
            score = _keyword_score(query, content)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                {
                    "id": doc.get("memory_id"),
                    "score": score,
                    "metadata": {
                        **props,
                        "memory_id": doc.get("memory_id"),
                        "content": content,
                    },
                }
            )
        results.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return results[:top_k]


def _merge_hits_by_memory_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 多查询命中同一片段时按 memory_id 去重，并保留最高语义分。
    merged: dict[str, dict[str, Any]] = {}

    for item in items:
        metadata = dict(item.get("metadata") or {})
        memory_id = str(metadata.get("memory_id") or item.get("id") or "")
        if not memory_id:
            memory_id = hashlib.md5(str(item).encode("utf-8")).hexdigest()

        current = merged.get(memory_id)
        if current is None or float(item.get("score") or 0.0) > float(
            current.get("score") or 0.0
        ):
            updated = {
                **item,
                "metadata": {**metadata, "memory_id": memory_id},
            }
            merged[memory_id] = updated

    results = list(merged.values())
    results.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return results


async def search_vectors(
    query: str,
    *,
    top_k: int = 8,
    score_threshold: float | None = None,
    namespace: str = "default",
    store: VectorStore | None = None,
    document_store: DocumentStore | None = None,
    embedder: EmbeddingModel | None = None,
    where: dict[str, Any] | None = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
    llm: LLM | None = None,
) -> list[dict[str, Any]]:
    """执行检索，支持 MQE/HyDE 扩展并在异常时回退关键词检索。"""
    normalized_query = query.strip()
    if not normalized_query:
        return []

    vector_store = store or ChromaVectorStore()
    doc_store = document_store or SQLiteDocumentStore()

    # 兼容默认行为：不开启扩展时完全复用原单查询路径。
    should_expand = (enable_mqe and mqe_expansions > 0) or enable_hyde
    if not should_expand:
        return await _search_single_query(
            normalized_query,
            top_k=top_k,
            score_threshold=score_threshold,
            namespace=namespace,
            vector_store=vector_store,
            doc_store=doc_store,
            embedder=embedder,
            where=where,
        )

    # 组合扩展框架：原始 query + MQE 改写 + HyDE 伪文档。
    expanded_queries = [normalized_query]
    if enable_mqe and mqe_expansions > 0:
        expanded_queries.extend(
            await _prompt_mqe(normalized_query, n=mqe_expansions, llm=llm)
        )
    if enable_hyde:
        hyde_text = await _prompt_hyde(normalized_query, llm=llm)
        if hyde_text:
            expanded_queries.append(hyde_text)

    # 扩展结果为空时自动退化，保证行为稳定可预期。
    deduped_queries = _dedupe_queries(expanded_queries)
    if len(deduped_queries) <= 1:
        return await _search_single_query(
            normalized_query,
            top_k=top_k,
            score_threshold=score_threshold,
            namespace=namespace,
            vector_store=vector_store,
            doc_store=doc_store,
            embedder=embedder,
            where=where,
        )

    # 用候选池控制多路召回规模，再按查询数均分每路检索额度。
    pool_size = max(top_k * max(1, candidate_pool_multiplier), 20)
    per_query_k = max(1, pool_size // len(deduped_queries))

    tasks = [
        _search_single_query(
            expanded_query,
            top_k=per_query_k,
            score_threshold=score_threshold,
            namespace=namespace,
            vector_store=vector_store,
            doc_store=doc_store,
            embedder=embedder,
            where=where,
        )
        for expanded_query in deduped_queries
    ]
    settled = await asyncio.gather(*tasks, return_exceptions=True)

    merged_inputs: list[dict[str, Any]] = []
    for item in settled:
        if isinstance(item, BaseException):
            continue
        merged_inputs.extend(item)

    # 当所有扩展路由都不可用时，回退到原始 query 的单路检索。
    if not merged_inputs:
        return await _search_single_query(
            normalized_query,
            top_k=top_k,
            score_threshold=score_threshold,
            namespace=namespace,
            vector_store=vector_store,
            doc_store=doc_store,
            embedder=embedder,
            where=where,
        )

    merged = _merge_hits_by_memory_id(merged_inputs)
    return merged[:top_k]


def _keyword_score(query: str, content: str) -> float:
    q_terms = [term for term in query.lower().split() if term]
    if not q_terms:
        return 0.0
    text = content.lower()
    hits = sum(1 for term in q_terms if term in text)
    return hits / len(q_terms)


def rank(
    items: list[dict[str, Any]],
    query: str,
    *,
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in items:
        metadata = dict(item.get("metadata") or {})
        content = str(metadata.get("content") or "")
        vector_score = float(item.get("score") or 0.0)
        keyword_score = _keyword_score(query, content)
        # 混合语义分与关键词分，提升语义召回下的可解释性。
        final_score = vector_score * vector_weight + keyword_score * keyword_weight
        ranked.append(
            {
                "memory_id": str(metadata.get("memory_id") or item.get("id") or ""),
                "score": final_score,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "content": content,
                "metadata": metadata,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def merge_snippets(items: list[dict[str, Any]], max_chars: int = 3000) -> str:
    segments: list[str] = []
    total = 0
    # 按排序结果顺序拼接，严格控制上下文长度以适配下游模型输入。
    for item in items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if total + len(content) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                segments.append(content[:remaining])
            break
        segments.append(content)
        total += len(content) + 2
    return "\n\n".join(segments)


def create_rag_pipeline(
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 8,
    namespace: str = "default",
    vector_store: VectorStore | None = None,
    document_store: DocumentStore | None = None,
    embedder: EmbeddingModel | None = None,
    llm: LLM | None = None,
) -> dict[str, Callable[..., Any]]:
    """创建包含入库、检索与统计能力的 pipeline。"""
    vs = vector_store or ChromaVectorStore()
    ds = document_store or SQLiteDocumentStore()
    emb = embedder or get_text_embedder()
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 暴露 add/search/stats 三个闭包并复用同一组依赖。
    async def add_documents(file_paths: list[str]) -> int:
        chunks = load_and_chunk_texts(
            file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            processor=processor,
            namespace=namespace,
        )
        return await index_chunks(
            chunks,
            store=vs,
            document_store=ds,
            embedder=emb,
            namespace=namespace,
        )

    async def search(
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float | None = None,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
    ) -> list[dict[str, Any]]:
        hits = await search_vectors(
            query,
            top_k=limit or top_k,
            score_threshold=score_threshold,
            namespace=namespace,
            store=vs,
            document_store=ds,
            embedder=emb,
            enable_mqe=enable_mqe,
            mqe_expansions=mqe_expansions,
            enable_hyde=enable_hyde,
            candidate_pool_multiplier=candidate_pool_multiplier,
            llm=llm,
        )
        return rank(hits, query)

    async def get_stats() -> dict[str, Any]:
        vector_stats = await vs.get_collection_stats()
        doc_stats = await ds.get_database_stats()
        return {
            "namespace": namespace,
            "vector": vector_stats,
            "document": doc_stats,
        }

    async def close() -> None:
        try:
            await vs.close()
        except Exception:
            pass
        try:
            await ds.close()
        except Exception:
            pass

    return {
        "add_documents": add_documents,
        "search": search,
        "get_stats": get_stats,
        "close": close,
    }
