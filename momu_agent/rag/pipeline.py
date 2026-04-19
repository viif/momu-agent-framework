"""RAG pipeline 核心流程。"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    doc_processor = processor or DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    items: list[dict[str, Any]] = []
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
    if not chunks:
        return 0

    vector_store = store or ChromaVectorStore()
    doc_store = document_store or SQLiteDocumentStore()
    text_embedder = embedder or get_text_embedder()

    expected_dim = int(get_dimension())

    all_vectors: list[list[float]] = []
    all_metadata: list[dict[str, Any]] = []
    all_ids: list[str] = []

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
    text_embedder = embedder or get_text_embedder()
    dim = int(expected_dim or get_dimension())
    vector = text_embedder.encode(query)
    return _normalize_vector(vector, dim)


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
) -> list[dict[str, Any]]:
    vector_store = store or ChromaVectorStore()
    doc_store = document_store or SQLiteDocumentStore()

    query_vector = embed_query(query, embedder=embedder)
    conditions: dict[str, Any] = {
        "memory_type": "rag_chunk",
        "rag_namespace": namespace,
    }
    if where:
        conditions.update(where)

    try:
        return await vector_store.search_similar(
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            where=conditions,
        )
    except Exception:
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
) -> dict[str, Callable[..., Any]]:
    vs = vector_store or ChromaVectorStore()
    ds = document_store or SQLiteDocumentStore()
    emb = embedder or get_text_embedder()
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
    ) -> list[dict[str, Any]]:
        hits = await search_vectors(
            query,
            top_k=limit or top_k,
            score_threshold=score_threshold,
            namespace=namespace,
            store=vs,
            document_store=ds,
            embedder=emb,
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

    return {
        "add_documents": add_documents,
        "search": search,
        "get_stats": get_stats,
    }
