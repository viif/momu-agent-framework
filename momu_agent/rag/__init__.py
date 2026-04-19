"""RAG 模块导出。"""

from .document import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)
from .pipeline import (
    create_rag_pipeline,
    embed_query,
    index_chunks,
    load_and_chunk_texts,
    merge_snippets,
    rank,
    search_vectors,
)

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "create_document",
    "load_text_file",
    "load_and_chunk_texts",
    "index_chunks",
    "embed_query",
    "search_vectors",
    "rank",
    "merge_snippets",
    "create_rag_pipeline",
]
