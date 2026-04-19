"""RAG 文档处理模块。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """文档实体。"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None

    def __post_init__(self) -> None:
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode("utf-8")).hexdigest()


@dataclass
class DocumentChunk:
    """文档分块实体。"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None
    doc_id: str | None = None
    chunk_index: int = 0

    def __post_init__(self) -> None:
        if self.chunk_id is None:
            basis = f"{self.doc_id or ''}:{self.chunk_index}:{self.content[:64]}"
            self.chunk_id = hashlib.md5(basis.encode("utf-8")).hexdigest()


class DocumentProcessor:
    """文档分块处理器。"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size - 1)
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]

    def process_document(self, document: Document) -> list[DocumentChunk]:
        chunks = self._split_text(document.content)
        total = len(chunks)
        processed: list[DocumentChunk] = []

        for idx, content in enumerate(chunks):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "doc_id": document.doc_id,
                    "chunk_index": idx,
                    "total_chunks": total,
                }
            )
            processed.append(
                DocumentChunk(
                    content=content,
                    metadata=metadata,
                    doc_id=document.doc_id,
                    chunk_index=idx,
                )
            )

        return processed

    def process_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        for document in documents:
            all_chunks.extend(self.process_document(document))
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) <= self.chunk_size:
            return [stripped]

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            split = self._find_split_point(text, start, end)
            if split <= start:
                split = end

            chunk = text[start:split].strip()
            if chunk:
                chunks.append(chunk)

            if split >= text_len:
                break

            next_start = max(split - self.chunk_overlap, start + 1)
            start = next_start

        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        for separator in self.separators:
            window_start = max(start, end - 100)
            for i in range(end - len(separator), window_start - 1, -1):
                if text[i : i + len(separator)] == separator:
                    return i + len(separator)
        return -1

    def merge_chunks(
        self, chunks: list[DocumentChunk], max_length: int = 2000
    ) -> list[DocumentChunk]:
        if not chunks:
            return []

        merged: list[DocumentChunk] = []
        current = DocumentChunk(
            content=chunks[0].content,
            metadata=dict(chunks[0].metadata),
            doc_id=chunks[0].doc_id,
            chunk_index=chunks[0].chunk_index,
        )

        for chunk in chunks[1:]:
            can_merge = (
                current.doc_id == chunk.doc_id
                and len(current.content) + 1 + len(chunk.content) <= max_length
            )
            if can_merge:
                current.content = f"{current.content}\n{chunk.content}"
                current.metadata["merged_chunks"] = (
                    current.metadata.get("merged_chunks", 1) + 1
                )
                continue

            merged.append(current)
            current = DocumentChunk(
                content=chunk.content,
                metadata=dict(chunk.metadata),
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
            )

        merged.append(current)
        return merged

    def filter_chunks(
        self, chunks: list[DocumentChunk], min_length: int = 50
    ) -> list[DocumentChunk]:
        return [chunk for chunk in chunks if len(chunk.content.strip()) >= min_length]

    def add_chunk_metadata(
        self, chunks: list[DocumentChunk], metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks


def _load_with_markitdown(file_path: str) -> str | None:
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None

    try:
        result = MarkItDown().convert(file_path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return None
    except Exception:
        return None


def _load_with_fallback(file_path: str, encoding: str = "utf-8") -> str:
    with open(file_path, "r", encoding=encoding, errors="ignore") as file:
        return file.read()


def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
    path = Path(file_path)
    content = _load_with_markitdown(file_path)
    if content is None:
        content = _load_with_fallback(file_path, encoding=encoding)

    metadata = {
        "source": str(path),
        "file_ext": path.suffix.lower(),
        "type": "text_file",
        "loaded_at": datetime.now().isoformat(),
    }
    return Document(content=content, metadata=metadata)


def create_document(content: str, **metadata: Any) -> Document:
    return Document(content=content, metadata=dict(metadata))
