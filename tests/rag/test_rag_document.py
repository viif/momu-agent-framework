from pathlib import Path

import pytest

from momu_agent.rag.document import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)


def test_document_auto_doc_id_stable():
    a = Document(content="hello world", metadata={})
    b = Document(content="hello world", metadata={})
    assert a.doc_id == b.doc_id


def test_document_chunk_auto_chunk_id_stable():
    a = DocumentChunk(content="chunk", metadata={}, doc_id="d1", chunk_index=0)
    b = DocumentChunk(content="chunk", metadata={}, doc_id="d1", chunk_index=0)
    assert a.chunk_id == b.chunk_id


def test_process_document_splits_and_sets_metadata():
    processor = DocumentProcessor(chunk_size=12, chunk_overlap=4)
    doc = Document(content="abc def ghi jkl mno pqr", metadata={"source": "s"})

    chunks = processor.process_document(doc)

    assert len(chunks) >= 2
    assert all(chunk.doc_id == doc.doc_id for chunk in chunks)
    assert chunks[0].metadata["source"] == "s"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == len(chunks)


def test_chunk_overlap_larger_than_chunk_size_is_clamped():
    processor = DocumentProcessor(chunk_size=8, chunk_overlap=100)
    doc = Document(content="01234567890123456789", metadata={})

    chunks = processor.process_document(doc)

    assert len(chunks) > 1
    assert processor.chunk_overlap == 7


def test_filter_merge_and_add_metadata():
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=5)
    chunks = [
        DocumentChunk(content="short", metadata={}, doc_id="d1", chunk_index=0),
        DocumentChunk(
            content="this is long enough",
            metadata={},
            doc_id="d1",
            chunk_index=1,
        ),
        DocumentChunk(content="tail", metadata={}, doc_id="d1", chunk_index=2),
    ]

    filtered = processor.filter_chunks(chunks, min_length=8)
    assert len(filtered) == 1

    merged = processor.merge_chunks(
        [
            DocumentChunk(content="a", metadata={}, doc_id="d1", chunk_index=0),
            DocumentChunk(content="b", metadata={}, doc_id="d1", chunk_index=1),
        ],
        max_length=10,
    )
    assert len(merged) == 1
    assert "a\nb" in merged[0].content

    with_meta = processor.add_chunk_metadata(filtered, {"namespace": "n1"})
    assert with_meta[0].metadata["namespace"] == "n1"


def test_create_document():
    doc = create_document("content", source="local")
    assert doc.content == "content"
    assert doc.metadata["source"] == "local"


def test_load_text_file_reads_plain_text(tmp_path: Path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")

    doc = load_text_file(str(file_path))

    assert doc.content == "hello"
    assert doc.metadata["source"] == str(file_path)
    assert doc.metadata["file_ext"] == ".txt"


def test_load_text_file_falls_back_when_markitdown_missing(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "b.md"
    file_path.write_text("fallback", encoding="utf-8")

    import momu_agent.rag.document as document_module

    monkeypatch.setattr(document_module, "_load_with_markitdown", lambda _: None)

    doc = load_text_file(str(file_path))
    assert doc.content == "fallback"


def test_invalid_chunk_parameters_raise():
    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=0)

    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=10, chunk_overlap=-1)
