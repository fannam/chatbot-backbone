from __future__ import annotations

from chatbot_api.document_ingestion import TextChunker


def test_chunker_uses_fixed_overlap_when_hard_splitting() -> None:
    text = "x" * 2600

    chunker = TextChunker(chunk_size=1200, chunk_overlap=200)
    chunks = chunker.chunk_text(text)

    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 1200),
        (1000, 2200),
        (2000, 2600),
    ]
    assert all(chunk.content == text[chunk.start_offset : chunk.end_offset] for chunk in chunks)


def test_chunker_prefers_paragraph_boundaries_when_available() -> None:
    text = ("A" * 700) + "\n\n" + ("B" * 700)

    chunker = TextChunker(chunk_size=1200, chunk_overlap=200)
    chunks = chunker.chunk_text(text)

    assert len(chunks) == 2
    assert chunks[0].end_offset == 702
    assert chunks[0].content.endswith("\n\n")


def test_chunker_returns_single_chunk_for_short_text() -> None:
    chunker = TextChunker(chunk_size=1200, chunk_overlap=200)

    chunks = chunker.chunk_text("short text")

    assert len(chunks) == 1
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len("short text")
