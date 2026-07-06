from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from chatbot_api.providers import ChatCitation, ChatCompletion, ChatProviderError, ChatTurn
from chatbot_api.repositories import RetrievedDocumentChunk
from chatbot_api.retrieval import (
    DocumentRetriever,
    build_retrieval_prompt,
    parse_rerank_response,
    select_retrieved_chunks,
)


class StubEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = embeddings
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self._embeddings


class StubDocumentRepository:
    def __init__(self, chunks: list[RetrievedDocumentChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[list[float], int, str | None]] = []

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        self.calls.append((query_embedding, limit, owner_user_id))
        return self._chunks[:limit]


def make_chunk(
    *,
    document_id: str,
    chunk_index: int,
    score: float,
    filename: str | None = None,
    content: str | None = None,
) -> RetrievedDocumentChunk:
    resolved_filename = filename or f"{document_id}.md"
    resolved_content = content or f"content for {document_id} chunk {chunk_index}"
    return RetrievedDocumentChunk(
        document_id=document_id,
        filename=resolved_filename,
        chunk_index=chunk_index,
        content=resolved_content,
        start_offset=chunk_index * 100,
        end_offset=(chunk_index * 100) + len(resolved_content),
        metadata=None,
        score=score,
    )


class StubChatProvider:
    def __init__(
        self,
        *,
        content: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._error = error
        self.calls: list[list[ChatTurn]] = []

    async def generate_response(
        self,
        messages: Sequence[ChatTurn],
        *,
        tools: Sequence[object] = (),
        previous_response_id: str | None = None,
        tool_outputs: Sequence[object] = (),
    ) -> ChatCompletion:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        return ChatCompletion(content=self._content or "", provider="openai", model="gpt-4.1-mini")


@pytest.mark.anyio
async def test_retriever_skips_whitespace_queries_without_embedding_or_search() -> None:
    embedding_provider = StubEmbeddingProvider([[1.0, 0.0]])
    repository = StubDocumentRepository([make_chunk(document_id="doc-1", chunk_index=0, score=0.9)])
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=4,
        min_score=0.35,
        max_chunks_per_document=1,
        candidate_limit=12,
    )

    result = await retriever.retrieve("   ")

    assert result is None
    assert embedding_provider.calls == []
    assert repository.calls == []


@pytest.mark.anyio
async def test_retriever_filters_low_score_candidates_and_returns_none() -> None:
    embedding_provider = StubEmbeddingProvider([[0.5, 0.5]])
    repository = StubDocumentRepository(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.34),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.2),
        ]
    )
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=4,
        min_score=0.35,
        max_chunks_per_document=1,
        candidate_limit=12,
    )

    result = await retriever.retrieve("what does the guide say?")

    assert result is None
    assert embedding_provider.calls == [["what does the guide say?"]]
    assert repository.calls == [([0.5, 0.5], 12, None)]


@pytest.mark.anyio
async def test_retriever_deduplicates_by_document_and_keeps_score_order() -> None:
    embedding_provider = StubEmbeddingProvider([[0.1, 0.9]])
    repository = StubDocumentRepository(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.95, filename="guide.md"),
            make_chunk(document_id="doc-1", chunk_index=1, score=0.9, filename="guide.md"),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.8, filename="faq.md"),
            make_chunk(document_id="doc-3", chunk_index=0, score=0.72, filename="runbook.md"),
        ]
    )
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=3,
        min_score=0.35,
        max_chunks_per_document=1,
        candidate_limit=12,
    )

    result = await retriever.retrieve("summarize the docs")

    assert result is not None
    assert result.prompt == build_retrieval_prompt(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.95, filename="guide.md"),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.8, filename="faq.md"),
            make_chunk(document_id="doc-3", chunk_index=0, score=0.72, filename="runbook.md"),
        ]
    )
    assert result.metadata is not None
    assert result.metadata.citations == [
        ChatCitation(
            document_id="doc-1",
            filename="guide.md",
            chunk_index=0,
            start_offset=0,
            end_offset=len("content for doc-1 chunk 0"),
            snippet="content for doc-1 chunk 0",
        ),
        ChatCitation(
            document_id="doc-2",
            filename="faq.md",
            chunk_index=0,
            start_offset=0,
            end_offset=len("content for doc-2 chunk 0"),
            snippet="content for doc-2 chunk 0",
        ),
        ChatCitation(
            document_id="doc-3",
            filename="runbook.md",
            chunk_index=0,
            start_offset=0,
            end_offset=len("content for doc-3 chunk 0"),
            snippet="content for doc-3 chunk 0",
        ),
    ]


def test_select_retrieved_chunks_applies_threshold_and_per_document_limit() -> None:
    chunks = [
        make_chunk(document_id="doc-1", chunk_index=0, score=0.9),
        make_chunk(document_id="doc-1", chunk_index=1, score=0.88),
        make_chunk(document_id="doc-2", chunk_index=0, score=0.8),
        make_chunk(document_id="doc-3", chunk_index=0, score=0.2),
    ]

    selected = select_retrieved_chunks(
        chunks,
        top_k=4,
        min_score=0.35,
        max_chunks_per_document=1,
    )

    assert [(chunk.document_id, chunk.chunk_index) for chunk in selected] == [
        ("doc-1", 0),
        ("doc-2", 0),
    ]


def test_build_retrieval_prompt_contains_conservative_instructions() -> None:
    prompt = build_retrieval_prompt(
        [make_chunk(document_id="doc-1", chunk_index=0, score=0.91, filename="guide.md")]
    )

    assert "only when it directly supports the user's question" in prompt
    assert "answer conservatively" in prompt
    assert "Do not imply source-backed certainty" in prompt
    assert "[Source 1] file=guide.md document_id=doc-1 chunk=0" in prompt


@pytest.mark.anyio
async def test_retrieve_chunks_reorders_by_rerank_when_enabled() -> None:
    embedding_provider = StubEmbeddingProvider([[0.1, 0.9]])
    repository = StubDocumentRepository(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.9),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.85),
            make_chunk(document_id="doc-3", chunk_index=0, score=0.8),
        ]
    )
    chat_provider = StubChatProvider(
        content=json.dumps(
            {
                "rankings": [
                    {"index": 2, "relevance": 0.99},
                    {"index": 0, "relevance": 0.5},
                    {"index": 1, "relevance": 0.4},
                ]
            }
        )
    )
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=3,
        min_score=0.0,
        max_chunks_per_document=1,
        candidate_limit=3,
        chat_provider=chat_provider,
        rerank_enabled=True,
    )

    chunks = await retriever.retrieve_chunks("summarize the docs")

    assert [chunk.document_id for chunk in chunks] == ["doc-3", "doc-1", "doc-2"]
    assert len(chat_provider.calls) == 1


@pytest.mark.anyio
async def test_retrieve_chunks_falls_back_to_original_order_on_invalid_rerank_json() -> None:
    embedding_provider = StubEmbeddingProvider([[0.1, 0.9]])
    repository = StubDocumentRepository(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.9),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.85),
        ]
    )
    chat_provider = StubChatProvider(content="not valid json")
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=2,
        min_score=0.0,
        max_chunks_per_document=1,
        candidate_limit=2,
        chat_provider=chat_provider,
        rerank_enabled=True,
    )

    chunks = await retriever.retrieve_chunks("summarize the docs")

    assert [chunk.document_id for chunk in chunks] == ["doc-1", "doc-2"]


@pytest.mark.anyio
async def test_retrieve_chunks_falls_back_to_original_order_on_provider_error() -> None:
    embedding_provider = StubEmbeddingProvider([[0.1, 0.9]])
    repository = StubDocumentRepository(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.9),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.85),
        ]
    )
    chat_provider = StubChatProvider(error=ChatProviderError("boom"))
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=2,
        min_score=0.0,
        max_chunks_per_document=1,
        candidate_limit=2,
        chat_provider=chat_provider,
        rerank_enabled=True,
    )

    chunks = await retriever.retrieve_chunks("summarize the docs")

    assert [chunk.document_id for chunk in chunks] == ["doc-1", "doc-2"]


@pytest.mark.anyio
async def test_retrieve_chunks_never_invokes_provider_when_rerank_disabled() -> None:
    embedding_provider = StubEmbeddingProvider([[0.1, 0.9]])
    repository = StubDocumentRepository(
        [
            make_chunk(document_id="doc-1", chunk_index=0, score=0.9),
            make_chunk(document_id="doc-2", chunk_index=0, score=0.85),
        ]
    )
    chat_provider = StubChatProvider(
        content=json.dumps({"rankings": [{"index": 1, "relevance": 0.99}]})
    )
    retriever = DocumentRetriever(
        repository,
        embedding_provider,
        top_k=2,
        min_score=0.0,
        max_chunks_per_document=1,
        candidate_limit=2,
        chat_provider=chat_provider,
        rerank_enabled=False,
    )

    chunks = await retriever.retrieve_chunks("summarize the docs")

    assert [chunk.document_id for chunk in chunks] == ["doc-1", "doc-2"]
    assert chat_provider.calls == []


def test_parse_rerank_response_preserves_unmentioned_chunks() -> None:
    chunks = [
        make_chunk(document_id="doc-1", chunk_index=0, score=0.9),
        make_chunk(document_id="doc-2", chunk_index=0, score=0.85),
        make_chunk(document_id="doc-3", chunk_index=0, score=0.8),
    ]
    content = json.dumps({"rankings": [{"index": 2, "relevance": 0.9}]})

    reranked = parse_rerank_response(content, chunks)

    assert reranked is not None
    assert [chunk.document_id for chunk in reranked] == ["doc-3", "doc-1", "doc-2"]
