from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatbot_api.observability import ObservabilityService
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletionMetadata,
    ChatProvider,
    ChatProviderError,
    ChatTurn,
    ToolCallBatch,
)
from chatbot_api.retrieval.embeddings import EmbeddingProvider
from chatbot_api.text_utils import strip_markdown_code_fence
from chatbot_api.tracing import NoopTraceSink, TraceSink

if TYPE_CHECKING:
    from chatbot_api.repositories import DocumentRepository, RetrievedDocumentChunk

RETRIEVAL_SNIPPET_MAX_CHARS = 240
RERANK_SYSTEM_PROMPT = (
    "You rank retrieved document chunks by relevance to a user's query. "
    "Return strict JSON only, never call tools."
)


class RerankedItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    relevance: float = Field(ge=0.0, le=1.0)


class RerankResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rankings: list[RerankedItem] = Field(default_factory=list)


def build_rerank_prompt(query: str, chunks: list[RetrievedDocumentChunk]) -> str:
    candidates = "\n".join(
        f"[{index}] file={chunk.filename} chunk={chunk.chunk_index}\n{chunk.content}"
        for index, chunk in enumerate(chunks)
    )
    return (
        f"Query:\n{query}\n\n"
        f"Candidates:\n{candidates}\n\n"
        "Return strict JSON with shape "
        '{"rankings":[{"index":0,"relevance":0.0}]}, '
        "ordering the array by relevance descending. Include every candidate index "
        "exactly once."
    )


def parse_rerank_response(
    content: str,
    chunks: list[RetrievedDocumentChunk],
) -> list[RetrievedDocumentChunk] | None:
    payload_text = strip_markdown_code_fence(content.strip())
    if not payload_text:
        return None

    try:
        raw_payload = json.loads(payload_text)
        payload = RerankResponse.model_validate(raw_payload)
    except (json.JSONDecodeError, ValidationError):
        return None

    ordered_items = sorted(payload.rankings, key=lambda item: item.relevance, reverse=True)
    reranked: list[RetrievedDocumentChunk] = []
    seen_indexes: set[int] = set()
    for item in ordered_items:
        if item.index in seen_indexes or not (0 <= item.index < len(chunks)):
            continue
        seen_indexes.add(item.index)
        reranked.append(chunks[item.index])

    for index, chunk in enumerate(chunks):
        if index not in seen_indexes:
            reranked.append(chunk)

    return reranked


@dataclass(frozen=True)
class RetrievalResult:
    prompt: str
    metadata: ChatCompletionMetadata


class DocumentRetriever:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        *,
        top_k: int,
        min_score: float,
        max_chunks_per_document: int,
        candidate_limit: int,
        observability: ObservabilityService | None = None,
        trace_sink: TraceSink | None = None,
        chat_provider: ChatProvider | None = None,
        rerank_enabled: bool = False,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._top_k = top_k
        self._min_score = min_score
        self._max_chunks_per_document = max_chunks_per_document
        self._candidate_limit = candidate_limit
        self._observability = observability
        self._trace_sink = trace_sink or NoopTraceSink()
        self._chat_provider = chat_provider
        self._rerank_enabled = rerank_enabled

    async def retrieve(
        self,
        query: str,
        *,
        owner_user_id: str | None = None,
    ) -> RetrievalResult | None:
        selected_chunks = await self.retrieve_chunks(query, owner_user_id=owner_user_id)
        if not selected_chunks:
            return None

        return RetrievalResult(
            prompt=build_retrieval_prompt(selected_chunks),
            metadata=ChatCompletionMetadata(
                citations=[build_citation(chunk) for chunk in selected_chunks]
            ),
        )

    async def retrieve_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chunks_per_document: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        started_at = perf_counter()
        normalized_query = query.strip()
        resolved_top_k = self._top_k if top_k is None else top_k
        resolved_max_chunks_per_document = (
            self._max_chunks_per_document
            if max_chunks_per_document is None
            else max_chunks_per_document
        )
        span = self._trace_sink.start_span(
            "retrieval.retrieve_chunks",
            run_type="retriever",
            inputs={
                "query": normalized_query,
                "top_k": resolved_top_k,
                "candidate_limit": self._candidate_limit,
                "max_chunks_per_document": resolved_max_chunks_per_document,
            },
            tags=["retrieval"],
        )
        with span:
            if (
                not normalized_query
                or resolved_top_k <= 0
                or self._candidate_limit <= 0
                or resolved_max_chunks_per_document <= 0
            ):
                duration_seconds = perf_counter() - started_at
                self._record_retrieval(
                    outcome="skipped",
                    selected_chunks=[],
                    duration_seconds=duration_seconds,
                    query=normalized_query,
                )
                span.finish_success(
                    outputs={
                        "outcome": "skipped",
                        "selected_chunk_count": 0,
                        "duration_ms": round(duration_seconds * 1000, 6),
                    }
                )
                return []

            try:
                query_embedding = await self._embed_query(normalized_query)
                chunks = await self._repository.search_similar_chunks(
                    query_embedding=query_embedding,
                    limit=max(resolved_top_k, self._candidate_limit),
                    owner_user_id=owner_user_id,
                )
                chunks, rerank_applied = await self._rerank_candidates(normalized_query, chunks)
                selected_chunks = select_retrieved_chunks(
                    chunks,
                    top_k=resolved_top_k,
                    min_score=self._min_score,
                    max_chunks_per_document=resolved_max_chunks_per_document,
                )
            except Exception as exc:
                self._record_retrieval(
                    outcome="error",
                    selected_chunks=[],
                    duration_seconds=perf_counter() - started_at,
                    query=normalized_query,
                    error=str(exc),
                )
                span.annotate(metadata={"outcome": "error"})
                raise

            duration_seconds = perf_counter() - started_at
            outcome = "hit" if selected_chunks else "miss"
            self._record_retrieval(
                outcome=outcome,
                selected_chunks=selected_chunks,
                duration_seconds=duration_seconds,
                query=normalized_query,
            )
            span.finish_success(
                outputs={
                    "outcome": outcome,
                    "selected_chunk_count": len(selected_chunks),
                    "top_score": None if not selected_chunks else selected_chunks[0].score,
                    "duration_ms": round(duration_seconds * 1000, 6),
                    "rerank_applied": rerank_applied,
                }
            )
            return selected_chunks

    async def _rerank_candidates(
        self,
        query: str,
        chunks: list[RetrievedDocumentChunk],
    ) -> tuple[list[RetrievedDocumentChunk], bool]:
        if not self._rerank_enabled or self._chat_provider is None or len(chunks) <= 1:
            return chunks, False

        span = self._trace_sink.start_span(
            "retrieval.rerank",
            run_type="llm",
            inputs={"query": query, "candidate_count": len(chunks)},
            tags=["retrieval", "rerank"],
        )
        with span:
            try:
                result = await self._chat_provider.generate_response(
                    [
                        ChatTurn(role="system", content=RERANK_SYSTEM_PROMPT),
                        ChatTurn(role="user", content=build_rerank_prompt(query, chunks)),
                    ]
                )
            except ChatProviderError as exc:
                span.annotate(metadata={"outcome": "error", "error": str(exc)})
                return chunks, False

            if isinstance(result, ToolCallBatch):
                span.annotate(metadata={"outcome": "invalid_tool_call"})
                return chunks, False

            reranked = parse_rerank_response(result.content, chunks)
            if reranked is None:
                span.annotate(metadata={"outcome": "parse_error"})
                return chunks, False

            span.finish_success(outputs={"outcome": "reranked"})
            return reranked, True

    async def _embed_query(self, query: str) -> list[float]:
        embeddings = await self._embedding_provider.embed_texts([query])
        if len(embeddings) != 1:
            raise ValueError("embedding provider returned an invalid query embedding")
        return embeddings[0]

    def _record_retrieval(
        self,
        *,
        outcome: str,
        selected_chunks: list[RetrievedDocumentChunk],
        duration_seconds: float,
        query: str,
        error: str | None = None,
    ) -> None:
        if self._observability is None:
            return

        top_score = selected_chunks[0].score if selected_chunks else None
        self._observability.record_retrieval(
            outcome=outcome,
            selected_chunk_count=len(selected_chunks),
            top_score=top_score,
        )
        self._observability.log_event(
            "retrieval.completed" if error is None else "retrieval.failed",
            level="info" if error is None else "error",
            outcome=outcome,
            duration_ms=duration_seconds * 1000,
            query_chars=len(query),
            selected_chunk_count=len(selected_chunks),
            top_score=top_score,
            error=error,
        )


def build_retrieval_prompt(chunks: list[RetrievedDocumentChunk]) -> str:
    sections = [
        "Use the retrieved context below only when it directly supports the user's question.",
        "Do not claim to have used sources that are not present in the retrieved context.",
        "If the retrieved context is partial, weak, or insufficient, answer conservatively.",
        (
            "Do not imply source-backed certainty unless the answer is grounded "
            "in the retrieved context."
        ),
        "",
        "Retrieved context:",
    ]
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            (
                f"[Source {index}] file={chunk.filename} "
                f"document_id={chunk.document_id} chunk={chunk.chunk_index} "
                f"offsets={chunk.start_offset}:{chunk.end_offset}\n"
                f"{chunk.content}"
            )
        )
    return "\n".join(sections)


def build_citation(chunk: RetrievedDocumentChunk) -> ChatCitation:
    snippet = chunk.content.strip()
    if len(snippet) > RETRIEVAL_SNIPPET_MAX_CHARS:
        snippet = f"{snippet[: RETRIEVAL_SNIPPET_MAX_CHARS - 3].rstrip()}..."

    return ChatCitation(
        document_id=chunk.document_id,
        filename=chunk.filename,
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        snippet=snippet,
    )


def select_retrieved_chunks(
    chunks: list[RetrievedDocumentChunk],
    *,
    top_k: int,
    min_score: float,
    max_chunks_per_document: int,
) -> list[RetrievedDocumentChunk]:
    if top_k <= 0 or max_chunks_per_document <= 0:
        return []

    selected: list[RetrievedDocumentChunk] = []
    chunks_per_document: dict[str, int] = {}

    for chunk in chunks:
        if chunk.score < min_score:
            continue

        selected_count = chunks_per_document.get(chunk.document_id, 0)
        if selected_count >= max_chunks_per_document:
            continue

        selected.append(chunk)
        chunks_per_document[chunk.document_id] = selected_count + 1
        if len(selected) >= top_k:
            break

    return selected
