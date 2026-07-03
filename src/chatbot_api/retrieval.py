from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from chatbot_api.embeddings import EmbeddingProvider
from chatbot_api.observability import ObservabilityService
from chatbot_api.providers import ChatCitation, ChatCompletionMetadata
from chatbot_api.repositories import DocumentRepository, RetrievedDocumentChunk
from chatbot_api.tracing import NoopTraceSink, TraceSink

RETRIEVAL_SNIPPET_MAX_CHARS = 240


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
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._top_k = top_k
        self._min_score = min_score
        self._max_chunks_per_document = max_chunks_per_document
        self._candidate_limit = candidate_limit
        self._observability = observability
        self._trace_sink = trace_sink or NoopTraceSink()

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
                }
            )
            return selected_chunks

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
