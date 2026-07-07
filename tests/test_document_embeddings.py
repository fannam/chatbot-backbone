from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from celery.exceptions import MaxRetriesExceededError, Retry
from sqlalchemy.exc import DBAPIError

from chatbot_api.repositories import ChunkEmbeddingUpdate, ChunkWithoutEmbedding
from chatbot_api.retrieval import (
    DocumentChunkCreate,
    DocumentEmbeddingService,
    DocumentRecord,
    EmbeddingProviderError,
)
from chatbot_api.settings import Settings
from chatbot_api.tasks.embedding_jobs import calculate_retry_countdown, execute_embed_document_task


@dataclass
class StoredDocument:
    record: DocumentRecord
    chunks: list[DocumentChunkCreate]


class InMemoryEmbeddingRepository:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.documents = {
            "doc-1": StoredDocument(
                record=DocumentRecord(
                    id="doc-1",
                    filename="doc.md",
                    content_type="text/markdown",
                    byte_size=100,
                    checksum_sha256="hash-1",
                    status="processing",
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                ),
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="first chunk",
                        start_offset=0,
                        end_offset=11,
                        embedding=None,
                    ),
                    DocumentChunkCreate(
                        chunk_index=1,
                        content="second chunk",
                        start_offset=12,
                        end_offset=24,
                        embedding=None,
                    ),
                ],
            ),
            "doc-failed": StoredDocument(
                record=DocumentRecord(
                    id="doc-failed",
                    filename="failed.md",
                    content_type="text/markdown",
                    byte_size=50,
                    checksum_sha256="hash-failed",
                    status="failed",
                    failure_reason="old failure",
                    created_at=now,
                    updated_at=now,
                ),
                chunks=[],
            ),
        }

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        stored = self.documents.get(document_id)
        return stored.record if stored is not None else None

    async def list_chunks_missing_embeddings(
        self,
        *,
        document_id: str | None = None,
        limit: int,
    ) -> list[ChunkWithoutEmbedding]:
        if document_id is None:
            return []

        stored = self.documents[document_id]
        missing = [
            ChunkWithoutEmbedding(
                document_id=document_id,
                chunk_id=index + 1,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                metadata=chunk.metadata,
            )
            for index, chunk in enumerate(stored.chunks)
            if chunk.embedding is None
        ]
        return missing[:limit]

    async def update_chunk_embeddings(
        self,
        *,
        updates: list[ChunkEmbeddingUpdate],
    ) -> int:
        stored = self.documents["doc-1"]
        for update in updates:
            chunk = stored.chunks[update.chunk_id - 1]
            stored.chunks[update.chunk_id - 1] = DocumentChunkCreate(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                metadata=chunk.metadata,
                embedding=update.embedding,
            )
        return len(updates)

    async def mark_document_ready(self, document_id: str) -> DocumentRecord | None:
        stored = self.documents[document_id]
        updated = DocumentRecord(
            id=stored.record.id,
            filename=stored.record.filename,
            content_type=stored.record.content_type,
            byte_size=stored.record.byte_size,
            checksum_sha256=stored.record.checksum_sha256,
            status="ready",
            failure_reason=None,
            created_at=stored.record.created_at,
            updated_at=datetime.now(UTC),
        )
        self.documents[document_id] = StoredDocument(record=updated, chunks=stored.chunks)
        return updated

    async def mark_document_failed(
        self,
        *,
        document_id: str,
        failure_reason: str,
    ) -> DocumentRecord | None:
        stored = self.documents[document_id]
        updated = DocumentRecord(
            id=stored.record.id,
            filename=stored.record.filename,
            content_type=stored.record.content_type,
            byte_size=stored.record.byte_size,
            checksum_sha256=stored.record.checksum_sha256,
            status="failed",
            failure_reason=failure_reason,
            created_at=stored.record.created_at,
            updated_at=datetime.now(UTC),
        )
        self.documents[document_id] = StoredDocument(record=updated, chunks=stored.chunks)
        return updated


class StubEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


@pytest.mark.anyio
async def test_document_embedding_service_updates_missing_chunks_and_marks_ready() -> None:
    repository = InMemoryEmbeddingRepository()
    provider = StubEmbeddingProvider()
    service = DocumentEmbeddingService(repository, provider, batch_size=1)

    result = await service.embed_document("doc-1")

    assert result is not None
    assert result.status == "ready"
    assert result.updated_chunks == 2
    assert repository.documents["doc-1"].record.status == "ready"
    assert repository.documents["doc-1"].chunks[0].embedding == [1.0, 11.0]
    assert repository.documents["doc-1"].chunks[1].embedding == [1.0, 12.0]


@pytest.mark.anyio
async def test_document_embedding_service_noops_for_failed_document() -> None:
    repository = InMemoryEmbeddingRepository()
    provider = StubEmbeddingProvider()
    service = DocumentEmbeddingService(repository, provider, batch_size=2)

    result = await service.embed_document("doc-failed")

    assert result is not None
    assert result.status == "failed"
    assert result.updated_chunks == 0
    assert provider.calls == []


def test_execute_embed_document_task_retries_recoverable_errors() -> None:
    captured: dict[str, object] = {}

    async def failing_job(document_id: str):
        raise EmbeddingProviderError(f"temporary failure for {document_id}")

    async def mark_failed(document_id: str, failure_reason: str) -> None:
        captured["marked_failed"] = (document_id, failure_reason)

    def retry(**kwargs):
        # Real `Task.retry()` never returns normally: it always raises, either
        # `Retry` when a retry was scheduled or `MaxRetriesExceededError` once
        # exhausted. Mirror that here instead of silently returning.
        captured["retry"] = kwargs
        raise Retry()

    settings = Settings(
        document_embedding_task_max_retries=4,
        document_embedding_task_retry_backoff_seconds=15,
    )

    with pytest.raises(Retry):
        execute_embed_document_task(
            document_id="doc-1",
            retry_count=2,
            retry=retry,
            settings=settings,
            run_job=failing_job,
            mark_failed_job=mark_failed,
        )

    assert isinstance(captured["retry"]["exc"], EmbeddingProviderError)
    assert captured["retry"]["countdown"] == 60
    assert captured["retry"]["max_retries"] == 4
    assert "marked_failed" not in captured


def test_execute_embed_document_task_retries_transient_db_errors() -> None:
    captured: dict[str, object] = {}

    async def failing_job(document_id: str):
        raise DBAPIError("connection reset", params=None, orig=Exception("reset"))

    async def mark_failed(document_id: str, failure_reason: str) -> None:
        captured["marked_failed"] = (document_id, failure_reason)

    def retry(**kwargs):
        captured["retry"] = kwargs
        raise Retry()

    settings = Settings(
        document_embedding_task_max_retries=4,
        document_embedding_task_retry_backoff_seconds=15,
    )

    with pytest.raises(Retry):
        execute_embed_document_task(
            document_id="doc-1",
            retry_count=0,
            retry=retry,
            settings=settings,
            run_job=failing_job,
            mark_failed_job=mark_failed,
        )

    assert isinstance(captured["retry"]["exc"], DBAPIError)
    assert "marked_failed" not in captured


def test_execute_embed_document_task_marks_failed_after_max_retries() -> None:
    captured: dict[str, object] = {}

    async def failing_job(document_id: str):
        raise EmbeddingProviderError(f"temporary failure for {document_id}")

    async def mark_failed(document_id: str, failure_reason: str) -> None:
        captured["marked_failed"] = (document_id, failure_reason)

    def retry(**kwargs):
        raise MaxRetriesExceededError()

    settings = Settings(
        document_embedding_task_max_retries=2,
        document_embedding_task_retry_backoff_seconds=10,
    )

    with pytest.raises(MaxRetriesExceededError):
        execute_embed_document_task(
            document_id="doc-9",
            retry_count=2,
            retry=retry,
            settings=settings,
            run_job=failing_job,
            mark_failed_job=mark_failed,
        )

    assert captured["marked_failed"] == ("doc-9", "temporary failure for doc-9")


def test_calculate_retry_countdown_uses_exponential_backoff() -> None:
    assert calculate_retry_countdown(backoff_seconds=30, retry_count=0) == 30
    assert calculate_retry_countdown(backoff_seconds=30, retry_count=2) == 120
