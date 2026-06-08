from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter

from celery import Task
from celery.exceptions import MaxRetriesExceededError

from chatbot_api.celery_app import celery_app
from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.document_embeddings import DocumentEmbeddingService
from chatbot_api.embeddings import (
    EmbeddingProviderError,
    EmbeddingProviderTimeoutError,
    OpenAIEmbeddingProvider,
)
from chatbot_api.observability import get_process_observability
from chatbot_api.repositories import SqlAlchemyDocumentRepository
from chatbot_api.settings import get_settings


async def run_embed_document(document_id: str) -> dict[str, str | int] | None:
    settings = get_settings()
    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            service = DocumentEmbeddingService(
                repository,
                OpenAIEmbeddingProvider(settings),
                batch_size=settings.document_embedding_batch_size,
            )
            result = await service.embed_document(document_id)
            if result is None:
                return None

            return {
                "document_id": result.document_id,
                "status": result.status,
                "updated_chunks": result.updated_chunks,
            }
    finally:
        await engine.dispose()


async def mark_document_failed(document_id: str, failure_reason: str) -> None:
    settings = get_settings()
    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            await repository.mark_document_failed(
                document_id=document_id,
                failure_reason=failure_reason,
            )
    finally:
        await engine.dispose()


def calculate_retry_countdown(*, backoff_seconds: int, retry_count: int) -> int:
    return backoff_seconds * (2**retry_count)


def execute_embed_document_task(
    *,
    document_id: str,
    retry_count: int,
    retry: Callable[..., object],
    settings=None,
    run_job: Callable[[str], object] = run_embed_document,
    mark_failed_job: Callable[[str, str], object] = mark_document_failed,
) -> dict[str, str | int] | None:
    resolved_settings = settings or get_settings()
    observability = get_process_observability(resolved_settings)
    started_at = perf_counter()
    observability.log_event(
        "document.embedding_job.started",
        document_id=document_id,
        retry_count=retry_count,
    )

    try:
        result = asyncio.run(run_job(document_id))
    except (EmbeddingProviderTimeoutError, EmbeddingProviderError) as exc:
        countdown = calculate_retry_countdown(
            backoff_seconds=resolved_settings.document_embedding_task_retry_backoff_seconds,
            retry_count=retry_count,
        )
        observability.log_event(
            "document.embedding_job.retrying",
            level="warning",
            document_id=document_id,
            retry_count=retry_count,
            countdown_seconds=countdown,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        try:
            retry(
                exc=exc,
                countdown=countdown,
                max_retries=resolved_settings.document_embedding_task_max_retries,
            )
        except MaxRetriesExceededError:
            asyncio.run(mark_failed_job(document_id, str(exc)))
            observability.record_document_embedding_job(
                outcome="failed",
                duration_seconds=perf_counter() - started_at,
            )
            observability.log_event(
                "document.embedding_job.failed",
                level="error",
                document_id=document_id,
                retry_count=retry_count,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

        observability.record_document_embedding_job(
            outcome="retry_scheduled",
            duration_seconds=perf_counter() - started_at,
        )
        raise
    except Exception as exc:
        asyncio.run(mark_failed_job(document_id, str(exc)))
        observability.record_document_embedding_job(
            outcome="failed",
            duration_seconds=perf_counter() - started_at,
        )
        observability.log_event(
            "document.embedding_job.failed",
            level="error",
            document_id=document_id,
            retry_count=retry_count,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise

    if result is None:
        observability.record_document_embedding_job(
            outcome="missing_document",
            duration_seconds=perf_counter() - started_at,
        )
        observability.log_event(
            "document.embedding_job.completed",
            level="warning",
            document_id=document_id,
            retry_count=retry_count,
            outcome="missing_document",
        )
        return None

    observability.record_document_embedding_job(
        outcome="success",
        duration_seconds=perf_counter() - started_at,
    )
    observability.log_event(
        "document.embedding_job.completed",
        document_id=document_id,
        retry_count=retry_count,
        outcome="success",
        status=result["status"],
        updated_chunks=result["updated_chunks"],
    )
    return result


@celery_app.task(bind=True, name="chatbot_api.embed_document")
def embed_document_task(
    self: Task,
    document_id: str,
) -> dict[str, str | int] | None:
    return execute_embed_document_task(
        document_id=document_id,
        retry_count=self.request.retries,
        retry=self.retry,
    )
