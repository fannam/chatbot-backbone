from __future__ import annotations

from pathlib import Path

import pytest

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.models import Base
from chatbot_api.repositories import SqlAlchemyDocumentRepository
from chatbot_api.retrieval import DocumentChunkCreate
from chatbot_api.settings import get_settings
from chatbot_api.tasks.reindex_embeddings import enqueue_documents_missing_embeddings


class StubDocumentTaskQueue:
    def __init__(self) -> None:
        self.enqueued_document_ids: list[str] = []

    def enqueue_embed_document(self, document_id: str) -> None:
        self.enqueued_document_ids.append(document_id)


@pytest.mark.anyio
async def test_enqueue_documents_missing_embeddings_enqueues_each_document_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "reindex.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", "")
    monkeypatch.setenv("DOCUMENT_EMBEDDING_BATCH_SIZE", "1")
    get_settings.cache_clear()

    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            await repository.create_document(
                document_id="doc-a",
                filename="a.md",
                content_type="text/markdown",
                byte_size=100,
                checksum_sha256="hash-a",
                status="processing",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="needs embedding",
                        start_offset=0,
                        end_offset=15,
                    )
                ],
            )
            await repository.create_document(
                document_id="doc-b",
                filename="b.md",
                content_type="text/markdown",
                byte_size=100,
                checksum_sha256="hash-b",
                status="ready",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="needs embedding too",
                        start_offset=0,
                        end_offset=19,
                    )
                ],
            )
            await repository.create_document(
                document_id="doc-c",
                filename="c.md",
                content_type="text/markdown",
                byte_size=100,
                checksum_sha256="hash-c",
                status="ready",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="already embedded",
                        embedding=[1.0, 0.0],
                        start_offset=0,
                        end_offset=15,
                    )
                ],
            )

        task_queue = StubDocumentTaskQueue()
        enqueued = await enqueue_documents_missing_embeddings(task_queue)
    finally:
        get_settings.cache_clear()
        await engine.dispose()

    assert enqueued == 2
    assert task_queue.enqueued_document_ids == ["doc-a", "doc-b"]
