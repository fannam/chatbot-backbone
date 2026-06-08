from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from chatbot_api.database import Base
from chatbot_api.document_ingestion import DocumentChunkCreate
from chatbot_api.models import Conversation, Document, DocumentChunk, Message, ToolRun
from chatbot_api.providers import ChatTurn
from chatbot_api.repositories import (
    ChunkEmbeddingUpdate,
    SqlAlchemyChatRepository,
    SqlAlchemyDocumentRepository,
)


class SyncAsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(self, statement):
        return self._session.execute(statement)

    async def get(self, entity, ident):
        return self._session.get(entity, ident)

    def add(self, instance) -> None:
        self._session.add(instance)

    def add_all(self, instances) -> None:
        self._session.add_all(instances)

    async def commit(self) -> None:
        self._session.commit()

    async def rollback(self) -> None:
        self._session.rollback()


@pytest.fixture
def session_factory(tmp_path: Path) -> sessionmaker[Session]:
    engine = create_engine(f"sqlite:///{tmp_path / 'chatbot.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)

    try:
        yield factory
    finally:
        engine.dispose()


@pytest.mark.anyio
async def test_repository_returns_messages_in_conversation_order(
    session_factory: sessionmaker[Session],
) -> None:
    write_session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(write_session))
        await repository.append_exchange(
            conversation_id="conv-123",
            user_message="First user",
            user_metadata={"source": "test-1"},
            assistant_message="First assistant",
        )
        await repository.append_exchange(
            conversation_id="conv-123",
            user_message="Second user",
            user_metadata={"source": "test-2"},
            assistant_message="Second assistant",
        )
    finally:
        write_session.close()

    read_session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(read_session))
        history = await repository.list_messages("conv-123")
        messages = (
            read_session.execute(
                select(Message)
                .where(Message.conversation_id == "conv-123")
                .order_by(Message.id)
            )
        ).scalars().all()
    finally:
        read_session.close()

    assert history == [
        ChatTurn(role="user", content="First user"),
        ChatTurn(role="assistant", content="First assistant"),
        ChatTurn(role="user", content="Second user"),
        ChatTurn(role="assistant", content="Second assistant"),
    ]
    assert messages[0].metadata_ == {"source": "test-1"}
    assert messages[1].metadata_ is None
    assert messages[2].metadata_ == {"source": "test-2"}
    assert messages[3].metadata_ is None


@pytest.mark.anyio
async def test_repository_updates_conversation_timestamp_on_new_exchange(
    session_factory: sessionmaker[Session],
) -> None:
    first_session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(first_session))
        await repository.append_exchange(
            conversation_id="conv-456",
            user_message="Opening message",
            user_metadata=None,
            assistant_message="Opening reply",
        )
        initial_conversation = first_session.get(Conversation, "conv-456")
    finally:
        first_session.close()

    assert initial_conversation is not None
    initial_created_at = initial_conversation.created_at
    initial_updated_at = initial_conversation.updated_at

    time.sleep(0.01)

    second_session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(second_session))
        await repository.append_exchange(
            conversation_id="conv-456",
            user_message="Follow-up",
            user_metadata=None,
            assistant_message="Follow-up reply",
        )
        updated_conversation = second_session.get(Conversation, "conv-456")
    finally:
        second_session.close()

    assert updated_conversation is not None
    assert updated_conversation.created_at == initial_created_at
    assert updated_conversation.updated_at > initial_updated_at


@pytest.mark.anyio
async def test_repository_persists_tool_run_lifecycle(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        created = await repository.create_tool_run(
            conversation_id="conv-tools",
            tool_call_id="tool-1",
            tool_name="calculator",
            input_payload={"expression": "2 + 2"},
        )
        completed = await repository.complete_tool_run(
            conversation_id="conv-tools",
            tool_call_id="tool-1",
            output_payload={"result": 4},
        )
        row = (
            session.execute(
                select(ToolRun)
                .where(ToolRun.conversation_id == "conv-tools")
                .where(ToolRun.tool_call_id == "tool-1")
            )
        ).scalar_one()
    finally:
        session.close()

    assert created.status == "running"
    assert created.output_payload is None
    assert completed is not None
    assert completed.status == "completed"
    assert completed.output_payload == {"result": 4}
    assert row.status == "completed"
    assert row.output_payload == {"result": 4}
    assert row.completed_at is not None


@pytest.mark.anyio
async def test_repository_reports_conversation_existence(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        await repository.append_exchange(
            conversation_id="conv-existing",
            user_message="Hello",
            user_metadata=None,
            assistant_message="Hi",
        )
        exists = await repository.conversation_exists("conv-existing")
        missing = await repository.conversation_exists("conv-missing")
    finally:
        session.close()

    assert exists is True
    assert missing is False


@pytest.mark.anyio
async def test_repository_lists_tool_runs_newest_first_with_limit(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        await repository.create_tool_run(
            conversation_id="conv-tools-list",
            tool_call_id="tool-1",
            tool_name="calculator",
            input_payload={"expression": "1 + 1"},
        )
        await repository.complete_tool_run(
            conversation_id="conv-tools-list",
            tool_call_id="tool-1",
            output_payload={"result": 2},
        )
        await repository.create_tool_run(
            conversation_id="conv-tools-list",
            tool_call_id="tool-2",
            tool_name="search_knowledge_base",
            input_payload={"query": "guide"},
        )
        await repository.fail_tool_run(
            conversation_id="conv-tools-list",
            tool_call_id="tool-2",
            status="failed",
            error_message="search failed",
        )

        tool_runs = await repository.list_tool_runs("conv-tools-list", limit=1)
    finally:
        session.close()

    assert len(tool_runs) == 1
    assert tool_runs[0].tool_call_id == "tool-2"
    assert tool_runs[0].tool_name == "search_knowledge_base"
    assert tool_runs[0].status == "failed"
    assert tool_runs[0].output_payload is None
    assert tool_runs[0].error_message == "search failed"
    assert tool_runs[0].completed_at is not None


@pytest.mark.anyio
async def test_document_repository_persists_document_and_chunks_in_order(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyDocumentRepository(SyncAsyncSessionAdapter(session))
        created = await repository.create_document(
            document_id="doc-123",
            filename="notes.md",
            content_type="text/markdown",
            byte_size=123,
            checksum_sha256="abc123",
            status="ready",
            failure_reason=None,
            chunks=[
                DocumentChunkCreate(
                    chunk_index=0,
                    content="First chunk",
                    embedding=[1.0, 0.0],
                    start_offset=0,
                    end_offset=11,
                    metadata={"page": 1},
                ),
                DocumentChunkCreate(
                    chunk_index=1,
                    content="Second chunk",
                    embedding=[0.0, 1.0],
                    start_offset=10,
                    end_offset=22,
                    metadata=None,
                ),
            ],
        )
        chunks = await repository.list_document_chunks("doc-123")
        document = session.get(Document, "doc-123")
        rows = (
            session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == "doc-123")
                .order_by(DocumentChunk.chunk_index)
            )
        ).scalars().all()
    finally:
        session.close()

    assert created.id == "doc-123"
    assert created.filename == "notes.md"
    assert created.content_type == "text/markdown"
    assert document is not None
    assert document.status == "ready"
    assert document.failure_reason is None
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].embedding == [1.0, 0.0]
    assert chunks[0].metadata == {"page": 1}
    assert [row.content for row in rows] == ["First chunk", "Second chunk"]


@pytest.mark.anyio
async def test_document_repository_searches_ready_chunks_by_similarity(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyDocumentRepository(SyncAsyncSessionAdapter(session))
        await repository.create_document(
            document_id="doc-1",
            filename="guide.md",
            content_type="text/markdown",
            byte_size=100,
            checksum_sha256="hash-1",
            status="ready",
            failure_reason=None,
            chunks=[
                DocumentChunkCreate(
                    chunk_index=0,
                    content="alpha chunk",
                    embedding=[1.0, 0.0],
                    start_offset=0,
                    end_offset=11,
                ),
                DocumentChunkCreate(
                    chunk_index=1,
                    content="beta chunk",
                    embedding=[0.0, 1.0],
                    start_offset=12,
                    end_offset=22,
                ),
            ],
        )
        await repository.create_document(
            document_id="doc-2",
            filename="draft.md",
            content_type="text/markdown",
            byte_size=100,
            checksum_sha256="hash-2",
            status="failed",
            failure_reason="embedding failed",
            chunks=[
                DocumentChunkCreate(
                    chunk_index=0,
                    content="ignored chunk",
                    embedding=[1.0, 1.0],
                    start_offset=0,
                    end_offset=13,
                )
            ],
        )
        results = await repository.search_similar_chunks(
            query_embedding=[0.9, 0.1],
            limit=2,
        )
    finally:
        session.close()

    assert [result.document_id for result in results] == ["doc-1", "doc-1"]
    assert [result.chunk_index for result in results] == [0, 1]
    assert results[0].filename == "guide.md"
    assert results[0].score > results[1].score


@pytest.mark.anyio
async def test_document_repository_updates_status_and_lists_missing_embeddings(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyDocumentRepository(SyncAsyncSessionAdapter(session))
        await repository.create_document(
            document_id="doc-processing",
            filename="processing.md",
            content_type="text/markdown",
            byte_size=50,
            checksum_sha256="hash-processing",
            status="processing",
            failure_reason=None,
            chunks=[
                DocumentChunkCreate(
                    chunk_index=0,
                    content="missing embedding",
                    start_offset=0,
                    end_offset=17,
                )
            ],
        )
        await repository.create_document(
            document_id="doc-ready",
            filename="ready.md",
            content_type="text/markdown",
            byte_size=50,
            checksum_sha256="hash-ready",
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
        missing_document_ids = await repository.list_documents_missing_embeddings(limit=10)
        missing_chunks = await repository.list_chunks_missing_embeddings(
            document_id="doc-processing",
            limit=10,
        )
        failed = await repository.mark_document_failed(
            document_id="doc-processing",
            failure_reason="queue failed",
        )
        fetched = await repository.get_document("doc-processing")
    finally:
        session.close()

    assert missing_document_ids == ["doc-processing"]
    assert [chunk.document_id for chunk in missing_chunks] == ["doc-processing"]
    assert failed is not None
    assert failed.status == "failed"
    assert failed.failure_reason == "queue failed"
    assert fetched is not None
    assert fetched.failure_reason == "queue failed"


@pytest.mark.anyio
async def test_document_repository_lists_and_updates_missing_embeddings(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyDocumentRepository(SyncAsyncSessionAdapter(session))
        await repository.create_document(
            document_id="doc-3",
            filename="notes.md",
            content_type="text/markdown",
            byte_size=100,
            checksum_sha256="hash-3",
            status="ready",
            failure_reason=None,
            chunks=[
                DocumentChunkCreate(
                    chunk_index=0,
                    content="missing embedding",
                    start_offset=0,
                    end_offset=17,
                )
            ],
        )
        missing = await repository.list_chunks_missing_embeddings(limit=10)
        updated = await repository.update_chunk_embeddings(
            updates=[ChunkEmbeddingUpdate(chunk_id=missing[0].chunk_id, embedding=[0.5, 0.5])]
        )
        chunks = await repository.list_document_chunks("doc-3")
    finally:
        session.close()

    assert len(missing) == 1
    assert missing[0].content == "missing embedding"
    assert updated == 1
    assert chunks[0].embedding == [0.5, 0.5]
