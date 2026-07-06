from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from chatbot_api.database import Base
from chatbot_api.document_ingestion import DocumentChunkCreate
from chatbot_api.models import (
    ApiKey,
    Conversation,
    ConversationSummary,
    Document,
    DocumentChunk,
    Memory,
    Message,
    ToolRun,
    User,
    utcnow,
)
from chatbot_api.providers import ChatTurn
from chatbot_api.repositories import (
    ChunkEmbeddingUpdate,
    OwnershipError,
    SqlAlchemyAuthRepository,
    SqlAlchemyChatRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyMemoryRepository,
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
async def test_repository_scopes_conversation_access_by_owner_user_id(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add(
            User(
                id="user-123",
                display_name="Alice",
                email=None,
                plan=None,
                locale=None,
                preferences_json={},
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.commit()
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        await repository.append_exchange(
            conversation_id="conv-owned",
            user_message="Hello",
            user_metadata=None,
            assistant_message="Hi",
            owner_user_id="user-123",
        )
        owned_exists = await repository.conversation_exists(
            "conv-owned",
            owner_user_id="user-123",
        )
        foreign_exists = await repository.conversation_exists(
            "conv-owned",
            owner_user_id="user-999",
        )
    finally:
        session.close()

    assert owned_exists is True
    assert foreign_exists is False


@pytest.mark.anyio
async def test_repository_rejects_appending_to_foreign_owned_conversation(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add_all(
            [
                User(
                    id="user-123",
                    display_name="Alice",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
                User(
                    id="user-999",
                    display_name="Mallory",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
            ]
        )
        session.commit()
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        await repository.append_exchange(
            conversation_id="conv-owned",
            user_message="Hello",
            user_metadata=None,
            assistant_message="Hi",
            owner_user_id="user-123",
        )
        with pytest.raises(OwnershipError):
            await repository.append_exchange(
                conversation_id="conv-owned",
                user_message="Intrude",
                user_metadata=None,
                assistant_message="Nope",
                owner_user_id="user-999",
            )
    finally:
        session.close()


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
async def test_repository_returns_message_records_and_persisted_exchange_ids(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        persisted_exchange = await repository.append_exchange(
            conversation_id="conv-records",
            user_message="Remember this",
            user_metadata={"source": "memory-test"},
            assistant_message="Stored",
        )
        records = await repository.list_message_records("conv-records")
    finally:
        session.close()

    assert persisted_exchange.conversation_id == "conv-records"
    assert persisted_exchange.user_message_id < persisted_exchange.assistant_message_id
    assert [record.id for record in records] == [
        persisted_exchange.user_message_id,
        persisted_exchange.assistant_message_id,
    ]
    assert records[0].metadata == {"source": "memory-test"}
    assert records[1].metadata is None


@pytest.mark.anyio
async def test_memory_repository_upserts_and_reads_conversation_summary(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        chat_repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        memory_repository = SqlAlchemyMemoryRepository(SyncAsyncSessionAdapter(session))
        persisted_exchange = await chat_repository.append_exchange(
            conversation_id="conv-summary",
            user_message="Opening",
            user_metadata=None,
            assistant_message="Reply",
        )
        created_summary = await memory_repository.upsert_conversation_summary(
            conversation_id="conv-summary",
            summary_text="Initial summary",
            last_summarized_message_id=persisted_exchange.assistant_message_id,
        )
        updated_summary = await memory_repository.upsert_conversation_summary(
            conversation_id="conv-summary",
            summary_text="Updated summary",
            last_summarized_message_id=persisted_exchange.assistant_message_id,
        )
        fetched_summary = await memory_repository.get_conversation_summary("conv-summary")
        summary_row = session.get(ConversationSummary, "conv-summary")
    finally:
        session.close()

    assert created_summary.conversation_id == "conv-summary"
    assert updated_summary.summary_text == "Updated summary"
    assert fetched_summary is not None
    assert fetched_summary.summary_text == "Updated summary"
    assert summary_row is not None
    assert summary_row.last_summarized_message_id == persisted_exchange.assistant_message_id


@pytest.mark.anyio
async def test_memory_repository_lists_active_memories_and_soft_deletes(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add(
            User(
                id="user-123",
                display_name="Alice",
                email=None,
                plan=None,
                locale=None,
                preferences_json={},
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.commit()
        chat_repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        memory_repository = SqlAlchemyMemoryRepository(SyncAsyncSessionAdapter(session))
        persisted_exchange = await chat_repository.append_exchange(
            conversation_id="conv-memory",
            user_message="I work at Example",
            user_metadata={"user_profile": {"user_id": "user-123"}},
            assistant_message="Noted",
        )
        created_memory = await memory_repository.upsert_memory(
            user_id="user-123",
            kind="profile",
            key="profile.company",
            value_json={"value": "Example"},
            confidence=0.7,
            source_message_id=persisted_exchange.user_message_id,
            extraction_method="llm",
        )
        updated_memory = await memory_repository.upsert_memory(
            user_id="user-123",
            kind="profile",
            key="profile.company",
            value_json={"value": "Example Inc"},
            confidence=0.9,
            source_message_id=persisted_exchange.user_message_id,
            extraction_method="llm",
        )
        listed_before_delete = await memory_repository.list_active_memories("user-123", limit=8)
        deleted = await memory_repository.delete_memory(
            user_id="user-123",
            memory_id=created_memory.id,
        )
        listed_after_delete = await memory_repository.list_active_memories("user-123", limit=8)
        deleted_row = session.get(Memory, created_memory.id)
    finally:
        session.close()

    assert created_memory.id == updated_memory.id
    assert listed_before_delete[0].value_json == {"value": "Example Inc"}
    assert deleted is True
    assert listed_after_delete == []
    assert deleted_row is not None
    assert deleted_row.deleted_at is not None


@pytest.mark.anyio
async def test_memory_repository_scopes_reads_by_owner_user_id(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add_all(
            [
                User(
                    id="user-123",
                    display_name="Alice",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
                User(
                    id="user-999",
                    display_name="Mallory",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
            ]
        )
        session.commit()
        chat_repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        memory_repository = SqlAlchemyMemoryRepository(SyncAsyncSessionAdapter(session))
        persisted_exchange = await chat_repository.append_exchange(
            conversation_id="conv-owned-memory",
            user_message="I work at Example",
            user_metadata={"user_profile": {"user_id": "user-123"}},
            assistant_message="Noted",
            owner_user_id="user-123",
        )
        await memory_repository.upsert_conversation_summary(
            conversation_id="conv-owned-memory",
            summary_text="Summary",
            last_summarized_message_id=persisted_exchange.assistant_message_id,
            owner_user_id="user-123",
        )
        await memory_repository.upsert_memory(
            user_id="user-123",
            kind="profile",
            key="profile.company",
            value_json={"value": "Example"},
            confidence=0.7,
            source_message_id=persisted_exchange.user_message_id,
            extraction_method="llm",
            owner_user_id="user-123",
        )

        owned_summary = await memory_repository.get_conversation_summary(
            "conv-owned-memory",
            owner_user_id="user-123",
        )
        foreign_summary = await memory_repository.get_conversation_summary(
            "conv-owned-memory",
            owner_user_id="user-999",
        )
        owned_memories = await memory_repository.list_active_memories(
            "user-123",
            limit=8,
            owner_user_id="user-123",
        )
        foreign_memories = await memory_repository.list_active_memories(
            "user-123",
            limit=8,
            owner_user_id="user-999",
        )
        foreign_delete = await memory_repository.delete_memory(
            user_id="user-123",
            memory_id=1,
            owner_user_id="user-999",
        )
    finally:
        session.close()

    assert owned_summary is not None
    assert foreign_summary is None
    assert len(owned_memories) == 1
    assert foreign_memories == []
    assert foreign_delete is False


@pytest.mark.anyio
async def test_memory_repository_rejects_writes_to_foreign_owned_conversation(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add_all(
            [
                User(
                    id="user-123",
                    display_name="Alice",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
                User(
                    id="user-999",
                    display_name="Mallory",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
            ]
        )
        session.commit()
        chat_repository = SqlAlchemyChatRepository(SyncAsyncSessionAdapter(session))
        memory_repository = SqlAlchemyMemoryRepository(SyncAsyncSessionAdapter(session))
        persisted_exchange = await chat_repository.append_exchange(
            conversation_id="conv-owned-memory-write",
            user_message="Hello",
            user_metadata=None,
            assistant_message="Hi",
            owner_user_id="user-123",
        )
        with pytest.raises(OwnershipError):
            await memory_repository.upsert_conversation_summary(
                conversation_id="conv-owned-memory-write",
                summary_text="Intrusion",
                last_summarized_message_id=persisted_exchange.assistant_message_id,
                owner_user_id="user-999",
            )
        with pytest.raises(OwnershipError):
            await memory_repository.upsert_memory(
                user_id="user-999",
                kind="profile",
                key="profile.company",
                value_json={"value": "Intrusion"},
                confidence=0.7,
                source_message_id=persisted_exchange.user_message_id,
                extraction_method="llm",
                owner_user_id="user-123",
            )
    finally:
        session.close()


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
async def test_document_repository_filters_document_reads_and_search_by_owner(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add_all(
            [
                User(
                    id="user-123",
                    display_name="Alice",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
                User(
                    id="user-999",
                    display_name="Mallory",
                    email=None,
                    plan=None,
                    locale=None,
                    preferences_json={},
                    created_at=timestamp,
                    updated_at=timestamp,
                ),
            ]
        )
        session.commit()
        repository = SqlAlchemyDocumentRepository(SyncAsyncSessionAdapter(session))
        await repository.create_document(
            document_id="doc-owned",
            filename="owned.md",
            content_type="text/markdown",
            byte_size=100,
            checksum_sha256="hash-owned",
            status="ready",
            failure_reason=None,
            chunks=[
                DocumentChunkCreate(
                    chunk_index=0,
                    content="owned chunk",
                    embedding=[1.0, 0.0],
                    start_offset=0,
                    end_offset=11,
                )
            ],
            owner_user_id="user-123",
        )
        visible_document = await repository.get_document(
            "doc-owned",
            owner_user_id="user-123",
        )
        hidden_document = await repository.get_document(
            "doc-owned",
            owner_user_id="user-999",
        )
        visible_results = await repository.search_similar_chunks(
            query_embedding=[1.0, 0.0],
            limit=5,
            owner_user_id="user-123",
        )
        hidden_results = await repository.search_similar_chunks(
            query_embedding=[1.0, 0.0],
            limit=5,
            owner_user_id="user-999",
        )
    finally:
        session.close()

    assert visible_document is not None
    assert hidden_document is None
    assert [result.document_id for result in visible_results] == ["doc-owned"]
    assert hidden_results == []


@pytest.mark.anyio
async def test_document_repository_find_by_checksum_respects_owner_scoping(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        timestamp = utcnow()
        session.add(
            User(
                id="user-123",
                display_name="Alice",
                email=None,
                plan=None,
                locale=None,
                preferences_json={},
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.commit()
        repository = SqlAlchemyDocumentRepository(SyncAsyncSessionAdapter(session))
        await repository.create_document(
            document_id="doc-owned",
            filename="owned.md",
            content_type="text/markdown",
            byte_size=100,
            checksum_sha256="hash-shared",
            status="ready",
            failure_reason=None,
            chunks=[],
            owner_user_id="user-123",
        )

        same_owner_match = await repository.find_document_by_checksum(
            "hash-shared", owner_user_id="user-123"
        )
        different_owner_match = await repository.find_document_by_checksum(
            "hash-shared", owner_user_id="user-999"
        )
        no_owner_filter_match = await repository.find_document_by_checksum("hash-shared")
        no_match = await repository.find_document_by_checksum("hash-missing")
    finally:
        session.close()

    assert same_owner_match is not None
    assert same_owner_match.id == "doc-owned"
    assert different_owner_match is None
    assert no_owner_filter_match is not None
    assert no_owner_filter_match.id == "doc-owned"
    assert no_match is None


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


@pytest.mark.anyio
async def test_auth_repository_creates_and_authenticates_api_keys(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    try:
        repository = SqlAlchemyAuthRepository(SyncAsyncSessionAdapter(session))
        user = await repository.upsert_user(
            user_id="user-123",
            display_name="Alice",
            email="alice@example.com",
            plan="pro",
            locale="en-US",
            preferences_json={"timezone": "UTC"},
        )
        created = await repository.create_api_key(user_id=user.id, name="dev")
        authenticated = await repository.authenticate_api_key(created.api_key)
        api_key_row = session.execute(select(ApiKey)).scalar_one()
        user_row = session.get(User, "user-123")
    finally:
        session.close()

    assert authenticated is not None
    assert authenticated.user_id == "user-123"
    assert authenticated.display_name == "Alice"
    assert authenticated.preferences == {"timezone": "UTC"}
    assert api_key_row.key_prefix == created.key_prefix
    assert api_key_row.last_used_at is not None
    assert user_row is not None
    assert user_row.email == "alice@example.com"
