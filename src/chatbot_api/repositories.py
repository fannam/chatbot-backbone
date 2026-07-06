from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

from sqlalchemy import Select, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_api.auth import AuthenticatedUser, build_api_key_prefix, generate_api_key, hash_api_key
from chatbot_api.document_ingestion import DocumentChunkCreate, DocumentRecord
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


class OwnershipError(Exception):
    """Raised when a record does not belong to the expected owner."""


class ChatRepository(Protocol):
    async def list_messages(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[ChatTurn]: ...

    async def list_message_records(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list["MessageRecord"]: ...

    async def conversation_exists(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> bool: ...

    async def list_tool_runs(
        self,
        conversation_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ) -> list["ToolRunRecord"]: ...

    async def create_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        input_payload: dict[str, Any],
        owner_user_id: str | None = None,
    ) -> "ToolRunRecord": ...

    async def complete_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        output_payload: dict[str, Any],
        owner_user_id: str | None = None,
    ) -> "ToolRunRecord | None": ...

    async def fail_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        error_message: str,
        owner_user_id: str | None = None,
    ) -> "ToolRunRecord | None": ...

    async def append_exchange(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_metadata: dict[str, Any] | None,
        assistant_message: str,
        owner_user_id: str | None = None,
    ) -> "PersistedExchange": ...


class MemoryRepository(Protocol):
    async def get_conversation_summary(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> "ConversationSummaryRecord | None": ...

    async def upsert_conversation_summary(
        self,
        *,
        conversation_id: str,
        summary_text: str,
        last_summarized_message_id: int,
        owner_user_id: str | None = None,
    ) -> "ConversationSummaryRecord": ...

    async def list_active_memories(
        self,
        user_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ) -> list["MemoryRecord"]: ...

    async def upsert_memory(
        self,
        *,
        user_id: str,
        kind: Literal["profile", "preference"],
        key: str,
        value_json: dict[str, Any],
        confidence: float,
        source_message_id: int,
        extraction_method: Literal["rule", "llm"],
        owner_user_id: str | None = None,
    ) -> "MemoryRecord": ...

    async def delete_memory(
        self,
        *,
        user_id: str,
        memory_id: int,
        owner_user_id: str | None = None,
    ) -> bool: ...


class DocumentChunkRecord(Protocol):
    id: int
    chunk_index: int
    content: str
    embedding: list[float] | None
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class RetrievedDocumentChunk:
    document_id: str
    filename: str
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] | None
    score: float


@dataclass(frozen=True)
class ChunkEmbeddingUpdate:
    chunk_id: int
    embedding: list[float]


@dataclass(frozen=True)
class ChunkWithoutEmbedding:
    document_id: str
    chunk_id: int
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class ToolRunRecord:
    id: int
    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None
    error_message: str | None
    started_at: Any
    completed_at: Any | None


@dataclass(frozen=True)
class MessageRecord:
    id: int
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] | None
    created_at: datetime


@dataclass(frozen=True)
class PersistedExchange:
    conversation_id: str
    user_message_id: int
    assistant_message_id: int
    created_at: datetime


@dataclass(frozen=True)
class ConversationSummaryRecord:
    conversation_id: str
    summary_text: str
    last_summarized_message_id: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    user_id: str
    kind: str
    key: str
    value_json: dict[str, Any]
    confidence: float
    source_message_id: int
    extraction_method: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


class DocumentRepository(Protocol):
    async def create_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        byte_size: int,
        checksum_sha256: str,
        status: str,
        failure_reason: str | None = None,
        chunks: list[DocumentChunkCreate],
        owner_user_id: str | None = None,
    ) -> DocumentRecord: ...

    async def get_document(
        self,
        document_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> DocumentRecord | None: ...

    async def list_document_chunks(self, document_id: str) -> list[DocumentChunkCreate]: ...

    async def count_document_chunks(
        self,
        document_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> int: ...

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]: ...

    async def list_chunks_missing_embeddings(
        self,
        *,
        document_id: str | None = None,
        limit: int,
    ) -> list[ChunkWithoutEmbedding]: ...

    async def list_documents_missing_embeddings(
        self,
        *,
        limit: int,
        after_document_id: str | None = None,
    ) -> list[str]: ...

    async def update_chunk_embeddings(
        self,
        *,
        updates: list[ChunkEmbeddingUpdate],
    ) -> int: ...

    async def mark_document_ready(self, document_id: str) -> DocumentRecord | None: ...

    async def mark_document_failed(
        self,
        *,
        document_id: str,
        failure_reason: str,
    ) -> DocumentRecord | None: ...


@dataclass(frozen=True)
class UserRecord:
    id: str
    display_name: str | None
    email: str | None
    plan: str | None
    locale: str | None
    preferences_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CreatedApiKey:
    user: UserRecord
    api_key: str
    key_prefix: str
    created_at: datetime


class SqlAlchemyChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_messages(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[ChatTurn]:
        return [
            ChatTurn(role=message.role, content=message.content)
            for message in await self.list_message_records(
                conversation_id,
                owner_user_id=owner_user_id,
            )
        ]

    async def list_message_records(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[MessageRecord]:
        conversation = await self._get_conversation(conversation_id)
        if conversation is None:
            return []
        if owner_user_id is not None and conversation.owner_user_id != owner_user_id:
            raise OwnershipError("conversation does not belong to the authenticated user")

        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return [
            self._message_to_record(message)
            for message in result.scalars().all()
        ]

    async def conversation_exists(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> bool:
        query = (
            select(literal(True))
            .select_from(Conversation)
            .where(Conversation.id == conversation_id)
        )
        if owner_user_id is not None:
            query = query.where(Conversation.owner_user_id == owner_user_id)
        result = await self._session.execute(query.limit(1))
        return result.scalar() is True

    async def list_tool_runs(
        self,
        conversation_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[ToolRunRecord]:
        if not await self.conversation_exists(conversation_id, owner_user_id=owner_user_id):
            return []
        result = await self._session.execute(
            select(ToolRun)
            .where(ToolRun.conversation_id == conversation_id)
            .order_by(ToolRun.id.desc())
            .limit(limit)
        )
        return [
            self._tool_run_to_record(tool_run)
            for tool_run in result.scalars().all()
        ]

    async def create_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        input_payload: dict[str, Any],
        owner_user_id: str | None = None,
    ) -> ToolRunRecord:
        timestamp = utcnow()

        try:
            conversation = await self._get_or_create_conversation(
                conversation_id,
                timestamp,
                owner_user_id=owner_user_id,
            )
            conversation.updated_at = timestamp

            tool_run = ToolRun(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status="running",
                input_payload=input_payload,
                output_payload=None,
                error_message=None,
                started_at=timestamp,
                completed_at=None,
            )
            self._session.add(tool_run)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return self._tool_run_to_record(tool_run)

    async def complete_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        output_payload: dict[str, Any],
        owner_user_id: str | None = None,
    ) -> ToolRunRecord | None:
        return await self._update_tool_run(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="completed",
            output_payload=output_payload,
            error_message=None,
            owner_user_id=owner_user_id,
        )

    async def fail_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        error_message: str,
        owner_user_id: str | None = None,
    ) -> ToolRunRecord | None:
        return await self._update_tool_run(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status=status,
            output_payload=None,
            error_message=error_message,
            owner_user_id=owner_user_id,
        )

    async def append_exchange(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_metadata: dict[str, Any] | None,
        assistant_message: str,
        owner_user_id: str | None = None,
    ) -> PersistedExchange:
        timestamp = utcnow()

        try:
            conversation = await self._get_or_create_conversation(
                conversation_id,
                timestamp,
                owner_user_id=owner_user_id,
            )
            conversation.updated_at = timestamp

            user_record = Message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                metadata_=user_metadata,
                created_at=timestamp,
            )
            assistant_record = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_message,
                metadata_=None,
                created_at=timestamp,
            )
            self._session.add_all([user_record, assistant_record])
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return PersistedExchange(
            conversation_id=conversation_id,
            user_message_id=user_record.id,
            assistant_message_id=assistant_record.id,
            created_at=timestamp,
        )

    async def _update_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        output_payload: dict[str, Any] | None,
        error_message: str | None,
        owner_user_id: str | None,
    ) -> ToolRunRecord | None:
        timestamp = utcnow()

        try:
            conversation = await self._get_conversation(
                conversation_id,
                owner_user_id=owner_user_id,
            )
            if conversation is None:
                return None
            result = await self._session.execute(
                select(ToolRun)
                .where(ToolRun.conversation_id == conversation_id)
                .where(ToolRun.tool_call_id == tool_call_id)
                .order_by(ToolRun.id.desc())
                .limit(1)
            )
            tool_run = result.scalar_one_or_none()
            if tool_run is None:
                return None

            conversation.updated_at = timestamp

            tool_run.status = status
            tool_run.output_payload = output_payload
            tool_run.error_message = error_message
            tool_run.completed_at = timestamp
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return self._tool_run_to_record(tool_run)

    async def _get_or_create_conversation(
        self,
        conversation_id: str,
        timestamp: Any,
        *,
        owner_user_id: str | None,
    ) -> Conversation:
        conversation = await self._get_conversation(conversation_id)
        if conversation is None:
            conversation = Conversation(
                id=conversation_id,
                owner_user_id=owner_user_id,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._session.add(conversation)
        elif owner_user_id is not None and conversation.owner_user_id != owner_user_id:
            raise OwnershipError("conversation does not belong to the authenticated user")
        return conversation

    async def _get_conversation(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> Conversation | None:
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .limit(1)
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            return None
        if owner_user_id is not None and conversation.owner_user_id != owner_user_id:
            return None
        return conversation

    def _tool_run_to_record(self, tool_run: ToolRun) -> ToolRunRecord:
        return ToolRunRecord(
            id=tool_run.id,
            conversation_id=tool_run.conversation_id,
            tool_call_id=tool_run.tool_call_id,
            tool_name=tool_run.tool_name,
            status=tool_run.status,
            input_payload=tool_run.input_payload,
            output_payload=tool_run.output_payload,
            error_message=tool_run.error_message,
            started_at=tool_run.started_at,
            completed_at=tool_run.completed_at,
        )

    def _message_to_record(self, message: Message) -> MessageRecord:
        return MessageRecord(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            metadata=message.metadata_,
            created_at=message.created_at,
        )


class SqlAlchemyMemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_conversation_summary(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> ConversationSummaryRecord | None:
        summary = await self._session.get(ConversationSummary, conversation_id)
        if summary is None:
            return None
        if owner_user_id is not None:
            conversation = await self._session.get(Conversation, conversation_id)
            if conversation is None or conversation.owner_user_id != owner_user_id:
                return None
        return self._summary_to_record(summary)

    async def upsert_conversation_summary(
        self,
        *,
        conversation_id: str,
        summary_text: str,
        last_summarized_message_id: int,
        owner_user_id: str | None = None,
    ) -> ConversationSummaryRecord:
        timestamp = utcnow()

        try:
            summary = await self._session.get(ConversationSummary, conversation_id)
            if summary is None:
                conversation = await self._session.get(Conversation, conversation_id)
                if conversation is None:
                    conversation = Conversation(
                        id=conversation_id,
                        owner_user_id=owner_user_id,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                    self._session.add(conversation)
                elif (
                    owner_user_id is not None
                    and conversation.owner_user_id != owner_user_id
                ):
                    raise OwnershipError(
                        "conversation does not belong to the authenticated user"
                    )

                summary = ConversationSummary(
                    conversation_id=conversation_id,
                    summary_text=summary_text,
                    last_summarized_message_id=last_summarized_message_id,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                self._session.add(summary)
            else:
                conversation = await self._session.get(Conversation, conversation_id)
                if (
                    owner_user_id is not None
                    and conversation is not None
                    and conversation.owner_user_id != owner_user_id
                ):
                    raise OwnershipError(
                        "conversation does not belong to the authenticated user"
                    )
                summary.summary_text = summary_text
                summary.last_summarized_message_id = last_summarized_message_id
                summary.updated_at = timestamp

            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return self._summary_to_record(summary)

    async def list_active_memories(
        self,
        user_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[MemoryRecord]:
        if owner_user_id is not None and owner_user_id != user_id:
            return []

        result = await self._session.execute(
            select(Memory)
            .where(Memory.user_id == user_id)
            .where(Memory.deleted_at.is_(None))
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
            .limit(limit)
        )
        return [
            self._memory_to_record(memory)
            for memory in result.scalars().all()
        ]

    async def upsert_memory(
        self,
        *,
        user_id: str,
        kind: Literal["profile", "preference"],
        key: str,
        value_json: dict[str, Any],
        confidence: float,
        source_message_id: int,
        extraction_method: Literal["rule", "llm"],
        owner_user_id: str | None = None,
    ) -> MemoryRecord:
        if owner_user_id is not None and owner_user_id != user_id:
            raise OwnershipError("memory does not belong to the authenticated user")

        timestamp = utcnow()

        try:
            result = await self._session.execute(
                select(Memory)
                .where(Memory.user_id == user_id)
                .where(Memory.key == key)
                .order_by(Memory.updated_at.desc(), Memory.id.desc())
                .limit(1)
            )
            memory = result.scalar_one_or_none()
            if memory is None:
                memory = Memory(
                    user_id=user_id,
                    kind=kind,
                    key=key,
                    value_json=value_json,
                    confidence=confidence,
                    source_message_id=source_message_id,
                    extraction_method=extraction_method,
                    created_at=timestamp,
                    updated_at=timestamp,
                    deleted_at=None,
                )
                self._session.add(memory)
            else:
                memory.kind = kind
                memory.value_json = value_json
                memory.confidence = confidence
                memory.source_message_id = source_message_id
                memory.extraction_method = extraction_method
                memory.updated_at = timestamp
                memory.deleted_at = None

            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return self._memory_to_record(memory)

    async def delete_memory(
        self,
        *,
        user_id: str,
        memory_id: int,
        owner_user_id: str | None = None,
    ) -> bool:
        if owner_user_id is not None and owner_user_id != user_id:
            return False

        timestamp = utcnow()
        try:
            result = await self._session.execute(
                select(Memory)
                .where(Memory.id == memory_id)
                .where(Memory.user_id == user_id)
                .where(Memory.deleted_at.is_(None))
                .limit(1)
            )
            memory = result.scalar_one_or_none()
            if memory is None:
                return False

            memory.deleted_at = timestamp
            memory.updated_at = timestamp
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return True

    def _summary_to_record(
        self,
        summary: ConversationSummary,
    ) -> ConversationSummaryRecord:
        return ConversationSummaryRecord(
            conversation_id=summary.conversation_id,
            summary_text=summary.summary_text,
            last_summarized_message_id=summary.last_summarized_message_id,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )

    def _memory_to_record(self, memory: Memory) -> MemoryRecord:
        return MemoryRecord(
            id=memory.id,
            user_id=memory.user_id,
            kind=memory.kind,
            key=memory.key,
            value_json=memory.value_json,
            confidence=memory.confidence,
            source_message_id=memory.source_message_id,
            extraction_method=memory.extraction_method,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            deleted_at=memory.deleted_at,
        )


API_KEY_LAST_USED_THROTTLE = timedelta(seconds=60)


class SqlAlchemyAuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def authenticate_api_key(self, api_key: str) -> AuthenticatedUser | None:
        api_key_hash = hash_api_key(api_key)
        result = await self._session.execute(
            select(ApiKey, User)
            .join(User, User.id == ApiKey.user_id)
            .where(ApiKey.key_hash == api_key_hash)
            .where(ApiKey.revoked_at.is_(None))
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None

        api_key_row, user_row = row
        now = utcnow()
        if (
            api_key_row.last_used_at is None
            or now - api_key_row.last_used_at > API_KEY_LAST_USED_THROTTLE
        ):
            api_key_row.last_used_at = now
            await self._session.commit()
        return AuthenticatedUser(
            user_id=user_row.id,
            display_name=user_row.display_name,
            email=user_row.email,
            plan=user_row.plan,
            locale=user_row.locale,
            preferences=dict(user_row.preferences_json or {}),
        )

    async def upsert_user(
        self,
        *,
        user_id: str,
        display_name: str | None,
        email: str | None,
        plan: str | None,
        locale: str | None,
        preferences_json: dict[str, Any] | None = None,
    ) -> UserRecord:
        timestamp = utcnow()
        payload = {} if preferences_json is None else dict(preferences_json)
        try:
            user = await self._session.get(User, user_id)
            if user is None:
                user = User(
                    id=user_id,
                    display_name=display_name,
                    email=email,
                    plan=plan,
                    locale=locale,
                    preferences_json=payload,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                self._session.add(user)
            else:
                user.display_name = display_name
                user.email = email
                user.plan = plan
                user.locale = locale
                user.preferences_json = payload
                user.updated_at = timestamp
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return self._user_to_record(user)

    async def create_api_key(
        self,
        *,
        user_id: str,
        name: str,
    ) -> CreatedApiKey:
        user = await self._session.get(User, user_id)
        if user is None:
            raise ValueError("user not found")

        plaintext_api_key = generate_api_key()
        api_key_record = ApiKey(
            user_id=user_id,
            name=name,
            key_prefix=build_api_key_prefix(plaintext_api_key),
            key_hash=hash_api_key(plaintext_api_key),
            created_at=utcnow(),
            last_used_at=None,
            revoked_at=None,
        )
        try:
            self._session.add(api_key_record)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return CreatedApiKey(
            user=self._user_to_record(user),
            api_key=plaintext_api_key,
            key_prefix=api_key_record.key_prefix,
            created_at=api_key_record.created_at,
        )

    def _user_to_record(self, user: User) -> UserRecord:
        return UserRecord(
            id=user.id,
            display_name=user.display_name,
            email=user.email,
            plan=user.plan,
            locale=user.locale,
            preferences_json=dict(user.preferences_json or {}),
            created_at=user.created_at,
            updated_at=user.updated_at,
        )


class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        byte_size: int,
        checksum_sha256: str,
        status: str,
        failure_reason: str | None = None,
        chunks: list[DocumentChunkCreate],
        owner_user_id: str | None = None,
    ) -> DocumentRecord:
        timestamp = utcnow()

        try:
            document = Document(
                id=document_id,
                owner_user_id=owner_user_id,
                filename=filename,
                content_type=content_type,
                byte_size=byte_size,
                checksum_sha256=checksum_sha256,
                status=status,
                failure_reason=failure_reason,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._session.add(document)
            self._session.add_all(
                [
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        embedding=chunk.embedding,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        metadata_=chunk.metadata,
                        created_at=timestamp,
                    )
                    for chunk in chunks
                ]
            )
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return DocumentRecord(
            id=document.id,
            owner_user_id=document.owner_user_id,
            filename=document.filename,
            content_type=document.content_type,
            byte_size=document.byte_size,
            checksum_sha256=document.checksum_sha256,
            status=document.status,
            failure_reason=document.failure_reason,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    async def get_document(
        self,
        document_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> DocumentRecord | None:
        document = await self._session.get(Document, document_id)
        if document is None:
            return None
        if owner_user_id is not None and document.owner_user_id != owner_user_id:
            return None

        return self._document_to_record(document)

    async def find_document_by_checksum(
        self,
        checksum_sha256: str,
        *,
        owner_user_id: str | None = None,
    ) -> DocumentRecord | None:
        stmt = select(Document).where(Document.checksum_sha256 == checksum_sha256)
        if owner_user_id is not None:
            stmt = stmt.where(Document.owner_user_id == owner_user_id)
        stmt = stmt.order_by(Document.created_at.desc()).limit(1)

        result = await self._session.execute(stmt)
        document = result.scalar_one_or_none()
        return None if document is None else self._document_to_record(document)

    async def list_document_chunks(self, document_id: str) -> list[DocumentChunkCreate]:
        result = await self._session.execute(
            select(
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.embedding,
                DocumentChunk.start_offset,
                DocumentChunk.end_offset,
                DocumentChunk.metadata_,
            )
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return [
            DocumentChunkCreate(
                chunk_index=row.chunk_index,
                content=row.content,
                embedding=row.embedding,
                start_offset=row.start_offset,
                end_offset=row.end_offset,
                metadata=row.metadata_,
            )
            for row in result.all()
        ]

    async def count_document_chunks(
        self,
        document_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> int:
        if owner_user_id is not None:
            document = await self._session.get(Document, document_id)
            if document is None or document.owner_user_id != owner_user_id:
                return 0
        result = await self._session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        )
        return int(result.scalar_one())

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        if limit <= 0:
            return []

        bind = getattr(self._session, "bind", None)
        dialect_name = bind.dialect.name if bind is not None else ""
        if dialect_name == "postgresql":
            return await self._search_similar_chunks_postgres(
                query_embedding=query_embedding,
                limit=limit,
                owner_user_id=owner_user_id,
            )

        return await self._search_similar_chunks_python(
            query_embedding=query_embedding,
            limit=limit,
            owner_user_id=owner_user_id,
        )

    async def list_chunks_missing_embeddings(
        self,
        *,
        document_id: str | None = None,
        limit: int,
    ) -> list[ChunkWithoutEmbedding]:
        statement = (
            select(
                DocumentChunk.document_id,
                DocumentChunk.id,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.start_offset,
                DocumentChunk.end_offset,
                DocumentChunk.metadata_,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status.in_(("processing", "ready")))
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            .limit(limit)
        )
        if document_id is not None:
            statement = statement.where(DocumentChunk.document_id == document_id)

        result = await self._session.execute(statement)
        return [
            ChunkWithoutEmbedding(
                document_id=row.document_id,
                chunk_id=row.id,
                chunk_index=row.chunk_index,
                content=row.content,
                start_offset=row.start_offset,
                end_offset=row.end_offset,
                metadata=row.metadata_,
            )
            for row in result.all()
        ]

    async def list_documents_missing_embeddings(
        self,
        *,
        limit: int,
        after_document_id: str | None = None,
    ) -> list[str]:
        if limit <= 0:
            return []

        statement = (
            select(DocumentChunk.document_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status.in_(("processing", "ready")))
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.document_id)
            .limit(limit)
        )
        if after_document_id is not None:
            statement = statement.where(DocumentChunk.document_id > after_document_id)

        result = await self._session.execute(statement.distinct())
        return [str(document_id) for document_id in result.scalars().all()]

    async def update_chunk_embeddings(
        self,
        *,
        updates: list[ChunkEmbeddingUpdate],
    ) -> int:
        if not updates:
            return 0

        try:
            rows = await self._session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.id.in_([update.chunk_id for update in updates])
                )
            )
            chunks_by_id = {chunk.id: chunk for chunk in rows.scalars()}
            for update in updates:
                chunk = chunks_by_id.get(update.chunk_id)
                if chunk is not None:
                    chunk.embedding = update.embedding
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return len(chunks_by_id)

    async def mark_document_ready(self, document_id: str) -> DocumentRecord | None:
        return await self._update_document_status(
            document_id=document_id,
            status="ready",
            failure_reason=None,
        )

    async def mark_document_failed(
        self,
        *,
        document_id: str,
        failure_reason: str,
    ) -> DocumentRecord | None:
        return await self._update_document_status(
            document_id=document_id,
            status="failed",
            failure_reason=failure_reason,
        )

    async def _search_similar_chunks_postgres(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        statement = self._base_retrieval_select(distance, owner_user_id=owner_user_id).where(
            DocumentChunk.embedding.is_not(None)
        )
        result = await self._session.execute(
            statement.order_by(distance, DocumentChunk.id).limit(limit)
        )
        return [self._row_to_retrieved_chunk(row) for row in result.all()]

    async def _search_similar_chunks_python(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        result = await self._session.execute(
            self._base_retrieval_select(literal(0.0), owner_user_id=owner_user_id)
            .where(DocumentChunk.embedding.is_not(None))
        )
        ranked = []
        for row in result.all():
            embedding = row.embedding
            if embedding is None:
                continue
            ranked.append(
                (
                    cosine_similarity(query_embedding, embedding),
                    self._row_to_retrieved_chunk(row, score_override=0.0),
                )
            )

        ranked.sort(key=lambda item: (-item[0], item[1].document_id, item[1].chunk_index))
        return [
            RetrievedDocumentChunk(
                document_id=item.document_id,
                filename=item.filename,
                chunk_index=item.chunk_index,
                content=item.content,
                start_offset=item.start_offset,
                end_offset=item.end_offset,
                metadata=item.metadata,
                score=score,
            )
            for score, item in ranked[:limit]
        ]

    def _base_retrieval_select(
        self,
        score_expression: Any,
        *,
        owner_user_id: str | None = None,
    ) -> Select[Any]:
        statement = (
            select(
                DocumentChunk.document_id,
                Document.filename,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.embedding,
                DocumentChunk.start_offset,
                DocumentChunk.end_offset,
                DocumentChunk.metadata_,
                score_expression.label("score"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "ready")
        )
        if owner_user_id is not None:
            statement = statement.where(Document.owner_user_id == owner_user_id)
        return statement

    def _row_to_retrieved_chunk(
        self,
        row: Any,
        *,
        score_override: float | None = None,
    ) -> RetrievedDocumentChunk:
        score = score_override if score_override is not None else max(0.0, 1.0 - float(row.score))
        return RetrievedDocumentChunk(
            document_id=row.document_id,
            filename=row.filename,
            chunk_index=row.chunk_index,
            content=row.content,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            metadata=row.metadata_,
            score=score,
        )

    async def _update_document_status(
        self,
        *,
        document_id: str,
        status: str,
        failure_reason: str | None,
    ) -> DocumentRecord | None:
        try:
            document = await self._session.get(Document, document_id)
            if document is None:
                return None

            document.status = status
            document.failure_reason = failure_reason
            document.updated_at = utcnow()
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

        return self._document_to_record(document)

    def _document_to_record(self, document: Document) -> DocumentRecord:
        return DocumentRecord(
            id=document.id,
            owner_user_id=document.owner_user_id,
            filename=document.filename,
            content_type=document.content_type,
            byte_size=document.byte_size,
            checksum_sha256=document.checksum_sha256,
            status=document.status,
            failure_reason=document.failure_reason,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)
