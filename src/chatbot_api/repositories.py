from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import Select, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_api.document_ingestion import DocumentChunkCreate, DocumentRecord
from chatbot_api.models import Conversation, Document, DocumentChunk, Message, ToolRun, utcnow
from chatbot_api.providers import ChatTurn


class ChatRepository(Protocol):
    async def list_messages(self, conversation_id: str) -> list[ChatTurn]: ...

    async def conversation_exists(self, conversation_id: str) -> bool: ...

    async def list_tool_runs(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list["ToolRunRecord"]: ...

    async def create_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        input_payload: dict[str, Any],
    ) -> "ToolRunRecord": ...

    async def complete_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        output_payload: dict[str, Any],
    ) -> "ToolRunRecord | None": ...

    async def fail_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        error_message: str,
    ) -> "ToolRunRecord | None": ...

    async def append_exchange(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_metadata: dict[str, Any] | None,
        assistant_message: str,
    ) -> None: ...


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
    ) -> DocumentRecord: ...

    async def get_document(self, document_id: str) -> DocumentRecord | None: ...

    async def list_document_chunks(self, document_id: str) -> list[DocumentChunkCreate]: ...

    async def count_document_chunks(self, document_id: str) -> int: ...

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
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


class SqlAlchemyChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_messages(self, conversation_id: str) -> list[ChatTurn]:
        result = await self._session.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return [
            ChatTurn(role=row.role, content=row.content)
            for row in result.all()
        ]

    async def conversation_exists(self, conversation_id: str) -> bool:
        result = await self._session.execute(
            select(literal(True))
            .select_from(Conversation)
            .where(Conversation.id == conversation_id)
            .limit(1)
        )
        return result.scalar() is True

    async def list_tool_runs(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[ToolRunRecord]:
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
    ) -> ToolRunRecord:
        timestamp = utcnow()

        try:
            conversation = await self._get_or_create_conversation(conversation_id, timestamp)
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
    ) -> ToolRunRecord | None:
        return await self._update_tool_run(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="completed",
            output_payload=output_payload,
            error_message=None,
        )

    async def fail_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        error_message: str,
    ) -> ToolRunRecord | None:
        return await self._update_tool_run(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status=status,
            output_payload=None,
            error_message=error_message,
        )

    async def append_exchange(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_metadata: dict[str, Any] | None,
        assistant_message: str,
    ) -> None:
        timestamp = utcnow()

        try:
            conversation = await self._get_or_create_conversation(conversation_id, timestamp)
            conversation.updated_at = timestamp

            self._session.add_all(
                [
                    Message(
                        conversation_id=conversation_id,
                        role="user",
                        content=user_message,
                        metadata_=user_metadata,
                        created_at=timestamp,
                    ),
                    Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_message,
                        metadata_=None,
                        created_at=timestamp,
                    ),
                ]
            )
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

    async def _update_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        output_payload: dict[str, Any] | None,
        error_message: str | None,
    ) -> ToolRunRecord | None:
        timestamp = utcnow()

        try:
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

            conversation = await self._get_or_create_conversation(conversation_id, timestamp)
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
    ) -> Conversation:
        conversation = await self._session.get(Conversation, conversation_id)
        if conversation is None:
            conversation = Conversation(
                id=conversation_id,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._session.add(conversation)
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
    ) -> DocumentRecord:
        timestamp = utcnow()

        try:
            document = Document(
                id=document_id,
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
            filename=document.filename,
            content_type=document.content_type,
            byte_size=document.byte_size,
            checksum_sha256=document.checksum_sha256,
            status=document.status,
            failure_reason=document.failure_reason,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        document = await self._session.get(Document, document_id)
        if document is None:
            return None

        return self._document_to_record(document)

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

    async def count_document_chunks(self, document_id: str) -> int:
        result = await self._session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        )
        return int(result.scalar_one())

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedDocumentChunk]:
        if limit <= 0:
            return []

        bind = getattr(self._session, "bind", None)
        dialect_name = bind.dialect.name if bind is not None else ""
        if dialect_name == "postgresql":
            return await self._search_similar_chunks_postgres(
                query_embedding=query_embedding,
                limit=limit,
            )

        return await self._search_similar_chunks_python(
            query_embedding=query_embedding,
            limit=limit,
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
    ) -> list[RetrievedDocumentChunk]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        result = await self._session.execute(
            self._base_retrieval_select(distance)
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance, DocumentChunk.id)
            .limit(limit)
        )
        return [self._row_to_retrieved_chunk(row) for row in result.all()]

    async def _search_similar_chunks_python(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedDocumentChunk]:
        result = await self._session.execute(
            self._base_retrieval_select(literal(0.0))
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

    def _base_retrieval_select(self, score_expression: Any) -> Select[Any]:
        return (
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
