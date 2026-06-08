from __future__ import annotations

from dataclasses import dataclass

from chatbot_api.embeddings import EmbeddingProvider
from chatbot_api.repositories import (
    ChunkEmbeddingUpdate,
    ChunkWithoutEmbedding,
    DocumentRepository,
)


@dataclass(frozen=True)
class DocumentEmbeddingResult:
    document_id: str
    status: str
    updated_chunks: int


class DocumentEmbeddingService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        *,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._repository = repository
        self._embedding_provider = embedding_provider
        self._batch_size = batch_size

    async def embed_document(self, document_id: str) -> DocumentEmbeddingResult | None:
        document = await self._repository.get_document(document_id)
        if document is None:
            return None
        if document.status == "failed":
            return DocumentEmbeddingResult(
                document_id=document_id,
                status=document.status,
                updated_chunks=0,
            )

        updated_chunks = 0
        while True:
            missing_chunks = await self._repository.list_chunks_missing_embeddings(
                document_id=document_id,
                limit=self._batch_size,
            )
            if not missing_chunks:
                break

            updated_chunks += await self._embed_chunk_batch(missing_chunks)

        refreshed_document = await self._repository.get_document(document_id)
        if refreshed_document is None:
            return None

        if refreshed_document.status != "ready":
            refreshed_document = await self._repository.mark_document_ready(document_id)
        if refreshed_document is None:
            return None

        return DocumentEmbeddingResult(
            document_id=document_id,
            status=refreshed_document.status,
            updated_chunks=updated_chunks,
        )

    async def mark_failed(self, *, document_id: str, failure_reason: str) -> None:
        await self._repository.mark_document_failed(
            document_id=document_id,
            failure_reason=failure_reason,
        )

    async def _embed_chunk_batch(self, chunks: list[ChunkWithoutEmbedding]) -> int:
        embeddings = await self._embedding_provider.embed_texts([chunk.content for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedding provider returned invalid chunk embeddings")

        return await self._repository.update_chunk_embeddings(
            updates=[
                ChunkEmbeddingUpdate(
                    chunk_id=chunk.chunk_id,
                    embedding=embeddings[index],
                )
                for index, chunk in enumerate(chunks)
            ]
        )
