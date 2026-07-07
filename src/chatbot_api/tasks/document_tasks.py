from __future__ import annotations

from typing import Protocol

from chatbot_api.tasks.embedding_jobs import embed_document_task


class DocumentTaskQueue(Protocol):
    def enqueue_embed_document(self, document_id: str) -> None: ...


class CeleryDocumentTaskQueue:
    def enqueue_embed_document(self, document_id: str) -> None:
        embed_document_task.delay(document_id)
