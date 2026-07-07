from __future__ import annotations

import asyncio

from chatbot_api.database import session_scope
from chatbot_api.repositories import SqlAlchemyDocumentRepository
from chatbot_api.settings import get_settings
from chatbot_api.tasks.document_tasks import CeleryDocumentTaskQueue, DocumentTaskQueue


async def enqueue_documents_missing_embeddings(
    task_queue: DocumentTaskQueue | None = None,
) -> int:
    settings = get_settings()
    resolved_task_queue = task_queue or CeleryDocumentTaskQueue()
    total_enqueued = 0
    last_document_id: str | None = None

    async with session_scope(settings.database_url) as session_factory:
        while True:
            async with session_factory() as session:
                repository = SqlAlchemyDocumentRepository(session)
                document_ids = await repository.list_documents_missing_embeddings(
                    limit=settings.document_reindex_page_size,
                    after_document_id=last_document_id,
                )

            if not document_ids:
                break

            for document_id in document_ids:
                resolved_task_queue.enqueue_embed_document(document_id)
                total_enqueued += 1

            last_document_id = document_ids[-1]

    return total_enqueued


def main() -> None:
    enqueued = asyncio.run(enqueue_documents_missing_embeddings())
    print(f"Enqueued embedding jobs for {enqueued} documents.")


if __name__ == "__main__":
    main()
