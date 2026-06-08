from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from chatbot_api.document_ingestion import (
    DefaultDocumentTextExtractor,
    DocumentChunkCreate,
    DocumentIngestionService,
    DocumentRecord,
    TextChunker,
)
from chatbot_api.main import (
    app,
    get_document_repository,
    get_document_service,
    get_document_task_queue,
)


@dataclass(frozen=True)
class StoredDocument:
    record: DocumentRecord
    chunks: list[DocumentChunkCreate]


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self.documents: dict[str, StoredDocument] = {}

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
        now = datetime.now(UTC)
        record = DocumentRecord(
            id=document_id,
            filename=filename,
            content_type=content_type,
            byte_size=byte_size,
            checksum_sha256=checksum_sha256,
            status=status,
            failure_reason=failure_reason,
            created_at=now,
            updated_at=now,
        )
        self.documents[document_id] = StoredDocument(record=record, chunks=list(chunks))
        return record

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        stored = self.documents.get(document_id)
        if stored is None:
            return None
        return stored.record

    async def count_document_chunks(self, document_id: str) -> int:
        stored = self.documents.get(document_id)
        return len(stored.chunks) if stored is not None else 0

    async def mark_document_ready(self, document_id: str) -> DocumentRecord | None:
        return await self._update_status(document_id, status="ready", failure_reason=None)

    async def mark_document_failed(
        self,
        *,
        document_id: str,
        failure_reason: str,
    ) -> DocumentRecord | None:
        return await self._update_status(
            document_id,
            status="failed",
            failure_reason=failure_reason,
        )

    async def _update_status(
        self,
        document_id: str,
        *,
        status: str,
        failure_reason: str | None,
    ) -> DocumentRecord | None:
        stored = self.documents.get(document_id)
        if stored is None:
            return None

        updated = DocumentRecord(
            id=stored.record.id,
            filename=stored.record.filename,
            content_type=stored.record.content_type,
            byte_size=stored.record.byte_size,
            checksum_sha256=stored.record.checksum_sha256,
            status=status,
            failure_reason=failure_reason,
            created_at=stored.record.created_at,
            updated_at=datetime.now(UTC),
        )
        self.documents[document_id] = StoredDocument(record=updated, chunks=stored.chunks)
        return updated


class StubDocumentTaskQueue:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.enqueued_document_ids: list[str] = []

    def enqueue_embed_document(self, document_id: str) -> None:
        self.enqueued_document_ids.append(document_id)
        if self.fail:
            raise RuntimeError("Queue unavailable")


def build_document_service_override(
    service: DocumentIngestionService,
):
    async def override() -> DocumentIngestionService:
        return service

    return override


def build_document_repository_override(repository: InMemoryDocumentRepository):
    async def override() -> InMemoryDocumentRepository:
        return repository

    return override


def build_document_task_queue_override(task_queue: StubDocumentTaskQueue):
    async def override() -> StubDocumentTaskQueue:
        return task_queue

    return override


def build_document_service(
    repository: InMemoryDocumentRepository,
    *,
    max_bytes: int = 5_242_880,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        repository,
        DefaultDocumentTextExtractor(),
        TextChunker(chunk_size=1200, chunk_overlap=200),
        max_bytes=max_bytes,
    )


def build_minimal_pdf_bytes(text: str) -> bytes:
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped_text}) Tj\nET\n"
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            "3 0 obj\n"
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            "endobj\n"
        ),
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream.encode('latin-1'))} >>\nstream\n"
            f"{stream}endstream\nendobj\n"
        ),
    ]

    header = "%PDF-1.4\n"
    body = header
    offsets: list[int] = []
    current_offset = len(header.encode("latin-1"))
    for obj in objects:
        offsets.append(current_offset)
        body += obj
        current_offset += len(obj.encode("latin-1"))

    xref_offset = len(body.encode("latin-1"))
    xref_rows = ["0000000000 65535 f \n"]
    xref_rows.extend(f"{offset:010d} 00000 n \n" for offset in offsets)
    xref = f"xref\n0 {len(offsets) + 1}\n{''.join(xref_rows)}"
    trailer = (
        f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    return (body + xref + trailer).encode("latin-1")


@pytest.fixture
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_document_upload_accepts_text_and_persists_chunks(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    response = await async_client.post(
        "/documents",
        files={"file": ("notes.txt", b"Hello world\n\nThis is a test document.", "text/plain")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "notes.txt"
    assert payload["content_type"] == "text/plain"
    assert payload["status"] == "processing"
    assert payload["chunk_count"] == 1
    stored = repository.documents[payload["document_id"]]
    assert stored.record.byte_size == len(b"Hello world\n\nThis is a test document.")
    assert [chunk.content for chunk in stored.chunks] == ["Hello world\n\nThis is a test document."]
    assert stored.chunks[0].embedding is None
    assert task_queue.enqueued_document_ids == [payload["document_id"]]


@pytest.mark.anyio
async def test_document_upload_accepts_markdown(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    response = await async_client.post(
        "/documents",
        files={"file": ("notes.md", b"# Title\n\n- item", "text/markdown")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["content_type"] == "text/markdown"
    stored = repository.documents[payload["document_id"]]
    assert stored.chunks[0].content == "# Title\n\n- item"


@pytest.mark.anyio
async def test_document_upload_accepts_pdf(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )
    pdf_bytes = build_minimal_pdf_bytes("Hello PDF")

    response = await async_client.post(
        "/documents",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["content_type"] == "application/pdf"
    stored = repository.documents[payload["document_id"]]
    assert stored.chunks[0].content == "Hello PDF"


@pytest.mark.anyio
async def test_document_upload_rejects_unsupported_type(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    response = await async_client.post(
        "/documents",
        files={"file": ("image.png", b"not-really-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "unsupported document type"}


@pytest.mark.anyio
async def test_document_upload_rejects_empty_text(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    response = await async_client.post(
        "/documents",
        files={"file": ("empty.txt", b"   \n\t", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "document text is empty"}


@pytest.mark.anyio
async def test_document_upload_rejects_oversized_file(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository, max_bytes=8)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    response = await async_client.post(
        "/documents",
        files={"file": ("big.txt", b"0123456789", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "document exceeds maximum allowed size"}


@pytest.mark.anyio
async def test_document_upload_returns_service_unavailable_when_enqueue_fails(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue(fail=True)
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    response = await async_client.post(
        "/documents",
        files={"file": ("notes.txt", b"Hello world", "text/plain")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "failed to enqueue document embedding task"}
    document_id = task_queue.enqueued_document_ids[0]
    assert repository.documents[document_id].record.status == "failed"
    assert repository.documents[document_id].record.failure_reason == "enqueue_failed"


@pytest.mark.anyio
async def test_get_document_returns_current_status(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = InMemoryDocumentRepository()
    task_queue = StubDocumentTaskQueue()
    app.dependency_overrides[get_document_service] = build_document_service_override(
        build_document_service(repository)
    )
    app.dependency_overrides[get_document_repository] = build_document_repository_override(
        repository
    )
    app.dependency_overrides[get_document_task_queue] = build_document_task_queue_override(
        task_queue
    )

    upload_response = await async_client.post(
        "/documents",
        files={"file": ("notes.txt", b"Hello world", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    await repository.mark_document_ready(document_id)

    response = await async_client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["chunk_count"] == 1
