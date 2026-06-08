from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from typing import Protocol
from uuid import uuid4

from pypdf import PdfReader


class DocumentIngestionError(Exception):
    """Base ingestion error."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when an uploaded file type is not supported."""


class DocumentTooLargeError(DocumentIngestionError):
    """Raised when an uploaded file exceeds the configured size limit."""


class DocumentContentError(DocumentIngestionError):
    """Raised when the uploaded file cannot be turned into usable text."""


@dataclass(frozen=True)
class ExtractedDocument:
    content_type: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class DocumentChunkCreate:
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    metadata: dict[str, str | int] | None = None
    embedding: list[float] | None = None


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    filename: str
    content_type: str
    byte_size: int
    checksum_sha256: str
    status: str
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime


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


class DocumentTextExtractor(Protocol):
    def extract_text(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> ExtractedDocument: ...


def normalize_document_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


class DefaultDocumentTextExtractor:
    _TEXT_CONTENT_TYPES = {"text/plain", "text/markdown"}
    _PDF_CONTENT_TYPES = {"application/pdf"}

    def extract_text(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> ExtractedDocument:
        normalized_content_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
        extension = filename.lower().rsplit(".", maxsplit=1)[-1] if "." in filename else ""

        if (
            extension in {"txt", "md", "markdown"}
            or normalized_content_type in self._TEXT_CONTENT_TYPES
        ):
            return ExtractedDocument(
                content_type="text/markdown" if extension in {"md", "markdown"} else "text/plain",
                text=self._extract_utf8_text(content),
            )

        if extension == "pdf" or normalized_content_type in self._PDF_CONTENT_TYPES:
            return ExtractedDocument(
                content_type="application/pdf",
                text=self._extract_pdf_text(content),
            )

        raise UnsupportedDocumentTypeError("unsupported document type")

    def _extract_utf8_text(self, content: bytes) -> str:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentContentError("document must be valid UTF-8 text") from exc

        normalized = normalize_document_text(text)
        if not normalized:
            raise DocumentContentError("document text is empty")

        return normalized

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:  # pragma: no cover - parser-specific failure modes
            raise DocumentContentError("failed to read PDF document") from exc

        page_texts = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            normalized_page = normalize_document_text(extracted)
            if normalized_page:
                page_texts.append(normalized_page)

        if not page_texts:
            raise DocumentContentError("document text is empty")

        return "\n\n".join(page_texts)


class TextChunker:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[TextChunk]:
        normalized = normalize_document_text(text)
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = self._determine_chunk_end(normalized, start)
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    content=normalized[start:end],
                    start_offset=start,
                    end_offset=end,
                )
            )
            if end >= text_length:
                break

            next_start = end - self._chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _determine_chunk_end(self, text: str, start: int) -> int:
        end_limit = min(start + self._chunk_size, len(text))
        if end_limit == len(text):
            return end_limit

        min_split = start + max(1, self._chunk_size // 2)
        for separator in ("\n\n", "\n", " "):
            candidate = text.rfind(separator, min_split, end_limit)
            if candidate != -1:
                return candidate + len(separator)

        return end_limit


class DocumentIngestionService:
    def __init__(
        self,
        repository: DocumentRepository,
        extractor: DocumentTextExtractor,
        chunker: TextChunker,
        *,
        max_bytes: int,
    ) -> None:
        self._repository = repository
        self._extractor = extractor
        self._chunker = chunker
        self._max_bytes = max_bytes

    async def ingest_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> tuple[DocumentRecord, int]:
        if len(content) > self._max_bytes:
            raise DocumentTooLargeError("document exceeds maximum allowed size")

        extracted = self._extractor.extract_text(
            filename=filename,
            content_type=content_type,
            content=content,
        )
        chunks = self._chunker.chunk_text(extracted.text)
        if not chunks:
            raise DocumentContentError("document text is empty")
        record = await self._repository.create_document(
            document_id=str(uuid4()),
            filename=filename,
            content_type=extracted.content_type,
            byte_size=len(content),
            checksum_sha256=sha256(content).hexdigest(),
            status="processing",
            failure_reason=None,
            chunks=[
                DocumentChunkCreate(
                    chunk_index=chunk.index,
                    content=chunk.content,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                )
                for chunk in chunks
            ],
        )
        return record, len(chunks)
