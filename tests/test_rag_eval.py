from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.document_ingestion import DocumentChunkCreate
from chatbot_api.models import Base
from chatbot_api.rag_eval import (
    RetrievalEvalCase,
    build_case_report,
    load_retrieval_eval_dataset,
    run_retrieval_eval,
)
from chatbot_api.repositories import RetrievedDocumentChunk, SqlAlchemyDocumentRepository
from chatbot_api.settings import Settings


class StubEmbeddingProvider:
    def __init__(self, embeddings_by_text: dict[str, list[float]]) -> None:
        self._embeddings_by_text = embeddings_by_text
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = list(texts)
        self.calls.append(normalized)
        return [self._embeddings_by_text[text] for text in normalized]


def make_chunk(
    *,
    document_id: str,
    filename: str,
    chunk_index: int,
    score: float,
) -> RetrievedDocumentChunk:
    content = f"{filename} chunk {chunk_index}"
    return RetrievedDocumentChunk(
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        content=content,
        start_offset=chunk_index * 100,
        end_offset=(chunk_index * 100) + len(content),
        metadata=None,
        score=score,
    )


def write_dataset(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_retrieval_eval_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "duplicate",
                    "query": "first question",
                    "expected_sources": [{"filename": "guide.md"}],
                },
                {
                    "id": "duplicate",
                    "query": "second question",
                    "expected_sources": [{"filename": "faq.md"}],
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate retrieval eval case id"):
        load_retrieval_eval_dataset(dataset_path)


def test_load_retrieval_eval_dataset_rejects_missing_source_selector(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "missing-source-selector",
                    "query": "where is the guide",
                    "expected_sources": [{}],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="expected source must include filename or document_id"):
        load_retrieval_eval_dataset(dataset_path)


def test_build_case_report_computes_partial_document_coverage() -> None:
    case = RetrievalEvalCase.model_validate(
        {
            "id": "partial-hit",
            "query": "compare guide and faq",
            "expected_sources": [
                {"filename": "guide.md"},
                {"filename": "faq.md"},
            ],
            "minimum_expected_hits": 2,
        }
    )

    report = build_case_report(
        case,
        [make_chunk(document_id="doc-guide", filename="guide.md", chunk_index=0, score=0.91)],
    )

    assert report.document_hit is False
    assert report.document_coverage == 0.5
    assert report.matched_source_count == 1
    assert report.retrieved_document_ids == ["doc-guide"]
    assert report.no_result is False


def test_build_case_report_marks_chunk_hits_and_misses() -> None:
    case = RetrievalEvalCase.model_validate(
        {
            "id": "chunk-match",
            "query": "show setup chunk",
            "expected_sources": [{"filename": "guide.md", "chunk_indexes": [1]}],
        }
    )

    hit_report = build_case_report(
        case,
        [make_chunk(document_id="doc-guide", filename="guide.md", chunk_index=1, score=0.95)],
    )
    miss_report = build_case_report(
        case,
        [make_chunk(document_id="doc-guide", filename="guide.md", chunk_index=0, score=0.95)],
    )

    assert hit_report.chunk_hit is True
    assert miss_report.chunk_hit is False


def test_build_case_report_marks_no_result_when_retrieval_is_empty() -> None:
    case = RetrievalEvalCase.model_validate(
        {
            "id": "no-result",
            "query": "unknown question",
            "expected_sources": [{"filename": "guide.md"}],
        }
    )

    report = build_case_report(case, [])

    assert report.no_result is True
    assert report.document_hit is False
    assert report.document_coverage == 0.0


@pytest.mark.anyio
async def test_run_retrieval_eval_writes_json_report(tmp_path: Path) -> None:
    database_path = tmp_path / "rag-eval.db"
    dataset_path = tmp_path / "dataset.json"
    output_path = tmp_path / "report.json"
    database_url = f"sqlite+aiosqlite:///{database_path}"

    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "guide-hit",
                    "query": "guide question",
                    "expected_sources": [{"filename": "guide.md", "chunk_indexes": [0]}],
                },
                {
                    "id": "faq-hit",
                    "query": "billing question",
                    "expected_sources": [{"filename": "faq.md"}],
                },
            ]
        },
    )

    settings = Settings(
        database_url=database_url,
        langgraph_checkpoint_database_url=None,
        retrieval_top_k=2,
        retrieval_min_score=0.3,
        retrieval_max_chunks_per_document=1,
        retrieval_candidate_limit=4,
    )
    embedding_provider = StubEmbeddingProvider(
        {
            "guide question": [1.0, 0.0],
            "billing question": [0.0, 1.0],
        }
    )

    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            await repository.create_document(
                document_id="doc-guide",
                filename="guide.md",
                content_type="text/markdown",
                byte_size=100,
                checksum_sha256="hash-guide",
                status="ready",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="guide setup steps",
                        embedding=[1.0, 0.0],
                        start_offset=0,
                        end_offset=17,
                    ),
                    DocumentChunkCreate(
                        chunk_index=1,
                        content="guide advanced notes",
                        embedding=[0.8, 0.2],
                        start_offset=18,
                        end_offset=38,
                    ),
                ],
            )
            await repository.create_document(
                document_id="doc-faq",
                filename="faq.md",
                content_type="text/markdown",
                byte_size=80,
                checksum_sha256="hash-faq",
                status="ready",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="billing FAQ details",
                        embedding=[0.0, 1.0],
                        start_offset=0,
                        end_offset=19,
                    )
                ],
            )

        report = await run_retrieval_eval(
            dataset_path=dataset_path,
            output_path=output_path,
            settings=settings,
            embedding_provider=embedding_provider,
        )
    finally:
        await engine.dispose()

    assert report.summary.total_cases == 2
    assert report.summary.document_hit_rate == 1.0
    assert report.summary.average_document_coverage == 1.0
    assert report.summary.chunk_hit_rate == 1.0
    assert report.summary.failed_case_ids == []
    assert embedding_provider.calls == [["guide question"], ["billing question"]]

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_cases"] == 2
    assert payload["summary"]["document_hit_rate"] == 1.0
    assert payload["cases"][0]["case_id"] == "guide-hit"
