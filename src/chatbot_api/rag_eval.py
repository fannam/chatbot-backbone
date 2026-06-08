from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from chatbot_api.eval_common import (
    ExpectedSource,
    safe_ratio,
    source_matches_reference,
    unique_preserving_order,
)
from chatbot_api.repositories import RetrievedDocumentChunk, SqlAlchemyDocumentRepository
from chatbot_api.retrieval import DocumentRetriever
from chatbot_api.settings import Settings, get_settings

DEFAULT_DATASET_PATH = Path("evals/rag_retrieval_baseline.json")


class ExpectedRetrievalSource(ExpectedSource):
    pass


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    query: str
    expected_sources: list[ExpectedRetrievalSource] = Field(min_length=1)
    minimum_expected_hits: int = 1
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case id must not be blank")
        return normalized

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_minimum_expected_hits(self) -> RetrievalEvalCase:
        if self.minimum_expected_hits < 1:
            raise ValueError("minimum_expected_hits must be at least 1")
        if self.minimum_expected_hits > len(self.expected_sources):
            raise ValueError("minimum_expected_hits cannot exceed expected_sources length")
        return self


class RetrievalEvalDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[RetrievalEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> RetrievalEvalDataset:
        seen_case_ids: set[str] = set()
        for case in self.cases:
            if case.id in seen_case_ids:
                raise ValueError(f"duplicate retrieval eval case id: {case.id}")
            seen_case_ids.add(case.id)
        return self


class RetrievalEvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int
    min_score: float
    max_chunks_per_document: int
    candidate_limit: int


class RetrievedChunkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    chunk_index: int
    score: float


class RetrievalEvalCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    query: str
    expected_source_count: int
    matched_source_count: int
    retrieved_document_ids: list[str]
    retrieved_filenames: list[str]
    retrieved_chunks: list[RetrievedChunkReport]
    document_hit: bool
    document_coverage: float
    chunk_hit: bool | None = None
    no_result: bool


class RetrievalEvalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cases: int
    cases_with_results: int
    document_hit_rate: float
    average_document_coverage: float
    chunk_expectation_case_count: int
    chunk_hit_rate: float | None = None
    failed_case_ids: list[str]


class RetrievalEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    dataset_path: str
    retrieval_config: RetrievalEvalConfig
    summary: RetrievalEvalSummary
    cases: list[RetrievalEvalCaseReport]


def load_retrieval_eval_dataset(path: str | Path) -> RetrievalEvalDataset:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"cases": payload}

    try:
        return RetrievalEvalDataset.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid retrieval eval dataset: {exc}") from exc


def build_eval_config(settings: Settings) -> RetrievalEvalConfig:
    return RetrievalEvalConfig(
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        max_chunks_per_document=settings.retrieval_max_chunks_per_document,
        candidate_limit=settings.retrieval_candidate_limit,
    )


async def evaluate_retrieval_dataset(
    cases: Sequence[RetrievalEvalCase],
    retriever: DocumentRetriever,
) -> list[RetrievalEvalCaseReport]:
    reports: list[RetrievalEvalCaseReport] = []
    for case in cases:
        chunks = await retriever.retrieve_chunks(case.query)
        reports.append(build_case_report(case, chunks))
    return reports


def build_case_report(
    case: RetrievalEvalCase,
    chunks: Sequence[RetrievedDocumentChunk],
) -> RetrievalEvalCaseReport:
    matched_source_count = sum(
        1
        for source in case.expected_sources
        if any(source_matches_chunk(source, chunk) for chunk in chunks)
    )
    chunk_hit = calculate_chunk_hit(case.expected_sources, chunks)

    return RetrievalEvalCaseReport(
        case_id=case.id,
        query=case.query,
        expected_source_count=len(case.expected_sources),
        matched_source_count=matched_source_count,
        retrieved_document_ids=unique_preserving_order(chunk.document_id for chunk in chunks),
        retrieved_filenames=unique_preserving_order(chunk.filename for chunk in chunks),
        retrieved_chunks=[
            RetrievedChunkReport(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
            )
            for chunk in chunks
        ],
        document_hit=matched_source_count >= case.minimum_expected_hits,
        document_coverage=matched_source_count / len(case.expected_sources),
        chunk_hit=chunk_hit,
        no_result=not chunks,
    )


def source_matches_chunk(
    expected_source: ExpectedRetrievalSource,
    chunk: RetrievedDocumentChunk,
) -> bool:
    return source_matches_reference(
        expected_source,
        filename=chunk.filename,
        document_id=chunk.document_id,
    )


def calculate_chunk_hit(
    expected_sources: Sequence[ExpectedRetrievalSource],
    chunks: Sequence[RetrievedDocumentChunk],
) -> bool | None:
    chunk_expectations = [source for source in expected_sources if source.chunk_indexes]
    if not chunk_expectations:
        return None

    for source in chunk_expectations:
        for chunk in chunks:
            if source_matches_chunk(source, chunk) and chunk.chunk_index in source.chunk_indexes:
                return True

    return False


def build_report(
    *,
    dataset_path: str | Path,
    retrieval_config: RetrievalEvalConfig,
    case_reports: Sequence[RetrievalEvalCaseReport],
) -> RetrievalEvalReport:
    total_cases = len(case_reports)
    chunk_case_reports = [report for report in case_reports if report.chunk_hit is not None]
    failed_case_ids = [
        report.case_id
        for report in case_reports
        if not report.document_hit or report.chunk_hit is False
    ]

    return RetrievalEvalReport(
        generated_at=datetime.now(UTC),
        dataset_path=str(Path(dataset_path)),
        retrieval_config=retrieval_config,
        summary=RetrievalEvalSummary(
            total_cases=total_cases,
            cases_with_results=sum(not report.no_result for report in case_reports),
            document_hit_rate=safe_ratio(
                sum(report.document_hit for report in case_reports),
                total_cases,
            ),
            average_document_coverage=safe_ratio(
                sum(report.document_coverage for report in case_reports),
                total_cases,
            ),
            chunk_expectation_case_count=len(chunk_case_reports),
            chunk_hit_rate=(
                safe_ratio(
                    sum(report.chunk_hit is True for report in chunk_case_reports),
                    len(chunk_case_reports),
                )
                if chunk_case_reports
                else None
            ),
            failed_case_ids=failed_case_ids,
        ),
        cases=list(case_reports),
    )


async def run_retrieval_eval(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path | None = None,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> RetrievalEvalReport:
    resolved_settings = settings or get_settings()
    retrieval_config = build_eval_config(resolved_settings)
    dataset = load_retrieval_eval_dataset(dataset_path)
    engine = create_database_engine(resolved_settings.database_url)
    session_factory = create_session_factory(engine)
    resolved_embedding_provider = embedding_provider or OpenAIEmbeddingProvider(resolved_settings)

    try:
        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            retriever = DocumentRetriever(
                repository,
                resolved_embedding_provider,
                top_k=retrieval_config.top_k,
                min_score=retrieval_config.min_score,
                max_chunks_per_document=retrieval_config.max_chunks_per_document,
                candidate_limit=retrieval_config.candidate_limit,
            )
            case_reports = await evaluate_retrieval_dataset(dataset.cases, retriever)
    finally:
        await engine.dispose()

    report = build_report(
        dataset_path=dataset_path,
        retrieval_config=retrieval_config,
        case_reports=case_reports,
    )
    if output_path is not None:
        write_report(output_path, report)
    return report


def write_report(path: str | Path, report: RetrievalEvalReport) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def format_summary(report: RetrievalEvalReport) -> str:
    summary = report.summary
    lines = [
        f"Dataset: {report.dataset_path}",
        (
            f"Retrieval config: top_k={report.retrieval_config.top_k}, "
            f"min_score={report.retrieval_config.min_score}, "
            f"candidate_limit={report.retrieval_config.candidate_limit}, "
            "max_chunks_per_document="
            f"{report.retrieval_config.max_chunks_per_document}"
        ),
        f"Cases: {summary.total_cases}",
        f"Cases with results: {summary.cases_with_results}",
        f"Document hit rate: {summary.document_hit_rate:.3f}",
        f"Average document coverage: {summary.average_document_coverage:.3f}",
    ]
    if summary.chunk_hit_rate is not None:
        lines.append(
            (
                f"Chunk hit rate: {summary.chunk_hit_rate:.3f} "
                f"({summary.chunk_expectation_case_count} cases with chunk expectations)"
            )
        )
    if summary.failed_case_ids:
        lines.append(f"Failed cases: {', '.join(summary.failed_case_ids)}")
    else:
        lines.append("Failed cases: none")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline RAG retrieval baseline.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the retrieval eval dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(
        run_retrieval_eval(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )
    print(format_summary(report))


if __name__ == "__main__":
    main()
