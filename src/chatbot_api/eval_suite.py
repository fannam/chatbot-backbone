from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from chatbot_api.chat_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_CHAT_DATASET_PATH,
)
from chatbot_api.chat_eval import (
    ChatEvalDataset,
    ChatEvalSummary,
    load_chat_eval_dataset,
    run_chat_eval,
)
from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.models import Document
from chatbot_api.rag_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_RAG_DATASET_PATH,
)
from chatbot_api.rag_eval import (
    RetrievalEvalConfig,
    RetrievalEvalSummary,
    build_eval_config,
    load_retrieval_eval_dataset,
    run_retrieval_eval,
)
from chatbot_api.settings import Settings, get_settings

DEFAULT_OUTPUT_DIR = Path(".artifacts")
DEFAULT_RAG_REPORT_FILENAME = "rag-eval-report.json"
DEFAULT_CHAT_REPORT_FILENAME = "chat-eval-report.json"
DEFAULT_SUITE_REPORT_FILENAME = "eval-suite-report.json"


class EvalSuiteThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_rag_document_hit_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_rag_chunk_hit_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_chat_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class EvalSuiteDatasets(BaseModel):
    model_config = ConfigDict(frozen=True)

    rag_dataset_path: str
    chat_dataset_path: str


class CorpusPreflightFileStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    document_count: int
    ready_count: int
    statuses: list[str]
    passed: bool


class CorpusPreflightReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_filenames: list[str]
    files: list[CorpusPreflightFileStatus]
    missing_filenames: list[str]
    not_ready_filenames: list[str]
    passed: bool


class EvalSuiteReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    datasets: EvalSuiteDatasets
    output_dir: str
    retrieval_config: RetrievalEvalConfig
    thresholds: EvalSuiteThresholds
    corpus_preflight: CorpusPreflightReport
    rag_summary: RetrievalEvalSummary | None
    chat_summary: ChatEvalSummary | None
    passed: bool
    failure_reasons: list[str]


def unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def collect_required_filenames(
    rag_dataset,
    chat_dataset: ChatEvalDataset,
) -> list[str]:
    filenames: list[str] = []

    for case in rag_dataset.cases:
        filenames.extend(
            source.filename
            for source in case.expected_sources
            if source.filename is not None
        )

    for case in chat_dataset.cases:
        filenames.extend(
            source.filename
            for source in case.expected_sources
            if source.filename is not None
        )
        if case.expected_tool_runs is None:
            continue
        for expected_tool_run in case.expected_tool_runs:
            filenames.extend(extract_filenames_from_payload(expected_tool_run.output_subset))

    return unique_preserving_order(filenames)


def extract_filenames_from_payload(payload: object) -> list[str]:
    if isinstance(payload, dict):
        filenames: list[str] = []
        for key, value in payload.items():
            if key == "filename" and isinstance(value, str):
                filenames.append(value)
                continue
            filenames.extend(extract_filenames_from_payload(value))
        return filenames

    if isinstance(payload, list):
        filenames: list[str] = []
        for item in payload:
            filenames.extend(extract_filenames_from_payload(item))
        return filenames

    return []


async def preflight_corpus(
    *,
    settings: Settings,
    required_filenames: Sequence[str],
) -> CorpusPreflightReport:
    if not required_filenames:
        return CorpusPreflightReport(
            required_filenames=[],
            files=[],
            missing_filenames=[],
            not_ready_filenames=[],
            passed=True,
        )

    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(Document.filename, Document.status).where(
                    Document.filename.in_(list(required_filenames))
                )
            )
            statuses_by_filename: dict[str, list[str]] = defaultdict(list)
            for row in result.all():
                statuses_by_filename[str(row.filename)].append(str(row.status))
    finally:
        await engine.dispose()

    files: list[CorpusPreflightFileStatus] = []
    missing_filenames: list[str] = []
    not_ready_filenames: list[str] = []

    for filename in required_filenames:
        statuses = statuses_by_filename.get(filename, [])
        ready_count = sum(status == "ready" for status in statuses)
        passed = bool(statuses) and ready_count > 0
        if not statuses:
            missing_filenames.append(filename)
        elif ready_count == 0:
            not_ready_filenames.append(filename)
        files.append(
            CorpusPreflightFileStatus(
                filename=filename,
                document_count=len(statuses),
                ready_count=ready_count,
                statuses=statuses,
                passed=passed,
            )
        )

    return CorpusPreflightReport(
        required_filenames=list(required_filenames),
        files=files,
        missing_filenames=missing_filenames,
        not_ready_filenames=not_ready_filenames,
        passed=not missing_filenames and not not_ready_filenames,
    )


def build_failure_reasons(
    *,
    thresholds: EvalSuiteThresholds,
    corpus_preflight: CorpusPreflightReport,
    rag_summary: RetrievalEvalSummary | None,
    chat_summary: ChatEvalSummary | None,
) -> list[str]:
    failures: list[str] = []

    if corpus_preflight.missing_filenames:
        failures.append(
            "corpus preflight missing required documents: "
            + ", ".join(corpus_preflight.missing_filenames)
        )
    if corpus_preflight.not_ready_filenames:
        failures.append(
            "corpus preflight found no ready document for: "
            + ", ".join(corpus_preflight.not_ready_filenames)
        )
    if rag_summary is not None:
        if rag_summary.document_hit_rate < thresholds.min_rag_document_hit_rate:
            failures.append(
                "rag document_hit_rate below threshold: "
                f"{rag_summary.document_hit_rate:.3f} < "
                f"{thresholds.min_rag_document_hit_rate:.3f}"
            )
        if (
            rag_summary.chunk_hit_rate is not None
            and rag_summary.chunk_hit_rate < thresholds.min_rag_chunk_hit_rate
        ):
            failures.append(
                "rag chunk_hit_rate below threshold: "
                f"{rag_summary.chunk_hit_rate:.3f} < "
                f"{thresholds.min_rag_chunk_hit_rate:.3f}"
            )
    if chat_summary is not None and chat_summary.pass_rate < thresholds.min_chat_pass_rate:
        failures.append(
            "chat pass_rate below threshold: "
            f"{chat_summary.pass_rate:.3f} < {thresholds.min_chat_pass_rate:.3f}"
        )

    return failures


async def run_eval_suite(
    *,
    rag_dataset_path: str | Path = DEFAULT_RAG_DATASET_PATH,
    chat_dataset_path: str | Path = DEFAULT_CHAT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_rag_document_hit_rate: float = 1.0,
    min_rag_chunk_hit_rate: float = 1.0,
    min_chat_pass_rate: float = 1.0,
    settings: Settings | None = None,
    embedding_provider=None,
) -> EvalSuiteReport:
    resolved_settings = settings or get_settings()
    rag_dataset = load_retrieval_eval_dataset(rag_dataset_path)
    chat_dataset = load_chat_eval_dataset(chat_dataset_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = EvalSuiteThresholds(
        min_rag_document_hit_rate=min_rag_document_hit_rate,
        min_rag_chunk_hit_rate=min_rag_chunk_hit_rate,
        min_chat_pass_rate=min_chat_pass_rate,
    )
    retrieval_config = build_eval_config(resolved_settings)
    required_filenames = collect_required_filenames(rag_dataset, chat_dataset)
    corpus_preflight = await preflight_corpus(
        settings=resolved_settings,
        required_filenames=required_filenames,
    )

    rag_summary: RetrievalEvalSummary | None = None
    chat_summary: ChatEvalSummary | None = None

    if corpus_preflight.passed:
        rag_report = await run_retrieval_eval(
            dataset_path=rag_dataset_path,
            output_path=resolved_output_dir / DEFAULT_RAG_REPORT_FILENAME,
            settings=resolved_settings,
            embedding_provider=embedding_provider,
        )
        rag_summary = rag_report.summary

        chat_report = await run_chat_eval(
            dataset_path=chat_dataset_path,
            output_path=resolved_output_dir / DEFAULT_CHAT_REPORT_FILENAME,
            settings=resolved_settings,
            embedding_provider=embedding_provider,
        )
        chat_summary = chat_report.summary

    failure_reasons = build_failure_reasons(
        thresholds=thresholds,
        corpus_preflight=corpus_preflight,
        rag_summary=rag_summary,
        chat_summary=chat_summary,
    )
    report = EvalSuiteReport(
        generated_at=datetime.now(UTC),
        datasets=EvalSuiteDatasets(
            rag_dataset_path=str(Path(rag_dataset_path)),
            chat_dataset_path=str(Path(chat_dataset_path)),
        ),
        output_dir=str(resolved_output_dir),
        retrieval_config=retrieval_config,
        thresholds=thresholds,
        corpus_preflight=corpus_preflight,
        rag_summary=rag_summary,
        chat_summary=chat_summary,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )
    write_report(resolved_output_dir / DEFAULT_SUITE_REPORT_FILENAME, report)
    return report


def write_report(path: str | Path, report: EvalSuiteReport) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def exit_code_for_report(report: EvalSuiteReport) -> int:
    return 0 if report.passed else 1


def format_summary(report: EvalSuiteReport) -> str:
    lines = [
        f"RAG dataset: {report.datasets.rag_dataset_path}",
        f"Chat dataset: {report.datasets.chat_dataset_path}",
        f"Output dir: {report.output_dir}",
        (
            "Corpus preflight: "
            f"{'passed' if report.corpus_preflight.passed else 'failed'} "
            f"({len(report.corpus_preflight.required_filenames)} required files)"
        ),
    ]
    if report.rag_summary is not None:
        lines.append(
            "RAG summary: "
            f"document_hit_rate={report.rag_summary.document_hit_rate:.3f}, "
            f"chunk_hit_rate="
            + (
                "n/a"
                if report.rag_summary.chunk_hit_rate is None
                else f"{report.rag_summary.chunk_hit_rate:.3f}"
            )
        )
    if report.chat_summary is not None:
        lines.append(
            "Chat summary: "
            f"pass_rate={report.chat_summary.pass_rate:.3f}, "
            f"failed_cases={len(report.chat_summary.failed_case_ids)}"
        )
    lines.append(f"Overall: {'passed' if report.passed else 'failed'}")
    if report.failure_reasons:
        lines.append("Failures: " + "; ".join(report.failure_reasons))
    else:
        lines.append("Failures: none")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the combined local eval suite.")
    parser.add_argument(
        "--rag-dataset",
        default=str(DEFAULT_RAG_DATASET_PATH),
        help="Path to the retrieval eval dataset JSON file.",
    )
    parser.add_argument(
        "--chat-dataset",
        default=str(DEFAULT_CHAT_DATASET_PATH),
        help="Path to the chat/tool eval dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write eval reports into.",
    )
    parser.add_argument(
        "--min-rag-document-hit-rate",
        type=float,
        default=1.0,
        help="Minimum allowed retrieval document hit rate.",
    )
    parser.add_argument(
        "--min-rag-chunk-hit-rate",
        type=float,
        default=1.0,
        help="Minimum allowed retrieval chunk hit rate when chunk expectations exist.",
    )
    parser.add_argument(
        "--min-chat-pass-rate",
        type=float,
        default=1.0,
        help="Minimum allowed chat/tool regression pass rate.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(
        run_eval_suite(
            rag_dataset_path=args.rag_dataset,
            chat_dataset_path=args.chat_dataset,
            output_dir=args.output_dir,
            min_rag_document_hit_rate=args.min_rag_document_hit_rate,
            min_rag_chunk_hit_rate=args.min_rag_chunk_hit_rate,
            min_chat_pass_rate=args.min_chat_pass_rate,
        )
    )
    print(format_summary(report))
    raise SystemExit(exit_code_for_report(report))


if __name__ == "__main__":
    main()
