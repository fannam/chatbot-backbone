from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from chatbot_api.evals.eval_common import safe_ratio, write_report
from chatbot_api.workflow import GuardrailsValidationError, build_input_guard

DEFAULT_DATASET_PATH = Path("evals/guardrails_jailbreak.json")


class GuardrailsEvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    message: str
    should_block: bool
    expected_check: Literal["jailbreak", "pii", "none"]
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case id must not be blank")
        return normalized

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None


class GuardrailsEvalDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[GuardrailsEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> GuardrailsEvalDataset:
        seen_case_ids: set[str] = set()
        for case in self.cases:
            if case.id in seen_case_ids:
                raise ValueError(f"duplicate guardrails eval case id: {case.id}")
            seen_case_ids.add(case.id)
        return self


class GuardrailsEvalCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    message: str
    should_block: bool
    expected_check: str
    blocked: bool
    correct: bool
    notes: str | None = None


class GuardrailsEvalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cases: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    failed_case_ids: list[str]


class GuardrailsEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    dataset_path: str
    summary: GuardrailsEvalSummary
    cases: list[GuardrailsEvalCaseReport]


def load_guardrails_eval_dataset(path: str | Path) -> GuardrailsEvalDataset:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"cases": payload}

    try:
        return GuardrailsEvalDataset.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid guardrails eval dataset: {exc}") from exc


async def evaluate_guardrails_dataset(
    cases: Sequence[GuardrailsEvalCase],
) -> list[GuardrailsEvalCaseReport]:
    guard = build_input_guard(jailbreak_detection_enabled=True, pii_detection_enabled=True)
    reports: list[GuardrailsEvalCaseReport] = []
    for case in cases:
        blocked = False
        if guard is not None:
            try:
                await guard.validate(case.message)
            except GuardrailsValidationError:
                blocked = True

        reports.append(
            GuardrailsEvalCaseReport(
                case_id=case.id,
                message=case.message,
                should_block=case.should_block,
                expected_check=case.expected_check,
                blocked=blocked,
                correct=blocked == case.should_block,
                notes=case.notes,
            )
        )
    return reports


def build_report(
    *,
    dataset_path: str | Path,
    case_reports: Sequence[GuardrailsEvalCaseReport],
) -> GuardrailsEvalReport:
    total_cases = len(case_reports)
    correct_count = sum(report.correct for report in case_reports)
    should_block_count = sum(report.should_block for report in case_reports)
    should_not_block_count = total_cases - should_block_count
    false_positive_count = sum(
        1 for report in case_reports if not report.should_block and report.blocked
    )
    false_negative_count = sum(
        1 for report in case_reports if report.should_block and not report.blocked
    )
    failed_case_ids = [report.case_id for report in case_reports if not report.correct]

    return GuardrailsEvalReport(
        generated_at=datetime.now(UTC),
        dataset_path=str(Path(dataset_path)),
        summary=GuardrailsEvalSummary(
            total_cases=total_cases,
            accuracy=safe_ratio(correct_count, total_cases),
            false_positive_rate=safe_ratio(false_positive_count, should_not_block_count),
            false_negative_rate=safe_ratio(false_negative_count, should_block_count),
            failed_case_ids=failed_case_ids,
        ),
        cases=list(case_reports),
    )


async def run_guardrails_eval(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path | None = None,
) -> GuardrailsEvalReport:
    dataset = load_guardrails_eval_dataset(dataset_path)
    case_reports = await evaluate_guardrails_dataset(dataset.cases)
    report = build_report(dataset_path=dataset_path, case_reports=case_reports)
    if output_path is not None:
        write_report(output_path, report)
    return report


def format_summary(report: GuardrailsEvalReport) -> str:
    summary = report.summary
    lines = [
        f"Dataset: {report.dataset_path}",
        f"Cases: {summary.total_cases}",
        f"Accuracy: {summary.accuracy:.3f}",
        f"False positive rate: {summary.false_positive_rate:.3f}",
        f"False negative rate: {summary.false_negative_rate:.3f}",
    ]
    if summary.failed_case_ids:
        lines.append(f"Failed cases: {', '.join(summary.failed_case_ids)}")
    else:
        lines.append("Failed cases: none")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline guardrails jailbreak/PII regression eval."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the guardrails eval dataset JSON file.",
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
        run_guardrails_eval(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )
    print(format_summary(report))


if __name__ == "__main__":
    main()
