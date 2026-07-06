from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot_api.guardrails_eval import (
    build_report,
    evaluate_guardrails_dataset,
    load_guardrails_eval_dataset,
    run_guardrails_eval,
)


def write_dataset(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_guardrails_eval_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "case-1",
                    "message": "hello",
                    "should_block": False,
                    "expected_check": "none",
                },
                {
                    "id": "case-1",
                    "message": "hello again",
                    "should_block": False,
                    "expected_check": "none",
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate guardrails eval case id"):
        load_guardrails_eval_dataset(dataset_path)


def test_load_guardrails_eval_dataset_accepts_bare_list_payload(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        [
            {
                "id": "case-1",
                "message": "hello",
                "should_block": False,
                "expected_check": "none",
            }
        ],
    )

    dataset = load_guardrails_eval_dataset(dataset_path)

    assert len(dataset.cases) == 1
    assert dataset.cases[0].id == "case-1"


@pytest.mark.anyio
async def test_evaluate_guardrails_dataset_flags_correct_and_incorrect_cases() -> None:
    dataset = load_guardrails_eval_dataset(
        Path(__file__).parent.parent / "evals" / "guardrails_jailbreak.json"
    )

    reports = await evaluate_guardrails_dataset(dataset.cases)

    assert len(reports) == len(dataset.cases)
    assert all(report.correct for report in reports)


@pytest.mark.anyio
async def test_run_guardrails_eval_reports_zero_false_positive_and_negative_rates() -> None:
    report = await run_guardrails_eval(
        dataset_path=Path(__file__).parent.parent / "evals" / "guardrails_jailbreak.json"
    )

    assert report.summary.accuracy == 1.0
    assert report.summary.false_positive_rate == 0.0
    assert report.summary.false_negative_rate == 0.0
    assert report.summary.failed_case_ids == []


def test_build_report_computes_false_positive_and_negative_rates() -> None:
    from chatbot_api.guardrails_eval import GuardrailsEvalCaseReport

    case_reports = [
        GuardrailsEvalCaseReport(
            case_id="should-block-but-passed",
            message="msg",
            should_block=True,
            expected_check="jailbreak",
            blocked=False,
            correct=False,
        ),
        GuardrailsEvalCaseReport(
            case_id="should-not-block-but-blocked",
            message="msg",
            should_block=False,
            expected_check="none",
            blocked=True,
            correct=False,
        ),
        GuardrailsEvalCaseReport(
            case_id="correct-case",
            message="msg",
            should_block=True,
            expected_check="jailbreak",
            blocked=True,
            correct=True,
        ),
    ]

    report = build_report(dataset_path="evals/guardrails_jailbreak.json", case_reports=case_reports)

    assert report.summary.total_cases == 3
    assert report.summary.false_negative_rate == 0.5
    assert report.summary.false_positive_rate == 1.0
    assert set(report.summary.failed_case_ids) == {
        "should-block-but-passed",
        "should-not-block-but-blocked",
    }
