from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot_api.database import create_database_engine
from chatbot_api.memory_eval import load_memory_eval_dataset, run_memory_eval
from chatbot_api.models import Base
from chatbot_api.settings import Settings


def write_dataset(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_memory_eval_dataset_rejects_invalid_script_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "memory-dataset.json"
    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "bad-order",
                    "message": "hello",
                    "script": [
                        {
                            "kind": "summary",
                            "response_id": "resp-1",
                            "content": "bad",
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="first script step must be chat"):
        load_memory_eval_dataset(dataset_path)


@pytest.mark.anyio
async def test_run_memory_eval_writes_report_and_summarizes_pass_fail(tmp_path: Path) -> None:
    database_path = tmp_path / "memory-eval.db"
    dataset_path = tmp_path / "memory-dataset.json"
    output_path = tmp_path / "memory-report.json"
    database_url = f"sqlite+aiosqlite:///{database_path}"

    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "prompt-and-memory-pass",
                    "message": "What should I do next?",
                    "request_metadata": {"user_profile": {"user_id": "user-123"}},
                    "history": [
                        {"role": "user", "content": "Old question"},
                        {"role": "assistant", "content": "Old answer"},
                        {"role": "user", "content": "Recent question"},
                    ],
                    "seeded_summary": {
                        "summary_text": "User prefers concise answers.",
                        "last_summarized_message_id": 2,
                    },
                    "seeded_memories": [
                        {
                            "kind": "preference",
                            "key": "preferences.language",
                            "value_json": {"value": "Vietnamese"},
                        }
                    ],
                    "script": [
                        {
                            "kind": "chat",
                            "response_id": "resp-1",
                            "content": "Continue from the recent question.",
                            "expected_prompt_substrings": [
                                "Conversation summary:",
                                "User prefers concise answers.",
                                "Stored user memory:",
                                "Preferred language: Vietnamese",
                                "user: Recent question",
                                "user: What should I do next?",
                            ],
                            "forbidden_prompt_substrings": [
                                "user: Old question",
                                "assistant: Old answer",
                            ],
                        },
                        {
                            "kind": "memory_extraction",
                            "response_id": "resp-2",
                            "content": "{\"memories\":[]}",
                        },
                    ],
                    "expected_answer_substrings": ["recent question"],
                    "expect_summary_present": True,
                    "expected_summary": {
                        "exact_text": "User prefers concise answers.",
                        "last_summarized_message_id": 2,
                    },
                    "expected_active_memory_count": 1,
                    "expected_active_memories": [
                        {
                            "key": "preferences.language",
                            "kind": "preference",
                            "extraction_method": "rule",
                            "value_subset": {"value": "Vietnamese"},
                        }
                    ],
                },
                {
                    "id": "answer-mismatch-fails",
                    "message": "Call me Bob.",
                    "request_metadata": {"user_profile": {"user_id": "user-456"}},
                    "script": [
                        {
                            "kind": "chat",
                            "response_id": "resp-3",
                            "content": "I will call you Bob.",
                        },
                        {
                            "kind": "memory_extraction",
                            "response_id": "resp-4",
                            "content": "{\"memories\":[]}",
                        },
                    ],
                    "expected_answer_substrings": ["Alice"],
                    "expect_summary_present": False,
                    "expected_active_memory_count": 1,
                    "expected_active_memories": [
                        {
                            "key": "profile.preferred_name",
                            "kind": "profile",
                            "extraction_method": "rule",
                            "value_subset": {"value": "Bob"},
                        }
                    ],
                },
            ]
        },
    )

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        report = await run_memory_eval(
            dataset_path=dataset_path,
            output_path=output_path,
            settings=Settings(
                database_url=database_url,
                langgraph_checkpoint_database_url=None,
            ),
        )
    finally:
        await engine.dispose()

    assert report.summary.total_cases == 2
    assert report.summary.passed_cases == 1
    assert report.summary.pass_rate == 0.5
    assert report.summary.failed_case_ids == ["answer-mismatch-fails"]
    assert report.summary.prompt_expectation_case_count == 1
    assert report.summary.prompt_match_rate == 1.0
    assert report.summary.summary_expectation_case_count == 2
    assert report.summary.summary_match_rate == 1.0
    assert report.summary.memory_expectation_case_count == 2
    assert report.summary.memory_match_rate == 1.0
    assert report.summary.answer_expectation_case_count == 2
    assert report.summary.answer_match_rate == 0.5

    first_case, second_case = report.cases
    assert first_case.case_id == "prompt-and-memory-pass"
    assert first_case.passed is True
    assert first_case.prompt_match is True
    assert first_case.active_memories[0].key == "preferences.language"
    assert first_case.provider_calls[0].kind == "chat"
    assert first_case.provider_calls[0].messages[0].role == "system"

    assert second_case.case_id == "answer-mismatch-fails"
    assert second_case.passed is False
    assert second_case.memory_match is True
    assert second_case.answer_match is False
    assert "missing expected answer substring: Alice" in second_case.failure_reasons

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["passed_cases"] == 1
    assert payload["cases"][0]["case_id"] == "prompt-and-memory-pass"


@pytest.mark.anyio
async def test_run_memory_eval_supports_summary_refresh_cases(tmp_path: Path) -> None:
    database_path = tmp_path / "memory-summary.db"
    dataset_path = tmp_path / "memory-summary-dataset.json"
    database_url = f"sqlite+aiosqlite:///{database_path}"

    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "summary-refresh",
                    "message": "Please summarize where we landed.",
                    "settings_overrides": {
                        "memory_summary_trigger_messages": 4,
                        "memory_recent_message_window": 2,
                    },
                    "history": [
                        {"role": "user", "content": "We need a deployment plan."},
                        {
                            "role": "assistant",
                            "content": "I will gather the deployment requirements.",
                        },
                        {"role": "user", "content": "The rollout must stay low risk."},
                        {
                            "role": "assistant",
                            "content": "Understood. I will stage it gradually.",
                        },
                        {"role": "user", "content": "Please also document the rollback."},
                        {"role": "assistant", "content": "I will add rollback steps."},
                    ],
                    "script": [
                        {
                            "kind": "chat",
                            "response_id": "resp-1",
                            "content": "We landed on a low-risk rollout with rollback steps.",
                        },
                        {
                            "kind": "summary",
                            "response_id": "resp-2",
                            "content": (
                                "The conversation is about a low-risk deployment plan "
                                "that also needs rollback documentation."
                            ),
                            "expected_prompt_substrings": [
                                "Update the conversation summary.",
                                "No previous summary.",
                                "user: We need a deployment plan.",
                                "assistant: I will add rollback steps.",
                            ],
                            "forbidden_prompt_substrings": [
                                "user: Please summarize where we landed."
                            ],
                        },
                    ],
                    "expected_answer_substrings": ["low-risk rollout"],
                    "expect_summary_present": True,
                    "expected_summary": {
                        "exact_text": (
                            "The conversation is about a low-risk deployment plan "
                            "that also needs rollback documentation."
                        ),
                        "last_summarized_message_id": 6,
                    },
                    "expected_active_memory_count": 0,
                }
            ]
        },
    )

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        report = await run_memory_eval(
            dataset_path=dataset_path,
            settings=Settings(
                database_url=database_url,
                langgraph_checkpoint_database_url=None,
            ),
        )
    finally:
        await engine.dispose()

    assert report.summary.total_cases == 1
    assert report.summary.passed_cases == 1
    assert report.summary.pass_rate == 1.0
    assert report.cases[0].summary is not None
    assert report.cases[0].summary.last_summarized_message_id == 6
    assert report.cases[0].provider_calls[1].kind == "summary"
