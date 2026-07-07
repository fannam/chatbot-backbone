from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.evals.chat_eval import (
    ChatEvalCase,
    ChatEvalHistoryTurn,
    ScriptedEvalProvider,
    ScriptedEvalProviderError,
    load_chat_eval_dataset,
    run_chat_eval,
    seed_history,
)
from chatbot_api.models import Base
from chatbot_api.providers import ChatTurn, ToolCallBatch, ToolResultMessage
from chatbot_api.repositories import SqlAlchemyChatRepository, SqlAlchemyDocumentRepository
from chatbot_api.retrieval import DocumentChunkCreate
from chatbot_api.settings import Settings


class StubEmbeddingProvider:
    def __init__(self, embeddings_by_text: dict[str, list[float]]) -> None:
        self._embeddings_by_text = embeddings_by_text
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = list(texts)
        self.calls.append(normalized)
        return [self._embeddings_by_text[text] for text in normalized]


def write_dataset(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_chat_eval_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "duplicate",
                    "message": "first",
                    "script": [
                        {
                            "type": "final_answer",
                            "response_id": "resp-1",
                            "content": "done",
                        }
                    ],
                },
                {
                    "id": "duplicate",
                    "message": "second",
                    "script": [
                        {
                            "type": "final_answer",
                            "response_id": "resp-2",
                            "content": "done",
                        }
                    ],
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate chat eval case id"):
        load_chat_eval_dataset(dataset_path)


def test_load_chat_eval_dataset_rejects_invalid_script_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "bad-script",
                    "message": "hello",
                    "script": [
                        {
                            "type": "final_answer",
                            "response_id": "resp-1",
                            "content": "too early",
                        },
                        {
                            "type": "tool_call_batch",
                            "response_id": "resp-2",
                            "tool_calls": [
                                {
                                    "call_id": "tool-1",
                                    "name": "calculator",
                                    "arguments": {"expression": "2 + 2"},
                                }
                            ],
                        },
                    ],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="only the final script step may be final_answer"):
        load_chat_eval_dataset(dataset_path)


@pytest.mark.anyio
async def test_scripted_eval_provider_validates_previous_response_id_and_tool_outputs() -> None:
    case = ChatEvalCase.model_validate(
        {
            "id": "provider-sequencing",
            "message": "2 + 2?",
            "script": [
                {
                    "type": "tool_call_batch",
                    "response_id": "resp-1",
                    "tool_calls": [
                        {
                            "call_id": "tool-1",
                            "name": "calculator",
                            "arguments": {"expression": "2 + 2"},
                        }
                    ],
                },
                {
                    "type": "final_answer",
                    "response_id": "resp-2",
                    "content": "The answer is 4.",
                },
            ],
        }
    )

    provider = ScriptedEvalProvider(case.script)
    result = await provider.generate_response([ChatTurn(role="user", content="2 + 2?")])

    assert isinstance(result, ToolCallBatch)

    with pytest.raises(ScriptedEvalProviderError, match="unexpected previous_response_id"):
        await provider.generate_response(
            [ChatTurn(role="user", content="2 + 2?")],
            previous_response_id=None,
            tool_outputs=[ToolResultMessage(call_id="tool-1", output='{"status":"completed"}')],
        )

    provider = ScriptedEvalProvider(case.script)
    await provider.generate_response([ChatTurn(role="user", content="2 + 2?")])

    with pytest.raises(ScriptedEvalProviderError, match="unexpected tool output count"):
        await provider.generate_response(
            [ChatTurn(role="user", content="2 + 2?")],
            previous_response_id="resp-1",
            tool_outputs=[],
        )


@pytest.mark.anyio
async def test_seed_history_persists_turn_order(tmp_path: Path) -> None:
    database_path = tmp_path / "chat-eval-history.db"
    engine = create_database_engine(f"sqlite+aiosqlite:///{database_path}")
    session_factory = create_session_factory(engine)

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            await seed_history(
                session,
                conversation_id="conv-history",
                history=[
                    ChatEvalHistoryTurn(role="user", content="Earlier question"),
                    ChatEvalHistoryTurn(role="assistant", content="Earlier answer"),
                ],
            )
            repository = SqlAlchemyChatRepository(session)
            messages = await repository.list_messages("conv-history")
    finally:
        await engine.dispose()

    assert messages == [
        ChatTurn(role="user", content="Earlier question"),
        ChatTurn(role="assistant", content="Earlier answer"),
    ]


@pytest.mark.anyio
async def test_run_chat_eval_writes_report_and_summarizes_pass_fail(tmp_path: Path) -> None:
    database_path = tmp_path / "chat-eval.db"
    dataset_path = tmp_path / "dataset.json"
    output_path = tmp_path / "report.json"
    database_url = f"sqlite+aiosqlite:///{database_path}"

    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "search-tool-pass",
                    "message": "What does the product overview say?",
                    "script": [
                        {
                            "type": "tool_call_batch",
                            "response_id": "resp-1",
                            "usage": {
                                "input_tokens": 50,
                                "output_tokens": 10,
                                "total_tokens": 60,
                            },
                            "tool_calls": [
                                {
                                    "call_id": "tool-1",
                                    "name": "search_knowledge_base",
                                    "arguments": {"query": "architecture query", "top_k": 1},
                                }
                            ],
                        },
                        {
                            "type": "final_answer",
                            "response_id": "resp-2",
                            "content": "The architecture uses FastAPI and LangGraph.",
                            "usage": {
                                "input_tokens": 20,
                                "output_tokens": 10,
                                "total_tokens": 30,
                            },
                        },
                    ],
                    "expected_tool_runs": [
                        {
                            "tool_name": "search_knowledge_base",
                            "status": "completed",
                            "input_subset": {"query": "architecture query", "top_k": 1},
                            "output_subset": {
                                "hits": [
                                    {"filename": "product_overview.md", "chunk_index": 0}
                                ]
                            },
                        }
                    ],
                    "expected_sources": [
                        {"filename": "product_overview.md", "chunk_indexes": [0]}
                    ],
                    "minimum_expected_source_hits": 1,
                    "expected_answer_substrings": ["FastAPI", "LangGraph"],
                },
                {
                    "id": "rejected-tool-answer-mismatch",
                    "message": "What is the weather?",
                    "script": [
                        {
                            "type": "tool_call_batch",
                            "response_id": "resp-3",
                            "usage": {
                                "input_tokens": 30,
                                "output_tokens": 5,
                                "total_tokens": 35,
                            },
                            "tool_calls": [
                                {
                                    "call_id": "tool-2",
                                    "name": "weather_lookup",
                                    "arguments": {"city": "San Francisco"},
                                }
                            ],
                        },
                        {
                            "type": "final_answer",
                            "response_id": "resp-4",
                            "content": "I cannot look up live weather here.",
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            },
                        },
                    ],
                    "expected_tool_runs": [
                        {
                            "tool_name": "weather_lookup",
                            "status": "rejected",
                            "input_subset": {"city": "San Francisco"},
                        }
                    ],
                    "expected_answer_substrings": ["sunny"],
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
        tool_search_top_k=2,
    )
    embedding_provider = StubEmbeddingProvider({"architecture query": [1.0, 0.0]})

    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            await repository.create_document(
                document_id="doc-overview",
                filename="product_overview.md",
                content_type="text/markdown",
                byte_size=100,
                checksum_sha256="hash-overview",
                status="ready",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="The chatbot API uses FastAPI and LangGraph.",
                        embedding=[1.0, 0.0],
                        start_offset=0,
                        end_offset=45,
                    )
                ],
            )
            await repository.create_document(
                document_id="doc-billing",
                filename="billing_faq.md",
                content_type="text/markdown",
                byte_size=80,
                checksum_sha256="hash-billing",
                status="ready",
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="Billing answers live here.",
                        embedding=[0.0, 1.0],
                        start_offset=0,
                        end_offset=26,
                    )
                ],
            )

        report = await run_chat_eval(
            dataset_path=dataset_path,
            output_path=output_path,
            settings=settings,
            embedding_provider=embedding_provider,
        )
    finally:
        await engine.dispose()

    assert report.summary.total_cases == 2
    assert report.summary.passed_cases == 1
    assert report.summary.pass_rate == 0.5
    assert report.summary.failed_case_ids == ["rejected-tool-answer-mismatch"]
    assert report.summary.tool_expectation_case_count == 2
    assert report.summary.tool_match_rate == 1.0
    assert report.summary.source_expectation_case_count == 1
    assert report.summary.source_match_rate == 1.0
    assert report.summary.answer_expectation_case_count == 2
    assert report.summary.answer_match_rate == 0.5
    assert embedding_provider.calls == [["architecture query"]]

    first_case, second_case = report.cases
    assert first_case.case_id == "search-tool-pass"
    assert first_case.passed is True
    assert first_case.tool_runs_match is True
    assert first_case.source_match is True
    assert first_case.answer_match is True
    assert first_case.matched_source_count == 1
    assert first_case.usage is not None
    assert first_case.usage.total_tokens == 90
    assert first_case.citations[0].filename == "product_overview.md"

    assert second_case.case_id == "rejected-tool-answer-mismatch"
    assert second_case.passed is False
    assert second_case.tool_runs[0].status == "rejected"
    assert second_case.tool_runs_match is True
    assert second_case.answer_match is False
    assert "missing expected answer substring: sunny" in second_case.failure_reasons

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["passed_cases"] == 1
    assert payload["cases"][0]["case_id"] == "search-tool-pass"


@pytest.mark.anyio
async def test_run_chat_eval_supports_current_user_profile_tool_cases(tmp_path: Path) -> None:
    database_path = tmp_path / "chat-eval-profile.db"
    dataset_path = tmp_path / "dataset-profile.json"
    database_url = f"sqlite+aiosqlite:///{database_path}"

    write_dataset(
        dataset_path,
        {
            "cases": [
                {
                    "id": "profile-found",
                    "message": "What plan am I on?",
                    "request_metadata": {
                        "user_profile": {
                            "user_id": "user-123",
                            "display_name": "Alice",
                            "plan": "pro",
                            "preferences": {"timezone": "UTC"},
                        }
                    },
                    "script": [
                        {
                            "type": "tool_call_batch",
                            "response_id": "resp-profile-1",
                            "tool_calls": [
                                {
                                    "call_id": "tool-profile-1",
                                    "name": "get_current_user_profile",
                                    "arguments": {},
                                }
                            ],
                        },
                        {
                            "type": "final_answer",
                            "response_id": "resp-profile-2",
                            "content": "You are on the pro plan.",
                        },
                    ],
                    "expected_tool_runs": [
                        {
                            "tool_name": "get_current_user_profile",
                            "status": "completed",
                            "output_subset": {
                                "found": True,
                                "profile": {
                                    "user_id": "user-123",
                                    "plan": "pro",
                                },
                            },
                        }
                    ],
                    "expected_answer_substrings": ["pro plan"],
                },
                {
                    "id": "profile-missing",
                    "message": "What is my locale?",
                    "script": [
                        {
                            "type": "tool_call_batch",
                            "response_id": "resp-profile-3",
                            "tool_calls": [
                                {
                                    "call_id": "tool-profile-2",
                                    "name": "get_current_user_profile",
                                    "arguments": {},
                                }
                            ],
                        },
                        {
                            "type": "final_answer",
                            "response_id": "resp-profile-4",
                            "content": "I do not have your profile in this request.",
                        },
                    ],
                    "expected_tool_runs": [
                        {
                            "tool_name": "get_current_user_profile",
                            "status": "completed",
                            "output_subset": {
                                "found": False,
                                "profile": None,
                            },
                        }
                    ],
                    "expected_answer_substrings": ["do not have your profile"],
                },
            ]
        },
    )

    settings = Settings(
        database_url=database_url,
        langgraph_checkpoint_database_url=None,
    )
    embedding_provider = StubEmbeddingProvider({})

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        report = await run_chat_eval(
            dataset_path=dataset_path,
            settings=settings,
            embedding_provider=embedding_provider,
        )
    finally:
        await engine.dispose()

    assert report.summary.total_cases == 2
    assert report.summary.passed_cases == 2
    assert report.summary.pass_rate == 1.0
    assert report.summary.failed_case_ids == []
    assert embedding_provider.calls == []
    assert [case.case_id for case in report.cases] == ["profile-found", "profile-missing"]
    assert report.cases[0].tool_runs[0].tool_name == "get_current_user_profile"
    assert report.cases[0].tool_runs[0].output == {
        "found": True,
        "profile": {
            "user_id": "user-123",
            "display_name": "Alice",
            "email": None,
            "plan": "pro",
            "locale": None,
            "preferences": {"timezone": "UTC"},
        },
    }
    assert report.cases[1].tool_runs[0].output == {"found": False, "profile": None}
