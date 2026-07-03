from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.document_ingestion import DocumentChunkCreate
from chatbot_api.eval_suite import exit_code_for_report, run_eval_suite
from chatbot_api.models import Base
from chatbot_api.repositories import SqlAlchemyDocumentRepository
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


async def seed_document(
    *,
    database_url: str,
    document_id: str,
    filename: str,
    status: str,
    embedding: list[float] | None,
) -> None:
    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            repository = SqlAlchemyDocumentRepository(session)
            await repository.create_document(
                document_id=document_id,
                filename=filename,
                content_type="text/markdown",
                byte_size=100,
                checksum_sha256=f"hash-{document_id}",
                status=status,
                failure_reason=None,
                chunks=[
                    DocumentChunkCreate(
                        chunk_index=0,
                        content="Guide content for evaluation.",
                        embedding=embedding,
                        start_offset=0,
                        end_offset=29,
                    )
                ],
            )
    finally:
        await engine.dispose()


def build_rag_dataset() -> dict[str, object]:
    return {
        "cases": [
            {
                "id": "guide-hit",
                "query": "guide question",
                "expected_sources": [{"filename": "guide.md", "chunk_indexes": [0]}],
            }
        ]
    }


def build_chat_dataset(*, answer_text: str, expected_answer_substring: str) -> dict[str, object]:
    return {
        "cases": [
            {
                "id": "guide-search",
                "message": "What does the guide say?",
                "script": [
                    {
                        "type": "tool_call_batch",
                        "response_id": "resp-1",
                        "tool_calls": [
                            {
                                "call_id": "tool-1",
                                "name": "search_knowledge_base",
                                "arguments": {"query": "guide question", "top_k": 1},
                            }
                        ],
                    },
                    {
                        "type": "final_answer",
                        "response_id": "resp-2",
                        "content": answer_text,
                    },
                ],
                "expected_tool_runs": [
                    {
                        "tool_name": "search_knowledge_base",
                        "status": "completed",
                        "output_subset": {
                            "hits": [{"filename": "guide.md", "chunk_index": 0}]
                        },
                    }
                ],
                "expected_sources": [{"filename": "guide.md", "chunk_indexes": [0]}],
                "minimum_expected_source_hits": 1,
                "expected_answer_substrings": [expected_answer_substring],
            }
        ]
    }


def build_memory_dataset(
    *,
    answer_text: str,
    expected_answer_substring: str,
) -> dict[str, object]:
    return {
        "cases": [
            {
                "id": "memory-rule-case",
                "message": "Call me Alice.",
                "request_metadata": {"user_profile": {"user_id": "user-memory-1"}},
                "script": [
                    {
                        "kind": "chat",
                        "response_id": "memory-resp-1",
                        "content": answer_text,
                        "expected_prompt_substrings": ["user: Call me Alice."],
                    },
                    {
                        "kind": "memory_extraction",
                        "response_id": "memory-resp-2",
                        "content": "{\"memories\":[]}",
                    },
                ],
                "expected_answer_substrings": [expected_answer_substring],
                "expect_summary_present": False,
                "expected_active_memory_count": 1,
                "expected_active_memories": [
                    {
                        "key": "profile.preferred_name",
                        "kind": "profile",
                        "extraction_method": "rule",
                        "value_subset": {"value": "Alice"},
                    }
                ],
            }
        ]
    }


@pytest.mark.anyio
async def test_run_eval_suite_fails_preflight_when_required_document_is_missing(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    rag_dataset_path = tmp_path / "rag.json"
    chat_dataset_path = tmp_path / "chat.json"
    memory_dataset_path = tmp_path / "memory.json"

    write_dataset(rag_dataset_path, build_rag_dataset())
    write_dataset(
        chat_dataset_path,
        build_chat_dataset(
            answer_text="The guide says hello.",
            expected_answer_substring="hello",
        ),
    )
    write_dataset(
        memory_dataset_path,
        build_memory_dataset(
            answer_text="I will call you Alice.",
            expected_answer_substring="Alice",
        ),
    )

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        report = await run_eval_suite(
            rag_dataset_path=rag_dataset_path,
            chat_dataset_path=chat_dataset_path,
            memory_dataset_path=memory_dataset_path,
            output_dir=tmp_path / "artifacts-missing",
            settings=Settings(
                database_url=database_url,
                langgraph_checkpoint_database_url=None,
            ),
            embedding_provider=StubEmbeddingProvider({"guide question": [1.0, 0.0]}),
        )
    finally:
        await engine.dispose()

    assert report.passed is False
    assert report.corpus_preflight.passed is False
    assert report.corpus_preflight.missing_filenames == ["guide.md"]
    assert report.rag_summary is None
    assert report.chat_summary is None
    assert report.memory_summary is None
    assert exit_code_for_report(report) == 1
    payload = json.loads(
        (tmp_path / "artifacts-missing" / "eval-suite-report.json").read_text(encoding="utf-8")
    )
    assert payload["passed"] is False
    assert payload["rag_summary"] is None
    assert payload["chat_summary"] is None
    assert payload["memory_summary"] is None


@pytest.mark.anyio
async def test_run_eval_suite_writes_all_reports_on_success(tmp_path: Path) -> None:
    database_path = tmp_path / "success.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    rag_dataset_path = tmp_path / "rag.json"
    chat_dataset_path = tmp_path / "chat.json"
    memory_dataset_path = tmp_path / "memory.json"
    output_dir = tmp_path / "artifacts-success"

    write_dataset(rag_dataset_path, build_rag_dataset())
    write_dataset(
        chat_dataset_path,
        build_chat_dataset(
            answer_text="The guide says hello.",
            expected_answer_substring="hello",
        ),
    )
    write_dataset(
        memory_dataset_path,
        build_memory_dataset(
            answer_text="I will call you Alice.",
            expected_answer_substring="Alice",
        ),
    )

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await seed_document(
            database_url=database_url,
            document_id="doc-guide",
            filename="guide.md",
            status="ready",
            embedding=[1.0, 0.0],
        )

        embedding_provider = StubEmbeddingProvider({"guide question": [1.0, 0.0]})
        report = await run_eval_suite(
            rag_dataset_path=rag_dataset_path,
            chat_dataset_path=chat_dataset_path,
            memory_dataset_path=memory_dataset_path,
            output_dir=output_dir,
            settings=Settings(
                database_url=database_url,
                langgraph_checkpoint_database_url=None,
                retrieval_top_k=1,
                retrieval_min_score=0.3,
                retrieval_max_chunks_per_document=1,
                retrieval_candidate_limit=2,
                tool_search_top_k=1,
            ),
            embedding_provider=embedding_provider,
        )
    finally:
        await engine.dispose()

    assert report.passed is True
    assert report.corpus_preflight.passed is True
    assert report.rag_summary is not None
    assert report.rag_summary.document_hit_rate == 1.0
    assert report.chat_summary is not None
    assert report.chat_summary.pass_rate == 1.0
    assert report.memory_summary is not None
    assert report.memory_summary.pass_rate == 1.0
    assert exit_code_for_report(report) == 0
    assert embedding_provider.calls == [["guide question"], ["guide question"]]
    assert (output_dir / "rag-eval-report.json").exists()
    assert (output_dir / "chat-eval-report.json").exists()
    assert (output_dir / "memory-eval-report.json").exists()
    assert (output_dir / "eval-suite-report.json").exists()


@pytest.mark.anyio
async def test_run_eval_suite_fails_quality_gate_when_chat_eval_fails(tmp_path: Path) -> None:
    database_path = tmp_path / "chat-fail.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    rag_dataset_path = tmp_path / "rag.json"
    chat_dataset_path = tmp_path / "chat.json"
    memory_dataset_path = tmp_path / "memory.json"

    write_dataset(rag_dataset_path, build_rag_dataset())
    write_dataset(
        chat_dataset_path,
        build_chat_dataset(
            answer_text="The guide says hello.",
            expected_answer_substring="goodbye",
        ),
    )
    write_dataset(
        memory_dataset_path,
        build_memory_dataset(
            answer_text="I will call you Alice.",
            expected_answer_substring="Alice",
        ),
    )

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await seed_document(
            database_url=database_url,
            document_id="doc-guide",
            filename="guide.md",
            status="ready",
            embedding=[1.0, 0.0],
        )

        report = await run_eval_suite(
            rag_dataset_path=rag_dataset_path,
            chat_dataset_path=chat_dataset_path,
            memory_dataset_path=memory_dataset_path,
            output_dir=tmp_path / "artifacts-chat-fail",
            settings=Settings(
                database_url=database_url,
                langgraph_checkpoint_database_url=None,
                retrieval_top_k=1,
                retrieval_min_score=0.3,
                retrieval_max_chunks_per_document=1,
                retrieval_candidate_limit=2,
                tool_search_top_k=1,
            ),
            embedding_provider=StubEmbeddingProvider({"guide question": [1.0, 0.0]}),
        )
    finally:
        await engine.dispose()

    assert report.passed is False
    assert report.chat_summary is not None
    assert report.chat_summary.pass_rate == 0.0
    assert report.memory_summary is not None
    assert report.memory_summary.pass_rate == 1.0
    assert any("chat pass_rate below threshold" in item for item in report.failure_reasons)
    assert exit_code_for_report(report) == 1


@pytest.mark.anyio
async def test_run_eval_suite_fails_quality_gate_when_memory_eval_fails(tmp_path: Path) -> None:
    database_path = tmp_path / "memory-fail.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    rag_dataset_path = tmp_path / "rag.json"
    chat_dataset_path = tmp_path / "chat.json"
    memory_dataset_path = tmp_path / "memory.json"

    write_dataset(rag_dataset_path, build_rag_dataset())
    write_dataset(
        chat_dataset_path,
        build_chat_dataset(
            answer_text="The guide says hello.",
            expected_answer_substring="hello",
        ),
    )
    write_dataset(
        memory_dataset_path,
        build_memory_dataset(
            answer_text="I will call you Alice.",
            expected_answer_substring="Bob",
        ),
    )

    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await seed_document(
            database_url=database_url,
            document_id="doc-guide",
            filename="guide.md",
            status="ready",
            embedding=[1.0, 0.0],
        )

        report = await run_eval_suite(
            rag_dataset_path=rag_dataset_path,
            chat_dataset_path=chat_dataset_path,
            memory_dataset_path=memory_dataset_path,
            output_dir=tmp_path / "artifacts-memory-fail",
            settings=Settings(
                database_url=database_url,
                langgraph_checkpoint_database_url=None,
                retrieval_top_k=1,
                retrieval_min_score=0.3,
                retrieval_max_chunks_per_document=1,
                retrieval_candidate_limit=2,
                tool_search_top_k=1,
            ),
            embedding_provider=StubEmbeddingProvider({"guide question": [1.0, 0.0]}),
        )
    finally:
        await engine.dispose()

    assert report.passed is False
    assert report.chat_summary is not None
    assert report.chat_summary.pass_rate == 1.0
    assert report.memory_summary is not None
    assert report.memory_summary.pass_rate == 0.0
    assert any("memory pass_rate below threshold" in item for item in report.failure_reasons)
    assert exit_code_for_report(report) == 1
