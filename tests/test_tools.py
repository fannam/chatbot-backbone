from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from chatbot_api.providers import ToolCallRequest
from chatbot_api.repositories import RetrievedDocumentChunk
from chatbot_api.workflow.tools import (
    RegisteredTool,
    ToolExecutionContext,
    ToolRegistry,
    build_tool_registry,
    evaluate_expression,
)


class StubRetriever:
    def __init__(self, chunks: list[RetrievedDocumentChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, int, int, str | None]] = []

    async def retrieve_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chunks_per_document: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        self.calls.append((query, top_k or 0, max_chunks_per_document or 0, owner_user_id))
        return self._chunks[: top_k or len(self._chunks)]


def make_chunk(
    *,
    document_id: str,
    chunk_index: int,
    content: str,
    score: float,
) -> RetrievedDocumentChunk:
    return RetrievedDocumentChunk(
        document_id=document_id,
        filename=f"{document_id}.md",
        chunk_index=chunk_index,
        content=content,
        start_offset=chunk_index * 100,
        end_offset=(chunk_index * 100) + len(content),
        metadata=None,
        score=score,
    )


def test_evaluate_expression_supports_basic_arithmetic() -> None:
    assert evaluate_expression("2 + 3 * 4") == 14
    assert evaluate_expression("(2 + 3) * 4") == 20
    assert evaluate_expression("-5 + 2") == -3


def test_evaluate_expression_rejects_unsafe_syntax() -> None:
    with pytest.raises(ValueError, match="unsupported|invalid"):
        evaluate_expression("__import__('os').system('whoami')")


def test_evaluate_expression_rejects_unbounded_exponentiation() -> None:
    with pytest.raises(ValueError, match="too large"):
        evaluate_expression("9**9**9")


def test_evaluate_expression_allows_reasonably_large_exponents() -> None:
    assert evaluate_expression("2 ** 10") == 1024
    assert evaluate_expression("10 ** 100") == 10**100


def test_evaluate_expression_rejects_boolean_constants() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_expression("True + 1")


def test_evaluate_expression_reports_division_by_zero_cleanly() -> None:
    with pytest.raises(ValueError, match="division by zero"):
        evaluate_expression("1 / 0")


@pytest.mark.anyio
async def test_tool_registry_executes_search_tool_and_collects_citations() -> None:
    retriever = StubRetriever(
        [
            make_chunk(document_id="doc-1", chunk_index=0, content="Guide snippet", score=0.91),
            make_chunk(document_id="doc-2", chunk_index=0, content="FAQ snippet", score=0.82),
        ]
    )
    registry = build_tool_registry(
        retriever=retriever,  # type: ignore[arg-type]
        search_top_k=2,
        timeout_seconds=5.0,
    )

    result = await registry.execute(
        ToolCallRequest(
            call_id="tool-1",
            name="search_knowledge_base",
            arguments={"query": "guide", "top_k": 5},
        ),
        context=ToolExecutionContext(
            conversation_id="conv-search",
            owner_user_id=None,
            request_metadata={"source": "unit-test"},
        ),
    )

    assert retriever.calls == [("guide", 2, 2, None)]
    assert result.tool_run.status == "completed"
    assert result.tool_run.output == {
        "hits": [
            {
                "document_id": "doc-1",
                "filename": "doc-1.md",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": len("Guide snippet"),
                "snippet": "Guide snippet",
                "score": 0.91,
            },
            {
                "document_id": "doc-2",
                "filename": "doc-2.md",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": len("FAQ snippet"),
                "snippet": "FAQ snippet",
                "score": 0.82,
            },
        ]
    }
    assert [citation.document_id for citation in result.citations] == ["doc-1", "doc-2"]


@pytest.mark.anyio
async def test_tool_registry_allows_caller_top_k_up_to_configured_max() -> None:
    retriever = StubRetriever(
        [make_chunk(document_id="doc-1", chunk_index=0, content="Guide snippet", score=0.91)]
    )
    registry = build_tool_registry(
        retriever=retriever,  # type: ignore[arg-type]
        search_top_k=2,
        search_max_top_k=5,
        timeout_seconds=5.0,
    )

    await registry.execute(
        ToolCallRequest(
            call_id="tool-1",
            name="search_knowledge_base",
            arguments={"query": "guide", "top_k": 5},
        ),
        context=ToolExecutionContext(
            conversation_id="conv-search",
            owner_user_id=None,
            request_metadata={"source": "unit-test"},
        ),
    )

    assert retriever.calls == [("guide", 5, 5, None)]


@pytest.mark.anyio
async def test_tool_registry_rejects_unknown_tool() -> None:
    registry = build_tool_registry(
        retriever=StubRetriever([]),  # type: ignore[arg-type]
        search_top_k=3,
        timeout_seconds=5.0,
    )

    result = await registry.execute(
        ToolCallRequest(call_id="tool-unknown", name="not_allowed", arguments={}),
        context=ToolExecutionContext(
            conversation_id="conv-reject",
            owner_user_id=None,
            request_metadata=None,
        ),
    )

    assert result.tool_run.status == "rejected"
    assert result.tool_run.error == "tool 'not_allowed' is not allowlisted"


class WrongOutput(BaseModel):
    unexpected: str = "oops"


@pytest.mark.anyio
async def test_tool_registry_reports_failure_when_handler_returns_wrong_output_model() -> None:
    class ExpectedOutput(BaseModel):
        value: str = "ok"

    class MisbehavingInput(BaseModel):
        pass

    async def misbehaving_handler(payload, context):
        del payload, context
        return WrongOutput()

    registry = ToolRegistry(
        [
            RegisteredTool(
                name="misbehaving",
                description="returns the wrong output model",
                input_model=MisbehavingInput,
                output_model=ExpectedOutput,
                handler=misbehaving_handler,
                timeout_seconds=5.0,
            )
        ]
    )

    result = await registry.execute(
        ToolCallRequest(call_id="tool-misbehaving", name="misbehaving", arguments={}),
        context=ToolExecutionContext(
            conversation_id="conv-misbehaving",
            owner_user_id=None,
            request_metadata=None,
        ),
    )

    assert result.tool_run.status == "failed"
    assert "unexpected output model" in result.tool_run.error


@pytest.mark.anyio
async def test_tool_registry_returns_current_user_profile_from_request_metadata() -> None:
    registry = build_tool_registry(
        retriever=StubRetriever([]),  # type: ignore[arg-type]
        search_top_k=3,
        timeout_seconds=5.0,
    )

    result = await registry.execute(
        ToolCallRequest(
            call_id="tool-profile",
            name="get_current_user_profile",
            arguments={},
        ),
        context=ToolExecutionContext(
            conversation_id="conv-profile",
            owner_user_id=None,
            request_metadata={
                "user_profile": {
                    "user_id": "user-123",
                    "display_name": "Alice",
                    "email": "alice@example.com",
                    "plan": "pro",
                    "locale": "en-US",
                    "preferences": {"timezone": "UTC"},
                }
            },
        ),
    )

    assert result.tool_run.status == "completed"
    assert result.tool_run.output == {
        "found": True,
        "profile": {
            "user_id": "user-123",
            "display_name": "Alice",
            "email": "alice@example.com",
            "plan": "pro",
            "locale": "en-US",
            "preferences": {"timezone": "UTC"},
        },
    }
    assert result.citations == []


@pytest.mark.anyio
async def test_tool_registry_returns_profile_not_found_when_metadata_is_missing() -> None:
    registry = build_tool_registry(
        retriever=StubRetriever([]),  # type: ignore[arg-type]
        search_top_k=3,
        timeout_seconds=5.0,
    )

    result = await registry.execute(
        ToolCallRequest(
            call_id="tool-profile-missing",
            name="get_current_user_profile",
            arguments={},
        ),
        context=ToolExecutionContext(
            conversation_id="conv-profile-missing",
            owner_user_id=None,
            request_metadata={"source": "unit-test"},
        ),
    )

    assert result.tool_run.status == "completed"
    assert result.tool_run.output == {"found": False, "profile": None}


@pytest.mark.anyio
async def test_tool_registry_returns_profile_not_found_when_metadata_is_malformed() -> None:
    # `configure_json_logger()` sets propagate=False on the "chatbot_api" logger
    # the first time it runs anywhere in the process, which breaks pytest's
    # caplog (it relies on propagation to the root logger) depending on test
    # order -- attach a handler directly instead, same convention used by
    # tests/test_chat.py's audit_log_handler fixture.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("chatbot_api")
    logger.addHandler(handler)
    try:
        registry = build_tool_registry(
            retriever=StubRetriever([]),  # type: ignore[arg-type]
            search_top_k=3,
            timeout_seconds=5.0,
        )

        result = await registry.execute(
            ToolCallRequest(
                call_id="tool-profile-malformed",
                name="get_current_user_profile",
                arguments={},
            ),
            context=ToolExecutionContext(
                conversation_id="conv-profile-malformed",
                owner_user_id=None,
                request_metadata={
                    "user_profile": {
                        "display_name": "Missing user id",
                        "preferences": [],
                    }
                },
            ),
        )
    finally:
        logger.removeHandler(handler)

    assert result.tool_run.status == "completed"
    assert result.tool_run.output == {"found": False, "profile": None}
    assert any(
        "malformed user_profile metadata ignored" in record.getMessage() for record in records
    )
