from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from chatbot_api.main import app, get_app_observability, get_chat_service
from chatbot_api.observability import (
    JsonLogFormatter,
    ObservabilityService,
    bind_request_context,
    get_process_observability,
    reset_process_observability,
    reset_request_context,
)
from chatbot_api.providers import (
    ChatCompletion,
    ChatCompletionMetadata,
    TokenUsage,
    ToolCallRequest,
    UsageCost,
)
from chatbot_api.repositories import RetrievedDocumentChunk
from chatbot_api.retrieval import DocumentRetriever
from chatbot_api.services import ChatService
from chatbot_api.settings import Settings, get_settings
from chatbot_api.tasks import execute_embed_document_task
from chatbot_api.tools import ToolExecutionContext, build_tool_registry


class StubRetrieverRepository:
    def __init__(self, chunks: list[RetrievedDocumentChunk]) -> None:
        self._chunks = chunks

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        del owner_user_id
        return self._chunks[:limit]


class StubEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = embeddings

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embeddings


class StubToolRetriever:
    def __init__(self, chunks: list[RetrievedDocumentChunk]) -> None:
        self._chunks = chunks

    async def retrieve_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chunks_per_document: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        del query, max_chunks_per_document, owner_user_id
        return self._chunks[: top_k or len(self._chunks)]


class StubChatService:
    def __init__(self, completion: ChatCompletion) -> None:
        self._completion = completion

    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
        owner_user_id: str | None = None,
    ) -> tuple[str, ChatCompletion]:
        del message, metadata, owner_user_id
        return conversation_id or "generated-conv", self._completion

    async def stream_chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
        owner_user_id: str | None = None,
    ) -> AsyncIterator[Any]:
        del conversation_id, message, metadata, owner_user_id
        if False:
            yield None


def build_chat_service_override(service: StubChatService):
    async def override() -> ChatService:
        return service  # type: ignore[return-value]

    return override


def make_chunk(
    *,
    document_id: str,
    chunk_index: int,
    score: float,
) -> RetrievedDocumentChunk:
    content = f"content for {document_id} chunk {chunk_index}"
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


@pytest.fixture(autouse=True)
def reset_observability_state() -> None:
    get_settings.cache_clear()
    get_app_observability(app, get_settings()).reset_for_tests()
    reset_process_observability()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    get_app_observability(app, get_settings()).reset_for_tests()
    reset_process_observability()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_metrics_endpoint_and_request_id_header(async_client: AsyncClient) -> None:
    health_response = await async_client.get("/health", headers={"X-Request-ID": "req-123"})

    assert health_response.status_code == 200
    assert health_response.headers["X-Request-ID"] == "req-123"

    metrics_response = await async_client.get("/metrics")

    assert metrics_response.status_code == 200
    assert "http_requests_total" in metrics_response.text
    assert (
        'http_requests_total{method="GET",route="/health",status_code="200"}'
        in metrics_response.text
    )


@pytest.mark.anyio
async def test_chat_success_updates_metrics(async_client: AsyncClient) -> None:
    service = StubChatService(
        ChatCompletion(
            content="Grounded answer",
            provider="openai",
            model="gpt-4.1-mini",
            metadata=ChatCompletionMetadata(
                usage=TokenUsage(input_tokens=80, output_tokens=20, total_tokens=100),
                cost=UsageCost(
                    input_cost_usd=0.000032,
                    output_cost_usd=0.000032,
                    total_cost_usd=0.000064,
                ),
            ),
        )
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    response = await async_client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200

    metrics_response = await async_client.get("/metrics")

    assert 'chat_requests_total{mode="sync",outcome="success"} 1' in metrics_response.text


def test_structured_logs_include_request_id() -> None:
    context_token = bind_request_context(request_id="req-log-1")
    try:
        formatter = JsonLogFormatter()
        payload = json.loads(
            formatter.format(
                logging.LogRecord(
                    name="chatbot_api",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=0,
                    msg={"event": "unit.test", "conversation_id": "conv-1"},
                    args=(),
                    exc_info=None,
                )
            )
        )
    finally:
        reset_request_context(context_token)

    assert payload["event"] == "unit.test"
    assert payload["request_id"] == "req-log-1"
    assert payload["conversation_id"] == "conv-1"


@pytest.mark.anyio
async def test_retrieval_metrics_capture_hit_and_miss() -> None:
    observability = ObservabilityService(Settings())
    retriever = DocumentRetriever(
        StubRetrieverRepository([make_chunk(document_id="doc-1", chunk_index=0, score=0.92)]),
        StubEmbeddingProvider([[0.3, 0.4]]),
        top_k=4,
        min_score=0.35,
        max_chunks_per_document=1,
        candidate_limit=12,
        observability=observability,
    )

    hit_result = await retriever.retrieve("guide")
    miss_result = await retriever.retrieve("   ")

    assert hit_result is not None
    assert miss_result is None
    metrics_output = observability.render_metrics()
    assert 'retrieval_requests_total{outcome="hit"} 1' in metrics_output
    assert 'retrieval_requests_total{outcome="skipped"} 1' in metrics_output


@pytest.mark.anyio
async def test_tool_registry_metrics_and_workflow_counter_are_emitted() -> None:
    observability = ObservabilityService(Settings())
    tool_registry = build_tool_registry(
        retriever=StubToolRetriever([]),  # type: ignore[arg-type]
        search_top_k=3,
        timeout_seconds=5.0,
        observability=observability,
    )

    result = await tool_registry.execute(
        ToolCallRequest(
            call_id="tool-1",
            name="calculator",
            arguments={"expression": "2 + 2"},
        ),
        context=ToolExecutionContext(
            conversation_id="conv-observability",
            owner_user_id=None,
            request_metadata=None,
        ),
    )
    observability.record_chat_workflow(
        mode="sync",
        outcome="completed",
        duration_seconds=0.05,
    )

    assert result.tool_run.status == "completed"
    metrics_output = observability.render_metrics()
    assert 'tool_calls_total{tool_name="calculator",status="completed"} 1' in metrics_output
    assert 'chat_workflow_runs_total{mode="sync",outcome="completed"} 1' in metrics_output


def test_llm_request_metrics_capture_usage_and_cost() -> None:
    observability = ObservabilityService(Settings())

    observability.record_llm_request(
        model="gpt-4.1-mini",
        outcome="completed",
        duration_seconds=0.2,
        usage=TokenUsage(input_tokens=123, output_tokens=45, total_tokens=168),
        cost=UsageCost(
            input_cost_usd=0.0000492,
            output_cost_usd=0.000072,
            total_cost_usd=0.0001212,
        ),
    )

    metrics_output = observability.render_metrics()
    assert 'llm_requests_total{model="gpt-4.1-mini",outcome="completed"} 1' in metrics_output
    assert 'llm_input_tokens_total{model="gpt-4.1-mini"} 123' in metrics_output
    assert 'llm_output_tokens_total{model="gpt-4.1-mini"} 45' in metrics_output
    assert 'llm_total_tokens_total{model="gpt-4.1-mini"} 168' in metrics_output
    assert 'llm_request_cost_usd_total{model="gpt-4.1-mini"} 0.0001212' in metrics_output


def test_embedding_job_metrics_capture_success() -> None:
    settings = Settings()
    reset_process_observability()

    async def successful_job(document_id: str) -> dict[str, str | int]:
        return {
            "document_id": document_id,
            "status": "ready",
            "updated_chunks": 2,
        }

    def retry(**kwargs):
        raise AssertionError("retry should not be called")

    result = execute_embed_document_task(
        document_id="doc-42",
        retry_count=0,
        retry=retry,
        settings=settings,
        run_job=successful_job,
    )

    assert result == {
        "document_id": "doc-42",
        "status": "ready",
        "updated_chunks": 2,
    }
    metrics_output = get_process_observability(settings).render_metrics()
    assert 'document_embedding_jobs_total{outcome="success"} 1' in metrics_output
