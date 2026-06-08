from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from chatbot_api.main import app, get_chat_service
from chatbot_api.providers import (
    ChatCompletion,
    ChatCompletionMetadata,
    ChatTurn,
    ToolCallBatch,
    ToolCallRequest,
    ToolRun,
)
from chatbot_api.repositories import RetrievedDocumentChunk
from chatbot_api.retrieval import DocumentRetriever
from chatbot_api.services import ChatService, ChatStreamComplete, ChatStreamStart
from chatbot_api.settings import Settings
from chatbot_api.tools import build_tool_registry
from chatbot_api.tracing import LangSmithTraceSink, NoopTraceSink, build_trace_sink
from chatbot_api.workflow import ChatWorkflow, build_chat_workflow

_CURRENT_SPAN: ContextVar["RecordedTraceSpan | None"] = ContextVar(
    "recorded_trace_span",
    default=None,
)


@dataclass
class RecordedTraceSpan:
    sink: RecordingTraceSink
    name: str
    run_type: str
    inputs: dict[str, Any]
    metadata: dict[str, Any]
    tags: list[str]
    parent: RecordedTraceSpan | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None
    finished: bool = False
    _activation_token: Token[RecordedTraceSpan | None] | None = None

    def start_child_span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RecordedTraceSpan:
        return self.sink.start_span(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            tags=tags,
            parent=self,
        )

    def annotate(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if metadata:
            self.metadata.update(metadata)
        if tags:
            self.tags.extend(tags)

    def finish_success(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if self.finished:
            return
        self.annotate(metadata=metadata, tags=tags)
        self.outputs = outputs
        self.finished = True
        self._reset_activation()

    def finish_error(
        self,
        error: BaseException | str,
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if self.finished:
            return
        self.annotate(metadata=metadata, tags=tags)
        self.error = str(error)
        self.finished = True
        self._reset_activation()

    def activate(self) -> Token[RecordedTraceSpan | None]:
        return _CURRENT_SPAN.set(self)

    def deactivate(self, token: Token[RecordedTraceSpan | None]) -> None:
        _CURRENT_SPAN.reset(token)

    def suspend(self) -> None:
        self._reset_activation()

    def __enter__(self) -> RecordedTraceSpan:
        if self._activation_token is None:
            self._activation_token = self.activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        if exc_value is None:
            self.finish_success()
        else:
            self.finish_error(exc_value)
        return None

    def _reset_activation(self) -> None:
        if self._activation_token is not None:
            try:
                self.deactivate(self._activation_token)
            except ValueError:
                pass
            self._activation_token = None


@dataclass
class RecordingTraceSink:
    enabled: bool = True
    spans: list[RecordedTraceSpan] = field(default_factory=list)

    def start_request_span(
        self,
        name: str,
        *,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RecordedTraceSpan:
        return self.start_span(
            name,
            run_type="chain",
            inputs=inputs,
            metadata=metadata,
            tags=tags,
        )

    def start_span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        parent: RecordedTraceSpan | None = None,
    ) -> RecordedTraceSpan:
        span = RecordedTraceSpan(
            sink=self,
            name=name,
            run_type=run_type,
            inputs={} if inputs is None else dict(inputs),
            metadata={} if metadata is None else dict(metadata),
            tags=[] if tags is None else list(tags),
            parent=parent or _CURRENT_SPAN.get(),
        )
        self.spans.append(span)
        return span

    def wrap_openai_client(self, client: Any) -> Any:
        return client

    def close(self) -> None:
        return None

    def find_span(self, name: str) -> RecordedTraceSpan:
        for span in self.spans:
            if span.name == name:
                return span
        raise AssertionError(f"missing span: {name}")


class StubRetrieverRepository:
    def __init__(self, chunks: list[RetrievedDocumentChunk]) -> None:
        self._chunks = chunks

    async def search_similar_chunks(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedDocumentChunk]:
        return self._chunks[:limit]


class StubEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = embeddings

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embeddings


class StubChatService:
    def __init__(
        self,
        *,
        completion: ChatCompletion | None = None,
        stream_events: list[Any] | None = None,
    ) -> None:
        self._completion = completion
        self._stream_events = [] if stream_events is None else stream_events

    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[str, ChatCompletion]:
        del message, metadata
        return conversation_id or "generated-conv", self._completion  # type: ignore[return-value]

    async def stream_chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[Any]:
        del conversation_id, message, metadata
        for event in self._stream_events:
            yield event


class InMemoryChatRepository:
    def __init__(self) -> None:
        self.messages_by_conversation: dict[str, list[ChatTurn]] = {}
        self.tool_runs: dict[str, ToolRun] = {}

    async def list_messages(self, conversation_id: str) -> list[ChatTurn]:
        return list(self.messages_by_conversation.get(conversation_id, []))

    async def create_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        input_payload: dict[str, Any],
    ) -> None:
        del conversation_id, tool_name, input_payload
        self.tool_runs[tool_call_id] = ToolRun(
            tool_call_id=tool_call_id,
            tool_name="search_knowledge_base",
            status="completed",
            input={},
        )

    async def complete_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        output_payload: dict[str, Any],
    ) -> None:
        del conversation_id, output_payload
        self.tool_runs[tool_call_id] = ToolRun(
            tool_call_id=tool_call_id,
            tool_name="search_knowledge_base",
            status="completed",
            input={},
            output={},
        )

    async def fail_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        error_message: str,
    ) -> None:
        del conversation_id
        self.tool_runs[tool_call_id] = ToolRun(
            tool_call_id=tool_call_id,
            tool_name="search_knowledge_base",
            status=status,
            input={},
            error=error_message,
        )

    async def append_exchange(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_metadata: dict[str, Any] | None,
        assistant_message: str,
    ) -> None:
        del user_metadata
        self.messages_by_conversation.setdefault(conversation_id, []).extend(
            [
                ChatTurn(role="user", content=user_message),
                ChatTurn(role="assistant", content=assistant_message),
            ]
        )


class ScriptedProvider:
    def __init__(self, responses: list[ChatCompletion | ToolCallBatch]) -> None:
        self._responses = list(responses)

    async def generate_response(
        self,
        messages: list[ChatTurn],
        *,
        tools=(),
        previous_response_id: str | None = None,
        tool_outputs=(),
    ):
        del messages, tools, previous_response_id, tool_outputs
        if not self._responses:
            raise AssertionError("provider called more times than expected")
        return self._responses.pop(0)


def build_chat_service_override(service: StubChatService):
    async def override() -> ChatService:
        return service  # type: ignore[return-value]

    return override


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


@pytest.fixture(autouse=True)
def clear_app_state() -> None:
    previous_trace_sink = getattr(app.state, "trace_sink", None)
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    if previous_trace_sink is None:
        if hasattr(app.state, "trace_sink"):
            delattr(app.state, "trace_sink")
    else:
        app.state.trace_sink = previous_trace_sink


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


def test_build_trace_sink_returns_noop_when_langsmith_is_disabled() -> None:
    sink = build_trace_sink(Settings(langsmith_tracing_enabled=False))

    assert isinstance(sink, NoopTraceSink)


def test_build_trace_sink_returns_langsmith_when_langsmith_is_enabled() -> None:
    sink = build_trace_sink(
        Settings(
            langsmith_tracing_enabled=True,
            langsmith_api_key="test-key",
            langsmith_project="chatbot-tests",
        )
    )

    assert isinstance(sink, LangSmithTraceSink)
    sink.close()


def test_langsmith_trace_sink_wraps_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_wrap_openai(client: Any) -> Any:
        captured["client"] = client
        return "wrapped-client"

    monkeypatch.setattr("chatbot_api.tracing.wrap_openai", fake_wrap_openai)
    sink = LangSmithTraceSink(
        Settings(
            langsmith_tracing_enabled=True,
            langsmith_api_key="test-key",
            langsmith_project="chatbot-tests",
        )
    )

    wrapped = sink.wrap_openai_client(object())

    assert wrapped == "wrapped-client"
    assert "client" in captured
    sink.close()


@pytest.mark.anyio
async def test_chat_endpoint_records_request_trace(async_client: AsyncClient) -> None:
    trace_sink = RecordingTraceSink()
    app.state.trace_sink = trace_sink
    app.dependency_overrides[get_chat_service] = build_chat_service_override(
        StubChatService(
            completion=ChatCompletion(
                content="Grounded answer",
                provider="openai",
                model="gpt-4.1-mini",
                metadata=ChatCompletionMetadata(),
            )
        )
    )

    response = await async_client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    request_span = trace_sink.find_span("chat.request")
    assert request_span.finished is True
    assert request_span.metadata["chat_mode"] == "sync"
    assert request_span.metadata["request_id"]
    assert request_span.outputs == {
        "conversation_id": "generated-conv",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "assistant_message": "Grounded answer",
        "citation_count": 0,
        "tool_run_count": 0,
        "usage": None,
        "cost": None,
    }


@pytest.mark.anyio
async def test_stream_chat_request_trace_finishes_on_terminal_event(
    async_client: AsyncClient,
) -> None:
    trace_sink = RecordingTraceSink()
    app.state.trace_sink = trace_sink
    app.dependency_overrides[get_chat_service] = build_chat_service_override(
        StubChatService(
            stream_events=[
                ChatStreamStart(conversation_id="conv-stream"),
                ChatStreamComplete(
                    conversation_id="conv-stream",
                    completion=ChatCompletion(
                        content="Done",
                        provider="openai",
                        model="gpt-4.1-mini",
                        metadata=ChatCompletionMetadata(),
                    ),
                ),
            ]
        )
    )

    async with async_client.stream(
        "POST",
        "/chat",
        json={"message": "hello", "stream": True},
    ) as response:
        async for _ in response.aiter_lines():
            pass

    assert response.status_code == 200
    request_span = trace_sink.find_span("chat.request")
    assert request_span.finished is True
    assert request_span.metadata["chat_mode"] == "stream"
    assert request_span.outputs == {
        "conversation_id": "conv-stream",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "assistant_message": "Done",
        "citation_count": 0,
        "tool_run_count": 0,
        "usage": None,
        "cost": None,
    }


@pytest.mark.anyio
async def test_workflow_traces_tool_and_retrieval_hierarchy() -> None:
    trace_sink = RecordingTraceSink()
    repository = InMemoryChatRepository()
    retriever = DocumentRetriever(
        StubRetrieverRepository(
            [
                make_chunk(
                    document_id="doc-1",
                    chunk_index=0,
                    content="Relevant context",
                    score=0.92,
                )
            ]
        ),
        StubEmbeddingProvider([[0.1, 0.2]]),
        top_k=3,
        min_score=0.35,
        max_chunks_per_document=1,
        candidate_limit=12,
        trace_sink=trace_sink,
    )
    tool_registry = build_tool_registry(
        retriever=retriever,
        search_top_k=3,
        timeout_seconds=5.0,
        trace_sink=trace_sink,
    )
    provider = ScriptedProvider(
        [
            ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-1",
                        name="search_knowledge_base",
                        arguments={"query": "guide"},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-1",
            ),
            ChatCompletion(
                content="Grounded answer",
                provider="openai",
                model="gpt-4.1-mini",
                metadata=ChatCompletionMetadata(),
                response_id="resp-2",
            ),
        ]
    )
    workflow: ChatWorkflow = build_chat_workflow()

    conversation_id, completion = await workflow.run(
        conversation_id="conv-1",
        message="What does the guide say?",
        metadata=None,
        provider=provider,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        trace_sink=trace_sink,
    )

    assert conversation_id == "conv-1"
    assert completion.content == "Grounded answer"
    workflow_span = trace_sink.find_span("workflow.run")
    execute_tools_span = trace_sink.find_span("workflow.execute_tools")
    tool_span = trace_sink.find_span("tool.execute")
    retrieval_span = trace_sink.find_span("retrieval.retrieve_chunks")

    assert workflow_span.finished is True
    assert execute_tools_span.parent is workflow_span
    assert tool_span.parent is execute_tools_span
    assert retrieval_span.parent is tool_span
    assert trace_sink.find_span("workflow.load_context").parent is workflow_span
    assert trace_sink.find_span("workflow.call_model").parent is workflow_span
    assert trace_sink.find_span("workflow.persist_response").parent is workflow_span
