from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.runtime import Runtime

from chatbot_api import workflow as workflow_module
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletion,
    ChatCompletionMetadata,
    ChatProviderError,
    ChatTurn,
    TokenUsage,
    ToolCallBatch,
    ToolCallRequest,
    ToolResultMessage,
    ToolRun,
    UsageCost,
)
from chatbot_api.repositories import RetrievedDocumentChunk
from chatbot_api.tools import ToolRegistry, build_tool_registry
from chatbot_api.tracing import NoopTraceSink
from chatbot_api.workflow import (
    WorkflowMessageComplete,
    WorkflowMessageDelta,
    WorkflowMessageStart,
    WorkflowToolComplete,
    WorkflowToolStart,
    build_chat_workflow,
    call_model_node,
    initial_workflow_state,
    load_context_node,
    persist_response_node,
    serialize_turn,
)


@dataclass(frozen=True)
class SavedExchange:
    conversation_id: str
    user_message: str
    user_metadata: dict[str, Any] | None
    assistant_message: str


@dataclass(frozen=True)
class SavedToolRun:
    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None
    error_message: str | None


class InMemoryChatRepository:
    def __init__(self) -> None:
        self.messages_by_conversation: dict[str, list[ChatTurn]] = {}
        self.saved_exchanges: list[SavedExchange] = []
        self.tool_runs: list[SavedToolRun] = []

    async def list_messages(self, conversation_id: str) -> list[ChatTurn]:
        return list(self.messages_by_conversation.get(conversation_id, []))

    async def create_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        input_payload: dict[str, Any],
    ):
        record = SavedToolRun(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status="running",
            input_payload=input_payload,
            output_payload=None,
            error_message=None,
        )
        self.tool_runs.append(record)
        return record

    async def complete_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        output_payload: dict[str, Any],
    ):
        for index, tool_run in enumerate(self.tool_runs):
            if (
                tool_run.conversation_id == conversation_id
                and tool_run.tool_call_id == tool_call_id
                and tool_run.status == "running"
            ):
                updated = SavedToolRun(
                    conversation_id=conversation_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_run.tool_name,
                    status="completed",
                    input_payload=tool_run.input_payload,
                    output_payload=output_payload,
                    error_message=None,
                )
                self.tool_runs[index] = updated
                return updated
        return None

    async def fail_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        error_message: str,
    ):
        for index, tool_run in enumerate(self.tool_runs):
            if (
                tool_run.conversation_id == conversation_id
                and tool_run.tool_call_id == tool_call_id
                and tool_run.status == "running"
            ):
                updated = SavedToolRun(
                    conversation_id=conversation_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_run.tool_name,
                    status=status,
                    input_payload=tool_run.input_payload,
                    output_payload=None,
                    error_message=error_message,
                )
                self.tool_runs[index] = updated
                return updated
        return None

    async def append_exchange(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_metadata: dict[str, Any] | None,
        assistant_message: str,
    ) -> None:
        self.saved_exchanges.append(
            SavedExchange(
                conversation_id=conversation_id,
                user_message=user_message,
                user_metadata=user_metadata,
                assistant_message=assistant_message,
            )
        )
        self.messages_by_conversation.setdefault(conversation_id, []).extend(
            [
                ChatTurn(role="user", content=user_message),
                ChatTurn(role="assistant", content=assistant_message),
            ]
        )


class ScriptedProvider:
    def __init__(self, responses: list[ChatCompletion | ToolCallBatch]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_response(
        self,
        messages: list[ChatTurn],
        *,
        tools=(),
        previous_response_id: str | None = None,
        tool_outputs=(),
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "tool_names": [tool.name for tool in tools],
                "previous_response_id": previous_response_id,
                "tool_outputs": list(tool_outputs),
            }
        )
        if not self._responses:
            raise AssertionError("provider called more times than expected")
        return self._responses.pop(0)


class FailingAfterToolProvider:
    def __init__(self) -> None:
        self._calls = 0

    async def generate_response(
        self,
        messages: list[ChatTurn],
        *,
        tools=(),
        previous_response_id: str | None = None,
        tool_outputs=(),
    ):
        self._calls += 1
        if self._calls == 1:
            return ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-1",
                        name="calculator",
                        arguments={"expression": "2 + 2"},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-1",
            )
        raise ChatProviderError("LLM provider request failed")


class StubRetriever:
    def __init__(self, chunks: list[RetrievedDocumentChunk]) -> None:
        self._chunks = chunks
        self.queries: list[tuple[str, int, int]] = []

    async def retrieve_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedDocumentChunk]:
        self.queries.append((query, top_k or 0, max_chunks_per_document or 0))
        return self._chunks[: top_k or len(self._chunks)]


def make_tool_registry(
    chunks: list[RetrievedDocumentChunk] | None = None,
    *,
    search_top_k: int = 3,
) -> ToolRegistry:
    retriever = StubRetriever(chunks or [])
    return build_tool_registry(
        retriever=retriever,  # type: ignore[arg-type]
        search_top_k=search_top_k,
        timeout_seconds=5.0,
    )


def build_runtime(
    provider: ScriptedProvider | FailingAfterToolProvider,
    repository: InMemoryChatRepository,
    tool_registry: ToolRegistry | None = None,
    tool_max_rounds: int = 4,
    pricing_model: str | None = "gpt-4.1-mini",
    input_price_per_1m_tokens: float | None = 0.40,
    output_price_per_1m_tokens: float | None = 1.60,
) -> Runtime[dict[str, Any]]:
    return Runtime(
        context={
            "provider": provider,
            "repository": repository,
            "tool_registry": tool_registry,
            "tool_max_rounds": tool_max_rounds,
            "observability": None,
            "trace_sink": NoopTraceSink(),
            "pricing_model": pricing_model,
            "input_price_per_1m_tokens": input_price_per_1m_tokens,
            "output_price_per_1m_tokens": output_price_per_1m_tokens,
        }
    )


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


@pytest.mark.anyio
async def test_load_context_node_replays_history_and_appends_current_user_message() -> None:
    repository = InMemoryChatRepository()
    repository.messages_by_conversation["conv-123"] = [
        ChatTurn(role="user", content="Earlier question"),
        ChatTurn(role="assistant", content="Earlier answer"),
    ]
    provider = ScriptedProvider([])

    state = initial_workflow_state(
        conversation_id="conv-123",
        message="New question",
        metadata={"source": "thread"},
        stream=False,
    )

    updates = await load_context_node(state, build_runtime(provider, repository))

    assert updates["history"] == [
        serialize_turn(ChatTurn(role="user", content="Earlier question")),
        serialize_turn(ChatTurn(role="assistant", content="Earlier answer")),
    ]
    assert updates["provider_messages"] == [
        serialize_turn(ChatTurn(role="user", content="Earlier question")),
        serialize_turn(ChatTurn(role="assistant", content="Earlier answer")),
        serialize_turn(ChatTurn(role="user", content="New question")),
    ]


@pytest.mark.anyio
async def test_call_model_node_returns_completion_for_sync_path() -> None:
    repository = InMemoryChatRepository()
    provider = ScriptedProvider(
        [
            ChatCompletion(
                content="Hello from the workflow",
                provider="openai",
                model="gpt-4.1-mini",
                metadata=ChatCompletionMetadata(
                    usage=TokenUsage(input_tokens=12, output_tokens=6, total_tokens=18)
                ),
                response_id="resp-1",
            )
        ]
    )
    state = initial_workflow_state(
        conversation_id="conv-123",
        message="Hello",
        metadata=None,
        stream=False,
    )
    state["provider_messages"] = [serialize_turn(ChatTurn(role="user", content="Hello"))]

    updates = await call_model_node(state, build_runtime(provider, repository))

    assert updates == {
        "provider_name": "openai",
        "model_name": "gpt-4.1-mini",
        "provider_response_id": "resp-1",
        "pending_tool_outputs": [],
        "response_metadata": {
            "citations": [],
            "tool_runs": [],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "cost": {
                "input_cost_usd": 0.0000048,
                "output_cost_usd": 0.0000096,
                "total_cost_usd": 0.0000144,
                "currency": "USD",
            },
        },
        "usage_totals": {
            "input_tokens": 12,
            "output_tokens": 6,
            "total_tokens": 18,
        },
        "cost_totals": {
            "input_cost_usd": 0.0000048,
            "output_cost_usd": 0.0000096,
            "total_cost_usd": 0.0000144,
            "currency": "USD",
        },
        "model_rounds": 1,
        "assistant_message": "Hello from the workflow",
        "pending_tool_calls": [],
    }
    assert provider.calls[0]["messages"] == [ChatTurn(role="user", content="Hello")]


@pytest.mark.anyio
async def test_call_model_node_keeps_usage_and_skips_cost_without_pricing() -> None:
    repository = InMemoryChatRepository()
    provider = ScriptedProvider(
        [
            ChatCompletion(
                content="Hello from the workflow",
                provider="openai",
                model="gpt-4.1-mini",
                metadata=ChatCompletionMetadata(
                    usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14)
                ),
                response_id="resp-1",
            )
        ]
    )
    state = initial_workflow_state(
        conversation_id="conv-123",
        message="Hello",
        metadata=None,
        stream=False,
    )
    state["provider_messages"] = [serialize_turn(ChatTurn(role="user", content="Hello"))]

    updates = await call_model_node(
        state,
        build_runtime(
            provider,
            repository,
            pricing_model=None,
            input_price_per_1m_tokens=None,
            output_price_per_1m_tokens=None,
        ),
    )

    assert updates["response_metadata"] == {
        "citations": [],
        "tool_runs": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        },
    }
    assert updates["cost_totals"] is None


@pytest.mark.anyio
async def test_persist_response_node_saves_exchange_and_emits_complete_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = InMemoryChatRepository()
    provider = ScriptedProvider([])
    state = initial_workflow_state(
        conversation_id="conv-123",
        message="Hello",
        metadata={"source": "test"},
        stream=True,
    )
    state["assistant_message"] = "Hello from the workflow"
    state["provider_name"] = "openai"
    state["model_name"] = "gpt-4.1-mini"
    state["response_metadata"] = {
        "citations": [],
        "tool_runs": [
            {
                "tool_call_id": "tool-1",
                "tool_name": "calculator",
                "status": "completed",
                "input": {"expression": "2 + 2"},
                "output": {"result": 4},
                "error": None,
            }
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
        "cost": {
            "input_cost_usd": 0.000008,
            "output_cost_usd": 0.000008,
            "total_cost_usd": 0.000016,
            "currency": "USD",
        },
    }
    emitted_events: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow_module, "get_stream_writer", lambda: emitted_events.append)

    updates = await persist_response_node(state, build_runtime(provider, repository))

    assert updates == {}
    assert repository.saved_exchanges == [
        SavedExchange(
            conversation_id="conv-123",
            user_message="Hello",
            user_metadata={"source": "test"},
            assistant_message="Hello from the workflow",
        )
    ]
    assert emitted_events == [
        WorkflowMessageDelta(type="message_delta", delta="Hello from the workflow"),
        WorkflowMessageComplete(
            type="message_complete",
            conversation_id="conv-123",
            assistant_message="Hello from the workflow",
            provider="openai",
            model="gpt-4.1-mini",
            metadata=state["response_metadata"],
        ),
    ]


@pytest.mark.anyio
async def test_chat_workflow_runs_sync_end_to_end_with_tool_execution() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    provider = ScriptedProvider(
        [
            ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-1",
                        name="search_knowledge_base",
                        arguments={"query": "guide answer", "top_k": 2},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-1",
                usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            ),
            ChatCompletion(
                content="Grounded answer",
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-2",
                metadata=ChatCompletionMetadata(
                    usage=TokenUsage(input_tokens=40, output_tokens=25, total_tokens=65)
                ),
            ),
        ]
    )
    tool_registry = make_tool_registry(
        [
            make_chunk(document_id="doc-1", chunk_index=0, content="Guide snippet", score=0.91),
            make_chunk(document_id="doc-2", chunk_index=0, content="FAQ snippet", score=0.88),
        ]
    )

    conversation_id, completion = await workflow.run(
        conversation_id="conv-123",
        message="What does the guide say?",
        metadata={"source": "thread"},
        provider=provider,
        repository=repository,
        tool_registry=tool_registry,
        pricing_model="gpt-4.1-mini",
        input_price_per_1m_tokens=0.40,
        output_price_per_1m_tokens=1.60,
    )

    assert conversation_id == "conv-123"
    assert completion == ChatCompletion(
        content="Grounded answer",
        provider="openai",
        model="gpt-4.1-mini",
        metadata=ChatCompletionMetadata(
            citations=[
                ChatCitation(
                    document_id="doc-1",
                    filename="doc-1.md",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=len("Guide snippet"),
                    snippet="Guide snippet",
                ),
                ChatCitation(
                    document_id="doc-2",
                    filename="doc-2.md",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=len("FAQ snippet"),
                    snippet="FAQ snippet",
                ),
            ],
            tool_runs=[
                ToolRun(
                    tool_call_id="tool-1",
                    tool_name="search_knowledge_base",
                    status="completed",
                    input={"query": "guide answer", "top_k": 2},
                    output={
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
                                "score": 0.88,
                            },
                        ]
                    },
                    error=None,
                )
            ],
            usage=TokenUsage(input_tokens=140, output_tokens=45, total_tokens=185),
            cost=UsageCost(
                input_cost_usd=0.000056,
                output_cost_usd=0.000072,
                total_cost_usd=0.000128,
            ),
        ),
        response_id="resp-2",
    )
    expected_tool_output = json.dumps(
        {
            "result": {
                "hits": [
                    {
                        "chunk_index": 0,
                        "document_id": "doc-1",
                        "end_offset": len("Guide snippet"),
                        "filename": "doc-1.md",
                        "score": 0.91,
                        "snippet": "Guide snippet",
                        "start_offset": 0,
                    },
                    {
                        "chunk_index": 0,
                        "document_id": "doc-2",
                        "end_offset": len("FAQ snippet"),
                        "filename": "doc-2.md",
                        "score": 0.88,
                        "snippet": "FAQ snippet",
                        "start_offset": 0,
                    },
                ]
            },
            "status": "completed",
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    assert provider.calls[1]["previous_response_id"] == "resp-1"
    assert provider.calls[1]["tool_outputs"] == [
        ToolResultMessage(
            call_id="tool-1",
            output=expected_tool_output,
        )
    ]
    assert repository.tool_runs[-1] == SavedToolRun(
        conversation_id="conv-123",
        tool_call_id="tool-1",
        tool_name="search_knowledge_base",
        status="completed",
        input_payload={"query": "guide answer", "top_k": 2},
        output_payload=completion.metadata.tool_runs[0].output,
        error_message=None,
    )
    assert repository.saved_exchanges[-1] == SavedExchange(
        conversation_id="conv-123",
        user_message="What does the guide say?",
        user_metadata={"source": "thread"},
        assistant_message="Grounded answer",
    )


@pytest.mark.anyio
async def test_chat_workflow_passes_request_metadata_to_current_user_profile_tool() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    provider = ScriptedProvider(
        [
            ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-profile",
                        name="get_current_user_profile",
                        arguments={},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-profile-1",
                usage=TokenUsage(input_tokens=30, output_tokens=6, total_tokens=36),
            ),
            ChatCompletion(
                content="You are on the pro plan.",
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-profile-2",
                metadata=ChatCompletionMetadata(
                    usage=TokenUsage(input_tokens=18, output_tokens=8, total_tokens=26)
                ),
            ),
        ]
    )
    tool_registry = make_tool_registry()

    conversation_id, completion = await workflow.run(
        conversation_id="conv-profile",
        message="What plan am I on?",
        metadata={
            "user_profile": {
                "user_id": "user-123",
                "display_name": "Alice",
                "plan": "pro",
                "preferences": {"timezone": "UTC"},
            }
        },
        provider=provider,
        repository=repository,
        tool_registry=tool_registry,
        pricing_model="gpt-4.1-mini",
        input_price_per_1m_tokens=0.40,
        output_price_per_1m_tokens=1.60,
    )

    assert conversation_id == "conv-profile"
    assert completion.content == "You are on the pro plan."
    assert completion.metadata is not None
    assert completion.metadata.citations == []
    assert completion.metadata.tool_runs == [
        ToolRun(
            tool_call_id="tool-profile",
            tool_name="get_current_user_profile",
            status="completed",
            input={},
            output={
                "found": True,
                "profile": {
                    "user_id": "user-123",
                    "display_name": "Alice",
                    "email": None,
                    "plan": "pro",
                    "locale": None,
                    "preferences": {"timezone": "UTC"},
                },
            },
            error=None,
        )
    ]
    assert provider.calls[1]["previous_response_id"] == "resp-profile-1"
    assert provider.calls[1]["tool_outputs"] == [
        ToolResultMessage(
            call_id="tool-profile",
            output=(
                '{"result":{"found":true,"profile":{"display_name":"Alice","email":null,'
                '"locale":null,"plan":"pro","preferences":{"timezone":"UTC"},'
                '"user_id":"user-123"}},"status":"completed"}'
            ),
        )
    ]
    assert repository.tool_runs[-1] == SavedToolRun(
        conversation_id="conv-profile",
        tool_call_id="tool-profile",
        tool_name="get_current_user_profile",
        status="completed",
        input_payload={},
        output_payload=completion.metadata.tool_runs[0].output,
        error_message=None,
    )


@pytest.mark.anyio
async def test_chat_workflow_streams_tool_events_and_persists_on_completion() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    provider = ScriptedProvider(
        [
            ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-1",
                        name="calculator",
                        arguments={"expression": "2 + 2"},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-1",
                usage=TokenUsage(input_tokens=50, output_tokens=10, total_tokens=60),
            ),
            ChatCompletion(
                content="The answer is 4.",
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-2",
                metadata=ChatCompletionMetadata(
                    usage=TokenUsage(input_tokens=20, output_tokens=12, total_tokens=32)
                ),
            ),
        ]
    )
    tool_registry = make_tool_registry()

    events = [
        event
        async for event in workflow.stream(
            conversation_id="conv-123",
            message="What is 2 + 2?",
            metadata={"source": "test"},
            provider=provider,
            repository=repository,
            tool_registry=tool_registry,
            pricing_model="gpt-4.1-mini",
            input_price_per_1m_tokens=0.40,
            output_price_per_1m_tokens=1.60,
        )
    ]

    assert events == [
        WorkflowMessageStart(type="message_start", conversation_id="conv-123"),
        WorkflowToolStart(
            type="tool_start",
            conversation_id="conv-123",
            tool_call_id="tool-1",
            tool_name="calculator",
            input={"expression": "2 + 2"},
        ),
        WorkflowToolComplete(
            type="tool_complete",
            conversation_id="conv-123",
            tool_call_id="tool-1",
            tool_name="calculator",
            status="completed",
            output={"result": 4},
        ),
        WorkflowMessageDelta(type="message_delta", delta="The answer is 4."),
        WorkflowMessageComplete(
            type="message_complete",
            conversation_id="conv-123",
            assistant_message="The answer is 4.",
            provider="openai",
            model="gpt-4.1-mini",
            metadata={
                "citations": [],
                "tool_runs": [
                    {
                        "tool_call_id": "tool-1",
                        "tool_name": "calculator",
                        "status": "completed",
                        "input": {"expression": "2 + 2"},
                        "output": {"result": 4},
                        "error": None,
                    }
                ],
                "usage": {
                    "input_tokens": 70,
                    "output_tokens": 22,
                    "total_tokens": 92,
                },
                "cost": {
                    "input_cost_usd": 0.000028,
                    "output_cost_usd": 0.0000352,
                    "total_cost_usd": 0.0000632,
                    "currency": "USD",
                },
            },
        ),
    ]
    assert repository.saved_exchanges == [
        SavedExchange(
            conversation_id="conv-123",
            user_message="What is 2 + 2?",
            user_metadata={"source": "test"},
            assistant_message="The answer is 4.",
        )
    ]


@pytest.mark.anyio
async def test_chat_workflow_stream_emits_error_event_and_skips_persistence_on_failure() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    provider = FailingAfterToolProvider()
    tool_registry = make_tool_registry()
    received_events: list[dict[str, Any]] = []

    with pytest.raises(ChatProviderError, match="LLM provider request failed"):
        async for event in workflow.stream(
            conversation_id="conv-123",
            message="New question",
            metadata=None,
            provider=provider,
            repository=repository,
            tool_registry=tool_registry,
        ):
            received_events.append(event)

    assert received_events == [
        WorkflowMessageStart(type="message_start", conversation_id="conv-123"),
        WorkflowToolStart(
            type="tool_start",
            conversation_id="conv-123",
            tool_call_id="tool-1",
            tool_name="calculator",
            input={"expression": "2 + 2"},
        ),
        WorkflowToolComplete(
            type="tool_complete",
            conversation_id="conv-123",
            tool_call_id="tool-1",
            tool_name="calculator",
            status="completed",
            output={"result": 4},
        ),
    ]
    assert repository.saved_exchanges == []


@pytest.mark.anyio
async def test_chat_workflow_raises_after_exceeding_tool_round_limit() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    provider = ScriptedProvider(
        [
            ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-1",
                        name="calculator",
                        arguments={"expression": "1 + 1"},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-1",
            ),
            ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id="tool-2",
                        name="calculator",
                        arguments={"expression": "2 + 2"},
                    )
                ],
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-2",
            ),
        ]
    )
    tool_registry = make_tool_registry()

    with pytest.raises(ChatProviderError, match="tool call limit"):
        await workflow.run(
            conversation_id="conv-123",
            message="Loop forever",
            metadata=None,
            provider=provider,
            repository=repository,
            tool_registry=tool_registry,
            tool_max_rounds=1,
        )
