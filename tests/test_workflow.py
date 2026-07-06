from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.runtime import Runtime

from chatbot_api import workflow as workflow_module
from chatbot_api.guardrails import AsyncGuard, build_output_guard
from chatbot_api.memory import MemoryManager, extract_rule_based_memories
from chatbot_api.models import utcnow
from chatbot_api.observability import ObservabilityService
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
from chatbot_api.repositories import (
    ConversationSummaryRecord,
    MemoryRecord,
    MessageRecord,
    PersistedExchange,
    RetrievedDocumentChunk,
)
from chatbot_api.settings import Settings
from chatbot_api.tools import ToolRegistry, build_tool_registry
from chatbot_api.tracing import NoopTraceSink
from chatbot_api.workflow import (
    MAX_METADATA_CITATIONS,
    WorkflowMessageComplete,
    WorkflowMessageDelta,
    WorkflowMessageStart,
    WorkflowToolComplete,
    WorkflowToolStart,
    build_chat_workflow,
    call_model_node,
    initial_workflow_state,
    load_context_node,
    load_memory_node,
    merge_citations,
    output_guardrail_node,
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
        self.message_records_by_conversation: dict[str, list[MessageRecord]] = {}
        self.saved_exchanges: list[SavedExchange] = []
        self.tool_runs: list[SavedToolRun] = []
        self._next_message_id = 1

    async def list_messages(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[ChatTurn]:
        del owner_user_id
        return list(self.messages_by_conversation.get(conversation_id, []))

    async def list_message_records(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[MessageRecord]:
        del owner_user_id
        return list(self.message_records_by_conversation.get(conversation_id, []))

    async def create_tool_run(
        self,
        *,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        input_payload: dict[str, Any],
        owner_user_id: str | None = None,
    ):
        del owner_user_id
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
        owner_user_id: str | None = None,
    ):
        del owner_user_id
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
        owner_user_id: str | None = None,
    ):
        del owner_user_id
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
        owner_user_id: str | None = None,
    ) -> PersistedExchange:
        del owner_user_id
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
        created_at = utcnow()
        records = self.message_records_by_conversation.setdefault(conversation_id, [])
        user_message_id = self._next_message_id
        records.append(
            MessageRecord(
                id=user_message_id,
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                metadata=user_metadata,
                created_at=created_at,
            )
        )
        self._next_message_id += 1
        assistant_message_id = self._next_message_id
        records.append(
            MessageRecord(
                id=assistant_message_id,
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_message,
                metadata=None,
                created_at=created_at,
            )
        )
        self._next_message_id += 1
        return PersistedExchange(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            created_at=created_at,
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


class InMemoryMemoryRepository:
    def __init__(self) -> None:
        self.summary_by_conversation: dict[str, ConversationSummaryRecord] = {}
        self.memories_by_user: dict[str, list[MemoryRecord]] = {}
        self.upserted_memories: list[MemoryRecord] = []
        self._next_memory_id = 1

    async def get_conversation_summary(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ):
        del owner_user_id
        return self.summary_by_conversation.get(conversation_id)

    async def upsert_conversation_summary(
        self,
        *,
        conversation_id: str,
        summary_text: str,
        last_summarized_message_id: int,
        owner_user_id: str | None = None,
    ) -> ConversationSummaryRecord:
        del owner_user_id
        now = utcnow()
        summary = ConversationSummaryRecord(
            conversation_id=conversation_id,
            summary_text=summary_text,
            last_summarized_message_id=last_summarized_message_id,
            created_at=now,
            updated_at=now,
        )
        self.summary_by_conversation[conversation_id] = summary
        return summary

    async def list_active_memories(
        self,
        user_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ) -> list[MemoryRecord]:
        if owner_user_id is not None and owner_user_id != user_id:
            return []
        return list(self.memories_by_user.get(user_id, []))[:limit]

    async def upsert_memory(
        self,
        *,
        user_id: str,
        kind: str,
        key: str,
        value_json: dict[str, Any],
        confidence: float,
        source_message_id: int,
        extraction_method: str,
        owner_user_id: str | None = None,
    ) -> MemoryRecord:
        del owner_user_id
        now = utcnow()
        existing = None
        for memory in self.memories_by_user.setdefault(user_id, []):
            if memory.key == key and memory.deleted_at is None:
                existing = memory
                break
        if existing is None:
            memory = MemoryRecord(
                id=self._next_memory_id,
                user_id=user_id,
                kind=kind,
                key=key,
                value_json=value_json,
                confidence=confidence,
                source_message_id=source_message_id,
                extraction_method=extraction_method,
                created_at=now,
                updated_at=now,
                deleted_at=None,
            )
            self._next_memory_id += 1
            self.memories_by_user[user_id].append(memory)
        else:
            memory = MemoryRecord(
                id=existing.id,
                user_id=user_id,
                kind=kind,
                key=key,
                value_json=value_json,
                confidence=confidence,
                source_message_id=source_message_id,
                extraction_method=extraction_method,
                created_at=existing.created_at,
                updated_at=now,
                deleted_at=None,
            )
            index = self.memories_by_user[user_id].index(existing)
            self.memories_by_user[user_id][index] = memory
        self.upserted_memories.append(memory)
        return memory

    async def delete_memory(
        self,
        *,
        user_id: str,
        memory_id: int,
        owner_user_id: str | None = None,
    ) -> bool:
        if owner_user_id is not None and owner_user_id != user_id:
            return False
        memories = self.memories_by_user.get(user_id, [])
        for index, memory in enumerate(memories):
            if memory.id != memory_id:
                continue
            now = utcnow()
            memories[index] = MemoryRecord(
                id=memory.id,
                user_id=memory.user_id,
                kind=memory.kind,
                key=memory.key,
                value_json=memory.value_json,
                confidence=memory.confidence,
                source_message_id=memory.source_message_id,
                extraction_method=memory.extraction_method,
                created_at=memory.created_at,
                updated_at=now,
                deleted_at=now,
            )
            return True
        return False


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
        self.queries: list[tuple[str, int, int, str | None]] = []

    async def retrieve_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chunks_per_document: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[RetrievedDocumentChunk]:
        self.queries.append((query, top_k or 0, max_chunks_per_document or 0, owner_user_id))
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
    memory_manager: MemoryManager | None = None,
    output_guard: AsyncGuard | None = None,
    tool_max_rounds: int = 4,
    pricing_model: str | None = "gpt-4.1-mini",
    input_price_per_1m_tokens: float | None = 0.40,
    output_price_per_1m_tokens: float | None = 1.60,
    observability: ObservabilityService | None = None,
) -> Runtime[dict[str, Any]]:
    return Runtime(
        context={
            "provider": provider,
            "repository": repository,
            "tool_registry": tool_registry,
            "memory_manager": memory_manager,
            "output_guard": output_guard,
            "tool_max_rounds": tool_max_rounds,
            "observability": observability,
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
        serialize_turn(workflow_module.build_base_system_message()),
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
async def test_output_guardrail_node_passes_through_when_guard_is_none() -> None:
    repository = InMemoryChatRepository()
    provider = ScriptedProvider([])
    state = initial_workflow_state(
        conversation_id="conv-123", message="Hello", metadata=None, stream=False
    )
    state["assistant_message"] = "Hello from the workflow"

    updates = await output_guardrail_node(
        state, build_runtime(provider, repository, output_guard=None)
    )

    assert updates == {}


@pytest.mark.anyio
async def test_output_guardrail_node_redacts_pii_in_assistant_message() -> None:
    repository = InMemoryChatRepository()
    provider = ScriptedProvider([])
    guard = build_output_guard(
        output_guardrails_enabled=True,
        moderation_client=None,
        moderation_model="omni-moderation-latest",
    )
    state = initial_workflow_state(
        conversation_id="conv-123", message="Hello", metadata=None, stream=False
    )
    state["assistant_message"] = "you can reach me at jane@example.com"

    updates = await output_guardrail_node(
        state, build_runtime(provider, repository, output_guard=guard)
    )

    assert updates == {"assistant_message": "you can reach me at [REDACTED_EMAIL]"}


@pytest.mark.anyio
async def test_output_guardrail_node_substitutes_refusal_message_on_hard_block() -> None:
    class AlwaysBlockGuard:
        async def validate(self, value: str):
            from chatbot_api.guardrails import GuardrailsValidationError

            raise GuardrailsValidationError("blocked for test")

    repository = InMemoryChatRepository()
    provider = ScriptedProvider([])
    state = initial_workflow_state(
        conversation_id="conv-123", message="Hello", metadata=None, stream=False
    )
    state["assistant_message"] = "some response"

    updates = await output_guardrail_node(
        state, build_runtime(provider, repository, output_guard=AlwaysBlockGuard())
    )

    assert updates == {"assistant_message": workflow_module.OUTPUT_GUARDRAIL_REFUSAL_MESSAGE}


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

    assert updates == {"persisted_user_message_id": 1}
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
async def test_chat_workflow_applies_output_guard_after_tool_execution_round() -> None:
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
            ChatCompletion(
                content="you can reach support at help@example.com",
                provider="openai",
                model="gpt-4.1-mini",
                response_id="resp-2",
            ),
        ]
    )
    tool_registry = make_tool_registry()
    output_guard = build_output_guard(
        output_guardrails_enabled=True,
        moderation_client=None,
        moderation_model="omni-moderation-latest",
    )

    conversation_id, completion = await workflow.run(
        conversation_id="conv-guard",
        message="Loop with a tool then answer",
        metadata=None,
        provider=provider,
        repository=repository,
        tool_registry=tool_registry,
        output_guard=output_guard,
    )

    assert conversation_id == "conv-guard"
    assert completion.content == "you can reach support at [REDACTED_EMAIL]"
    assert repository.saved_exchanges[-1].assistant_message == (
        "you can reach support at [REDACTED_EMAIL]"
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
async def test_chat_workflow_persists_synthetic_message_after_exceeding_tool_round_limit() -> (
    None
):
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

    conversation_id, completion = await workflow.run(
        conversation_id="conv-123",
        message="Loop forever",
        metadata=None,
        provider=provider,
        repository=repository,
        tool_registry=tool_registry,
        tool_max_rounds=1,
    )

    assert conversation_id == "conv-123"
    assert completion.content == workflow_module.TOOL_ROUND_LIMIT_MESSAGE
    assert len(repository.saved_exchanges) == 1
    assert (
        repository.saved_exchanges[0].assistant_message
        == workflow_module.TOOL_ROUND_LIMIT_MESSAGE
    )


@pytest.mark.anyio
async def test_call_model_node_returns_synthetic_message_when_tool_round_limit_reached() -> (
    None
):
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
            )
        ]
    )
    state = initial_workflow_state(
        conversation_id="conv-123",
        message="Loop forever",
        metadata=None,
        stream=False,
    )
    state["provider_messages"] = [serialize_turn(ChatTurn(role="user", content="Loop forever"))]
    state["tool_rounds"] = 1

    updates = await call_model_node(
        state, build_runtime(provider, repository, tool_max_rounds=1)
    )

    assert updates["assistant_message"] == workflow_module.TOOL_ROUND_LIMIT_MESSAGE
    assert updates["pending_tool_calls"] == []
    assert updates["provider_name"] == "openai"
    assert updates["model_name"] == "gpt-4.1-mini"


@pytest.mark.anyio
async def test_load_memory_node_injects_summary_and_active_memories_into_provider_messages(
) -> None:
    repository = InMemoryChatRepository()
    now = utcnow()
    repository.messages_by_conversation["conv-memory"] = [
        ChatTurn(role="user", content="Old question"),
        ChatTurn(role="assistant", content="Old answer"),
        ChatTurn(role="user", content="Recent question"),
    ]
    repository.message_records_by_conversation["conv-memory"] = [
        MessageRecord(
            id=1,
            conversation_id="conv-memory",
            role="user",
            content="Old question",
            metadata=None,
            created_at=now,
        ),
        MessageRecord(
            id=2,
            conversation_id="conv-memory",
            role="assistant",
            content="Old answer",
            metadata=None,
            created_at=now,
        ),
        MessageRecord(
            id=3,
            conversation_id="conv-memory",
            role="user",
            content="Recent question",
            metadata=None,
            created_at=now,
        ),
    ]
    memory_repository = InMemoryMemoryRepository()
    memory_repository.summary_by_conversation["conv-memory"] = ConversationSummaryRecord(
        conversation_id="conv-memory",
        summary_text="Summary of older context.",
        last_summarized_message_id=2,
        created_at=now,
        updated_at=now,
    )
    memory_repository.memories_by_user["user-123"] = [
        MemoryRecord(
            id=1,
            user_id="user-123",
            kind="preference",
            key="preferences.language",
            value_json={"value": "Vietnamese"},
            confidence=0.92,
            source_message_id=1,
            extraction_method="rule",
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
    ]
    memory_manager = MemoryManager(
        provider=ScriptedProvider([]),
        chat_repository=repository,
        memory_repository=memory_repository,
        settings=Settings(memory_recent_message_window=6),
        trace_sink=NoopTraceSink(),
    )
    state = initial_workflow_state(
        conversation_id="conv-memory",
        message="Current question",
        metadata={"user_profile": {"user_id": "user-123"}},
        stream=False,
    )
    state.update(await load_context_node(state, build_runtime(ScriptedProvider([]), repository)))

    updates = await load_memory_node(
        state,
        build_runtime(
            ScriptedProvider([]),
            repository,
            memory_manager=memory_manager,
        ),
    )

    provider_messages = [
        workflow_module.deserialize_turn(turn) for turn in updates["provider_messages"]
    ]
    assert provider_messages[0].role == "system"
    assert provider_messages[0] == workflow_module.build_base_system_message()
    assert provider_messages[1].role == "system"
    assert "Summary of older context." in provider_messages[1].content
    assert provider_messages[2].role == "system"
    assert "Preferred language: Vietnamese" in provider_messages[2].content
    assert provider_messages[3:] == [
        ChatTurn(role="user", content="Recent question"),
        ChatTurn(role="user", content="Current question"),
    ]


@pytest.mark.anyio
async def test_prepare_prompt_includes_base_system_message_when_memory_disabled() -> None:
    repository = InMemoryChatRepository()
    memory_manager = MemoryManager(
        provider=ScriptedProvider([]),
        chat_repository=repository,
        memory_repository=InMemoryMemoryRepository(),
        settings=Settings(memory_enabled=False),
        trace_sink=NoopTraceSink(),
    )

    state = await memory_manager.prepare_prompt(
        conversation_id="conv-disabled",
        history_records=[],
        user_message="Hello",
        user_metadata=None,
    )

    assert state.provider_messages[0] == workflow_module.build_base_system_message()
    assert state.provider_messages[-1] == ChatTurn(role="user", content="Hello")


@pytest.mark.anyio
async def test_chat_workflow_writes_long_term_memory_after_persist() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    memory_repository = InMemoryMemoryRepository()
    provider = ScriptedProvider(
        [
            ChatCompletion(
                content="I will remember that.",
                provider="openai",
                model="gpt-4.1-mini",
            ),
            ChatCompletion(
                content=(
                    '{"memories":[{"kind":"profile","key":"profile.company",'
                    '"value":"Example Inc","confidence":0.8}]}'
                ),
                provider="openai",
                model="gpt-4.1-mini",
            ),
        ]
    )
    memory_manager = MemoryManager(
        provider=provider,
        chat_repository=repository,
        memory_repository=memory_repository,
        settings=Settings(memory_summary_trigger_messages=20),
        trace_sink=NoopTraceSink(),
    )

    conversation_id, completion = await workflow.run(
        conversation_id="conv-memory-write",
        message="I work at Example Inc.",
        metadata={"user_profile": {"user_id": "user-123"}},
        provider=provider,
        repository=repository,
        memory_manager=memory_manager,
    )

    assert conversation_id == "conv-memory-write"
    assert completion.content == "I will remember that."
    assert repository.saved_exchanges[0].user_message == "I work at Example Inc."
    assert memory_repository.upserted_memories[0].key == "profile.company"
    assert memory_repository.upserted_memories[0].value_json == {"value": "Example Inc"}


@pytest.mark.anyio
async def test_chat_workflow_ignores_invalid_llm_memory_payload_and_returns_completion() -> None:
    workflow = build_chat_workflow()
    repository = InMemoryChatRepository()
    memory_repository = InMemoryMemoryRepository()
    provider = ScriptedProvider(
        [
            ChatCompletion(
                content="Answer still succeeds.",
                provider="openai",
                model="gpt-4.1-mini",
            ),
            ChatCompletion(
                content="not-json",
                provider="openai",
                model="gpt-4.1-mini",
            ),
        ]
    )
    memory_manager = MemoryManager(
        provider=provider,
        chat_repository=repository,
        memory_repository=memory_repository,
        settings=Settings(memory_summary_trigger_messages=20),
        trace_sink=NoopTraceSink(),
    )

    _, completion = await workflow.run(
        conversation_id="conv-memory-invalid",
        message="I work at Example Inc.",
        metadata={"user_profile": {"user_id": "user-123"}},
        provider=provider,
        repository=repository,
        memory_manager=memory_manager,
    )

    assert completion.content == "Answer still succeeds."
    assert memory_repository.upserted_memories == []


def make_citation(document_id: str, chunk_index: int):
    return {
        "document_id": document_id,
        "filename": f"{document_id}.md",
        "chunk_index": chunk_index,
        "start_offset": 0,
        "end_offset": 10,
        "snippet": "snippet",
    }


def test_merge_citations_does_not_exceed_cap_when_existing_is_already_at_cap() -> None:
    existing = [make_citation("doc", index) for index in range(MAX_METADATA_CITATIONS)]
    additions = [make_citation("doc", MAX_METADATA_CITATIONS)]

    merged = merge_citations(existing, additions)

    assert len(merged) == MAX_METADATA_CITATIONS
    assert merged == existing


def test_merge_citations_stops_exactly_at_cap() -> None:
    existing = [make_citation("doc", 0)]
    additions = [make_citation("doc", index) for index in range(1, MAX_METADATA_CITATIONS + 2)]

    merged = merge_citations(existing, additions)

    assert len(merged) == MAX_METADATA_CITATIONS


def test_extract_rule_based_memories_captures_stable_preferences() -> None:
    candidates = extract_rule_based_memories(
        "Call me Alice. Respond in Vietnamese. My timezone is Asia/Ho_Chi_Minh. Be concise."
    )

    assert {candidate.key for candidate in candidates} == {
        "profile.preferred_name",
        "preferences.language",
        "preferences.timezone",
        "preferences.response_style",
    }


def test_extract_rule_based_memories_ignores_reported_third_person_speech() -> None:
    candidates = extract_rule_based_memories("She told me to call me back later.")

    assert "profile.preferred_name" not in {candidate.key for candidate in candidates}


def test_extract_rule_based_memories_ignores_quoted_reported_name() -> None:
    candidates = extract_rule_based_memories('He said, "my name is John."')

    assert "profile.preferred_name" not in {candidate.key for candidate in candidates}


def test_extract_rule_based_memories_ignores_reported_timezone_and_language() -> None:
    candidates = extract_rule_based_memories(
        "They mentioned they use timezone America/New_York"
    )

    assert "preferences.timezone" not in {candidate.key for candidate in candidates}
