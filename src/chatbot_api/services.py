from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from chatbot_api.observability import ObservabilityService
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletion,
    ChatCompletionMetadata,
    ChatProvider,
    TokenUsage,
    ToolRun,
    UsageCost,
)
from chatbot_api.repositories import ChatRepository
from chatbot_api.tools import ToolRegistry
from chatbot_api.tracing import NoopTraceSink, TraceSink
from chatbot_api.workflow import ChatWorkflow, WorkflowStreamEvent, build_chat_workflow


@dataclass(frozen=True)
class ChatStreamStart:
    conversation_id: str


@dataclass(frozen=True)
class ChatStreamToolStart:
    conversation_id: str
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ChatStreamToolComplete:
    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: str
    output: dict[str, Any]


@dataclass(frozen=True)
class ChatStreamToolError:
    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: str
    error: str


@dataclass(frozen=True)
class ChatStreamChunk:
    delta: str


@dataclass(frozen=True)
class ChatStreamComplete:
    conversation_id: str
    completion: ChatCompletion


class ChatService:
    def __init__(
        self,
        provider: ChatProvider,
        repository: ChatRepository,
        workflow: ChatWorkflow | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_max_rounds: int = 4,
        observability: ObservabilityService | None = None,
        pricing_model: str | None = None,
        input_price_per_1m_tokens: float | None = None,
        output_price_per_1m_tokens: float | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._provider = provider
        self._repository = repository
        self._workflow = workflow or build_chat_workflow()
        self._tool_registry = tool_registry
        self._tool_max_rounds = tool_max_rounds
        self._observability = observability
        self._pricing_model = pricing_model
        self._input_price_per_1m_tokens = input_price_per_1m_tokens
        self._output_price_per_1m_tokens = output_price_per_1m_tokens
        self._trace_sink = trace_sink or NoopTraceSink()

    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[str, ChatCompletion]:
        resolved_conversation_id = conversation_id or str(uuid4())
        return await self._workflow.run(
            conversation_id=resolved_conversation_id,
            message=message,
            metadata=metadata,
            provider=self._provider,
            repository=self._repository,
            tool_registry=self._tool_registry,
            tool_max_rounds=self._tool_max_rounds,
            observability=self._observability,
            pricing_model=self._pricing_model,
            input_price_per_1m_tokens=self._input_price_per_1m_tokens,
            output_price_per_1m_tokens=self._output_price_per_1m_tokens,
            trace_sink=self._trace_sink,
        )

    async def stream_chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[
        ChatStreamStart
        | ChatStreamToolStart
        | ChatStreamToolComplete
        | ChatStreamToolError
        | ChatStreamChunk
        | ChatStreamComplete
    ]:
        resolved_conversation_id = conversation_id or str(uuid4())
        async for event in self._workflow.stream(
            conversation_id=resolved_conversation_id,
            message=message,
            metadata=metadata,
            provider=self._provider,
            repository=self._repository,
            tool_registry=self._tool_registry,
            tool_max_rounds=self._tool_max_rounds,
            observability=self._observability,
            pricing_model=self._pricing_model,
            input_price_per_1m_tokens=self._input_price_per_1m_tokens,
            output_price_per_1m_tokens=self._output_price_per_1m_tokens,
            trace_sink=self._trace_sink,
        ):
            yield deserialize_stream_event(event)


def deserialize_stream_event(
    event: WorkflowStreamEvent,
) -> (
    ChatStreamStart
    | ChatStreamToolStart
    | ChatStreamToolComplete
    | ChatStreamToolError
    | ChatStreamChunk
    | ChatStreamComplete
):
    event_type = event["type"]

    if event_type == "message_start":
        return ChatStreamStart(conversation_id=event["conversation_id"])

    if event_type == "tool_start":
        return ChatStreamToolStart(
            conversation_id=event["conversation_id"],
            tool_call_id=event["tool_call_id"],
            tool_name=event["tool_name"],
            input=event["input"],
        )

    if event_type == "tool_complete":
        return ChatStreamToolComplete(
            conversation_id=event["conversation_id"],
            tool_call_id=event["tool_call_id"],
            tool_name=event["tool_name"],
            status=event["status"],
            output=event["output"],
        )

    if event_type == "tool_error":
        return ChatStreamToolError(
            conversation_id=event["conversation_id"],
            tool_call_id=event["tool_call_id"],
            tool_name=event["tool_name"],
            status=event["status"],
            error=event["error"],
        )

    if event_type == "message_delta":
        return ChatStreamChunk(delta=event["delta"])

    return ChatStreamComplete(
        conversation_id=event["conversation_id"],
        completion=ChatCompletion(
            content=event["assistant_message"],
            provider=event["provider"],
            model=event["model"],
            metadata=deserialize_completion_metadata(event.get("metadata")),
        ),
    )


def deserialize_completion_metadata(
    metadata: dict[str, Any] | None,
) -> ChatCompletionMetadata | None:
    if metadata is None:
        return None

    citations = metadata.get("citations")
    tool_runs = metadata.get("tool_runs")
    usage = metadata.get("usage")
    cost = metadata.get("cost")
    return ChatCompletionMetadata(
        citations=[] if not isinstance(citations, list) else [
            ChatCitation(
                document_id=str(citation["document_id"]),
                filename=str(citation["filename"]),
                chunk_index=int(citation["chunk_index"]),
                start_offset=int(citation["start_offset"]),
                end_offset=int(citation["end_offset"]),
                snippet=str(citation["snippet"]),
            )
            for citation in citations
        ],
        tool_runs=[] if not isinstance(tool_runs, list) else [
            ToolRun(
                tool_call_id=str(tool_run["tool_call_id"]),
                tool_name=str(tool_run["tool_name"]),
                status=str(tool_run["status"]),
                input=dict(tool_run["input"]),
                output=None if tool_run.get("output") is None else dict(tool_run["output"]),
                error=None if tool_run.get("error") is None else str(tool_run["error"]),
            )
            for tool_run in tool_runs
        ],
        usage=deserialize_usage(usage),
        cost=deserialize_cost(cost),
    )


def deserialize_usage(metadata: Any) -> TokenUsage | None:
    if not isinstance(metadata, dict):
        return None
    return TokenUsage(
        input_tokens=int(metadata["input_tokens"]),
        output_tokens=int(metadata["output_tokens"]),
        total_tokens=int(metadata["total_tokens"]),
    )


def deserialize_cost(metadata: Any) -> UsageCost | None:
    if not isinstance(metadata, dict):
        return None
    return UsageCost(
        input_cost_usd=float(metadata["input_cost_usd"]),
        output_cost_usd=float(metadata["output_cost_usd"]),
        total_cost_usd=float(metadata["total_cost_usd"]),
        currency=str(metadata["currency"]),
    )
