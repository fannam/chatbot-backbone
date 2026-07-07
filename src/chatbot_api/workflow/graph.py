from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any, Literal, NotRequired, TypedDict, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from chatbot_api.memory import MemoryManager, build_base_system_message
from chatbot_api.observability import ObservabilityService
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletion,
    ChatCompletionMetadata,
    ChatProvider,
    ChatTurn,
    TokenUsage,
    ToolCallBatch,
    ToolCallRequest,
    ToolResultMessage,
    ToolRun,
    UsageCost,
    deserialize_cost,
    deserialize_usage,
)
from chatbot_api.repositories import ChatRepository
from chatbot_api.tracing import NoopTraceSink, TraceSink
from chatbot_api.workflow.guardrails import AsyncGuard, GuardrailsValidationError
from chatbot_api.workflow.tools import ToolExecutionContext, ToolRegistry

MAX_METADATA_CITATIONS = 4
TOOL_ROUND_LIMIT_MESSAGE = (
    "I wasn't able to finish gathering information for this request within the "
    "allotted number of tool calls. Here is what I found so far — please "
    "rephrase your question or ask me to continue."
)


class ChatWorkflowTurn(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatWorkflowCitation(TypedDict):
    document_id: str
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    snippet: str


class ChatWorkflowToolRun(TypedDict):
    tool_call_id: str
    tool_name: str
    status: Literal["completed", "failed", "rejected", "timed_out"]
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None


class ChatWorkflowUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatWorkflowCost(TypedDict):
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    currency: Literal["USD"]


class ChatWorkflowMetadata(TypedDict):
    citations: list[ChatWorkflowCitation]
    tool_runs: list[ChatWorkflowToolRun]
    usage: NotRequired[ChatWorkflowUsage]
    cost: NotRequired[ChatWorkflowCost]


class ChatWorkflowToolCall(TypedDict):
    call_id: str
    name: str
    arguments: dict[str, Any]


class ChatWorkflowToolOutput(TypedDict):
    call_id: str
    output: str


class ChatWorkflowState(TypedDict):
    conversation_id: str
    owner_user_id: str | None
    user_message: str
    user_metadata: dict[str, Any] | None
    stream: bool
    history: list[ChatWorkflowTurn]
    provider_messages: list[ChatWorkflowTurn]
    assistant_message: str | None
    provider_name: str | None
    model_name: str | None
    provider_response_id: str | None
    pending_tool_calls: list[ChatWorkflowToolCall]
    pending_tool_outputs: list[ChatWorkflowToolOutput]
    response_metadata: ChatWorkflowMetadata | None
    usage_totals: ChatWorkflowUsage | None
    cost_totals: ChatWorkflowCost | None
    persisted_user_message_id: int | None
    message_started: bool
    tool_rounds: int
    model_rounds: int


class ChatWorkflowContext(TypedDict):
    provider: ChatProvider
    repository: ChatRepository
    tool_registry: ToolRegistry | None
    memory_manager: MemoryManager | None
    output_guard: AsyncGuard | None
    tool_max_rounds: int
    observability: ObservabilityService | None
    trace_sink: TraceSink
    pricing_model: str | None
    input_price_per_1m_tokens: float | None
    output_price_per_1m_tokens: float | None


class WorkflowMessageStart(TypedDict):
    type: Literal["message_start"]
    conversation_id: str


class WorkflowToolStart(TypedDict):
    type: Literal["tool_start"]
    conversation_id: str
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]


class WorkflowToolComplete(TypedDict):
    type: Literal["tool_complete"]
    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: Literal["completed"]
    output: dict[str, Any]


class WorkflowToolError(TypedDict):
    type: Literal["tool_error"]
    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: Literal["failed", "rejected", "timed_out"]
    error: str


class WorkflowMessageDelta(TypedDict):
    type: Literal["message_delta"]
    delta: str


class WorkflowMessageComplete(TypedDict):
    type: Literal["message_complete"]
    conversation_id: str
    assistant_message: str
    provider: str
    model: str
    metadata: NotRequired[ChatWorkflowMetadata | None]


WorkflowStreamEvent = (
    WorkflowMessageStart
    | WorkflowToolStart
    | WorkflowToolComplete
    | WorkflowToolError
    | WorkflowMessageDelta
    | WorkflowMessageComplete
)


def serialize_turn(turn: ChatTurn) -> ChatWorkflowTurn:
    return ChatWorkflowTurn(role=turn.role, content=turn.content)


def deserialize_turn(turn: ChatWorkflowTurn) -> ChatTurn:
    return ChatTurn(role=turn["role"], content=turn["content"])


async def load_context_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    span = runtime.context["trace_sink"].start_span(
        "workflow.load_context",
        inputs={
            "conversation_id": state["conversation_id"],
            "stream": state["stream"],
        },
        metadata={"node": "load_context"},
        tags=["workflow", "node:load_context"],
    )
    with span:
        history = await runtime.context["repository"].list_messages(
            state["conversation_id"],
            owner_user_id=state["owner_user_id"],
        )
        provider_messages = [
            build_base_system_message(),
            *history,
            ChatTurn(role="user", content=state["user_message"]),
        ]
        updates = {
            "history": [serialize_turn(turn) for turn in history],
            "provider_messages": [serialize_turn(turn) for turn in provider_messages],
            "assistant_message": None,
            "provider_name": None,
            "model_name": None,
            "provider_response_id": None,
            "pending_tool_calls": [],
            "pending_tool_outputs": [],
            "response_metadata": None,
            "usage_totals": None,
            "cost_totals": None,
            "persisted_user_message_id": None,
            "message_started": False,
            "tool_rounds": 0,
            "model_rounds": 0,
        }
        span.finish_success(
            outputs={
                "history_count": len(history),
                "provider_message_count": len(provider_messages),
            }
        )
        return updates


async def load_memory_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    span = runtime.context["trace_sink"].start_span(
        "workflow.load_memory",
        inputs={"conversation_id": state["conversation_id"]},
        metadata={"node": "load_memory"},
        tags=["workflow", "node:load_memory"],
    )
    with span:
        memory_manager = runtime.context["memory_manager"]
        if memory_manager is None:
            span.finish_success(outputs={"memory_enabled": False})
            return {}

        try:
            history_records = await runtime.context["repository"].list_message_records(
                state["conversation_id"],
                owner_user_id=state["owner_user_id"],
            )
            prompt_state = await memory_manager.prepare_prompt(
                conversation_id=state["conversation_id"],
                history_records=history_records,
                user_message=state["user_message"],
                user_metadata=state["user_metadata"],
                owner_user_id=state["owner_user_id"],
            )
        except Exception as exc:
            annotate_memory_skipped_on_error(span, exc)
            return {}

        span.finish_success(
            outputs={
                "memory_enabled": True,
                "has_summary": prompt_state.summary is not None,
                "memory_count": len(prompt_state.active_memories),
                "provider_message_count": len(prompt_state.provider_messages),
            }
        )
        return {
            "provider_messages": [serialize_turn(turn) for turn in prompt_state.provider_messages]
        }


async def call_model_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    trace_sink = runtime.context["trace_sink"]
    node_span = trace_sink.start_span(
        "workflow.call_model",
        inputs={
            "conversation_id": state["conversation_id"],
            "stream": state["stream"],
            "existing_model_rounds": state["model_rounds"],
            "tool_round": state["tool_rounds"],
            "pending_tool_output_count": len(state["pending_tool_outputs"]),
        },
        metadata={"node": "call_model"},
        tags=["workflow", "node:call_model"],
    )
    with node_span:
        provider = runtime.context["provider"]
        tool_registry = runtime.context["tool_registry"]
        provider_messages = [deserialize_turn(turn) for turn in state["provider_messages"]]
        tool_outputs = [
            ToolResultMessage(call_id=item["call_id"], output=item["output"])
            for item in state["pending_tool_outputs"]
        ]
        round_index = state["model_rounds"] + 1
        llm_span = trace_sink.start_span(
            "workflow.llm_round",
            run_type="llm",
            inputs={
                "conversation_id": state["conversation_id"],
                "round_index": round_index,
                "tool_round": state["tool_rounds"],
                "message_count": len(provider_messages),
                "tool_count": (
                    0
                    if tool_registry is None
                    else len(tool_registry.provider_definitions())
                ),
                "tool_output_count": len(tool_outputs),
                "has_previous_response_id": state["provider_response_id"] is not None,
            },
            metadata={"provider": getattr(provider, "provider_name", "unknown")},
            tags=["workflow", "llm_round"],
        )
        started_at = perf_counter()
        with llm_span:
            try:
                result = await provider.generate_response(
                    provider_messages,
                    tools=[] if tool_registry is None else tool_registry.provider_definitions(),
                    previous_response_id=state["provider_response_id"],
                    tool_outputs=tool_outputs,
                )
            except Exception:
                record_llm_round(
                    observability=runtime.context["observability"],
                    conversation_id=state["conversation_id"],
                    provider_name=getattr(provider, "provider_name", "unknown"),
                    model_name=state["model_name"],
                    round_index=round_index,
                    tool_round=state["tool_rounds"],
                    outcome="error",
                    duration_seconds=perf_counter() - started_at,
                    usage=None,
                    cost=None,
                )
                llm_span.annotate(metadata={"outcome": "error"})
                raise

            usage = (
                result.usage
                if isinstance(result, ToolCallBatch)
                else extract_completion_usage(result)
            )
            cost = calculate_usage_cost(
                usage,
                model_name=result.model,
                pricing_model=runtime.context["pricing_model"],
                input_price_per_1m_tokens=runtime.context["input_price_per_1m_tokens"],
                output_price_per_1m_tokens=runtime.context["output_price_per_1m_tokens"],
            )
            if usage is not None and cost is None:
                record_cost_skip(
                    observability=runtime.context["observability"],
                    conversation_id=state["conversation_id"],
                    provider_name=result.provider,
                    model_name=result.model,
                    pricing_model=runtime.context["pricing_model"],
                    round_index=round_index,
                )

            usage_totals = accumulate_usage(state["usage_totals"], usage)
            cost_totals = accumulate_cost(state["cost_totals"], cost)
            metadata = state["response_metadata"]
            citations = [] if metadata is None else list(metadata["citations"])
            tool_runs = [] if metadata is None else list(metadata["tool_runs"])
            aggregated_metadata = build_metadata(
                citations,
                tool_runs,
                usage=usage_totals,
                cost=cost_totals,
            )
            tool_limit_exceeded = isinstance(result, ToolCallBatch) and (
                state["tool_rounds"] >= runtime.context["tool_max_rounds"]
            )
            outcome = (
                "tool_limit_exceeded"
                if tool_limit_exceeded
                else "tool_calls"
                if isinstance(result, ToolCallBatch)
                else "completed"
            )
            record_llm_round(
                observability=runtime.context["observability"],
                conversation_id=state["conversation_id"],
                provider_name=result.provider,
                model_name=result.model,
                round_index=round_index,
                tool_round=state["tool_rounds"],
                outcome=outcome,
                duration_seconds=perf_counter() - started_at,
                usage=usage,
                cost=cost,
            )

            updates: dict[str, Any] = {
                "provider_name": result.provider,
                "model_name": result.model,
                "provider_response_id": result.response_id,
                "pending_tool_outputs": [],
                "response_metadata": aggregated_metadata,
                "usage_totals": usage_totals,
                "cost_totals": cost_totals,
                "model_rounds": round_index,
            }
            if state["stream"] and not state["message_started"]:
                writer = get_stream_writer()
                writer(
                    WorkflowMessageStart(
                        type="message_start",
                        conversation_id=state["conversation_id"],
                    )
                )
                updates["message_started"] = True

            if isinstance(result, ToolCallBatch):
                if tool_limit_exceeded:
                    updates["assistant_message"] = TOOL_ROUND_LIMIT_MESSAGE
                    updates["pending_tool_calls"] = []
                    llm_span.finish_success(
                        outputs={
                            "provider": result.provider,
                            "model": result.model,
                            "outcome": outcome,
                            "tool_call_count": len(result.tool_calls),
                            "response_id": result.response_id,
                            "usage": None if usage is None else usage.__dict__,
                            "cost": None if cost is None else cost.__dict__,
                        }
                    )
                    node_span.finish_success(
                        outputs={
                            "outcome": outcome,
                            "round_index": round_index,
                            "assistant_message_chars": len(TOOL_ROUND_LIMIT_MESSAGE),
                        }
                    )
                    return updates

                updates["assistant_message"] = None
                updates["pending_tool_calls"] = [
                    serialize_tool_call(tool_call) for tool_call in result.tool_calls
                ]
                llm_span.finish_success(
                    outputs={
                        "provider": result.provider,
                        "model": result.model,
                        "outcome": outcome,
                        "tool_call_count": len(result.tool_calls),
                        "response_id": result.response_id,
                        "usage": None if usage is None else usage.__dict__,
                        "cost": None if cost is None else cost.__dict__,
                    }
                )
                node_span.finish_success(
                    outputs={
                        "outcome": outcome,
                        "round_index": round_index,
                        "pending_tool_call_count": len(result.tool_calls),
                    }
                )
                return updates

            updates["assistant_message"] = result.content
            updates["pending_tool_calls"] = []
            llm_span.finish_success(
                outputs={
                    "provider": result.provider,
                    "model": result.model,
                    "outcome": outcome,
                    "content_chars": len(result.content),
                    "response_id": result.response_id,
                    "usage": None if usage is None else usage.__dict__,
                    "cost": None if cost is None else cost.__dict__,
                }
            )
            node_span.finish_success(
                outputs={
                    "outcome": outcome,
                    "round_index": round_index,
                    "assistant_message_chars": len(result.content),
                }
            )
            return updates


async def execute_tools_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    tool_registry = runtime.context["tool_registry"]
    if tool_registry is None or not state["pending_tool_calls"]:
        return {"pending_tool_outputs": [], "pending_tool_calls": []}

    span = runtime.context["trace_sink"].start_span(
        "workflow.execute_tools",
        inputs={
            "conversation_id": state["conversation_id"],
            "pending_tool_call_count": len(state["pending_tool_calls"]),
            "tool_round": state["tool_rounds"] + 1,
        },
        metadata={"node": "execute_tools"},
        tags=["workflow", "node:execute_tools"],
    )
    with span:
        writer = get_stream_writer() if state["stream"] else None
        repository = runtime.context["repository"]
        metadata = state["response_metadata"]
        citations = [] if metadata is None else list(metadata["citations"])
        tool_runs = [] if metadata is None else list(metadata["tool_runs"])
        pending_outputs: list[ChatWorkflowToolOutput] = []

        for serialized_tool_call in state["pending_tool_calls"]:
            tool_call = deserialize_tool_call(serialized_tool_call)
            if writer is not None:
                writer(
                    WorkflowToolStart(
                        type="tool_start",
                        conversation_id=state["conversation_id"],
                        tool_call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )

            await repository.create_tool_run(
                conversation_id=state["conversation_id"],
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.name,
                input_payload=tool_call.arguments,
                owner_user_id=state["owner_user_id"],
            )
            result = await tool_registry.execute(
                tool_call,
                context=ToolExecutionContext(
                    conversation_id=state["conversation_id"],
                    owner_user_id=state["owner_user_id"],
                    request_metadata=state["user_metadata"],
                ),
            )

            pending_outputs.append(
                ChatWorkflowToolOutput(
                    call_id=tool_call.call_id,
                    output=result.model_output,
                )
            )
            tool_runs.append(serialize_tool_run(result.tool_run))
            citations = merge_citations(
                citations,
                [serialize_citation(citation) for citation in result.citations],
            )

            if result.tool_run.status == "completed":
                await repository.complete_tool_run(
                    conversation_id=state["conversation_id"],
                    tool_call_id=tool_call.call_id,
                    output_payload=result.tool_run.output or {},
                    owner_user_id=state["owner_user_id"],
                )
                if writer is not None:
                    writer(
                        WorkflowToolComplete(
                            type="tool_complete",
                            conversation_id=state["conversation_id"],
                            tool_call_id=result.tool_run.tool_call_id,
                            tool_name=result.tool_run.tool_name,
                            status="completed",
                            output=result.tool_run.output or {},
                        )
                    )
                continue

            await repository.fail_tool_run(
                conversation_id=state["conversation_id"],
                tool_call_id=tool_call.call_id,
                status=result.tool_run.status,
                error_message=result.tool_run.error or "tool execution failed",
                owner_user_id=state["owner_user_id"],
            )
            if writer is not None:
                writer(
                    WorkflowToolError(
                        type="tool_error",
                        conversation_id=state["conversation_id"],
                        tool_call_id=result.tool_run.tool_call_id,
                        tool_name=result.tool_run.tool_name,
                        status=result.tool_run.status,
                        error=result.tool_run.error or "tool execution failed",
                    )
                )

        updates = {
            "pending_tool_calls": [],
            "pending_tool_outputs": pending_outputs,
            "response_metadata": build_metadata(
                citations,
                tool_runs,
                usage=state["usage_totals"],
                cost=state["cost_totals"],
            ),
            "tool_rounds": state["tool_rounds"] + 1,
        }
        span.finish_success(
            outputs={
                "tool_round": state["tool_rounds"] + 1,
                "tool_run_count": len(tool_runs),
                "pending_tool_output_count": len(pending_outputs),
                "citation_count": len(citations),
            }
        )
        return updates


OUTPUT_GUARDRAIL_REFUSAL_MESSAGE = (
    "I'm not able to share that response as written. Please rephrase your "
    "request, or let me know if you'd like me to try answering differently."
)


async def output_guardrail_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    output_guard = runtime.context["output_guard"]
    if output_guard is None or state["assistant_message"] is None:
        return {}

    span = runtime.context["trace_sink"].start_span(
        "workflow.output_guardrail",
        inputs={"conversation_id": state["conversation_id"]},
        metadata={"node": "output_guardrail"},
        tags=["workflow", "node:output_guardrail"],
    )
    with span:
        try:
            outcome = await output_guard.validate(state["assistant_message"])
        except GuardrailsValidationError:
            record_guardrail_check(
                runtime.context["observability"],
                direction="output",
                check="moderation",
                outcome="blocked",
            )
            span.finish_success(outputs={"outcome": "blocked"})
            return {"assistant_message": OUTPUT_GUARDRAIL_REFUSAL_MESSAGE}

        redacted = outcome.validated_output
        if redacted != state["assistant_message"]:
            record_guardrail_check(
                runtime.context["observability"],
                direction="output",
                check="pii",
                outcome="redacted",
            )
            span.finish_success(outputs={"outcome": "redacted"})
            return {"assistant_message": redacted}

        span.finish_success(outputs={"outcome": "passed"})
        return {}


def record_guardrail_check(
    observability: ObservabilityService | None,
    *,
    direction: str,
    check: str,
    outcome: str,
) -> None:
    if observability is None:
        return
    observability.record_guardrail_check(direction=direction, check=check, outcome=outcome)
    event = "guardrail.output.blocked" if outcome == "blocked" else "guardrail.output.redacted"
    observability.log_event(event, level="warning" if outcome == "blocked" else "info", check=check)


async def persist_response_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    span = runtime.context["trace_sink"].start_span(
        "workflow.persist_response",
        inputs={
            "conversation_id": state["conversation_id"],
            "stream": state["stream"],
        },
        metadata={"node": "persist_response"},
        tags=["workflow", "node:persist_response"],
    )
    with span:
        assistant_message = state["assistant_message"]
        provider_name = state["provider_name"]
        model_name = state["model_name"]
        response_metadata = state["response_metadata"]

        if assistant_message is None or provider_name is None or model_name is None:
            raise ValueError("workflow completed without an assistant response")

        persisted_exchange = await runtime.context["repository"].append_exchange(
            conversation_id=state["conversation_id"],
            user_message=state["user_message"],
            user_metadata=state["user_metadata"],
            assistant_message=assistant_message,
            owner_user_id=state["owner_user_id"],
        )

        if state["stream"]:
            writer = get_stream_writer()
            writer(WorkflowMessageDelta(type="message_delta", delta=assistant_message))

            payload = WorkflowMessageComplete(
                type="message_complete",
                conversation_id=state["conversation_id"],
                assistant_message=assistant_message,
                provider=provider_name,
                model=model_name,
            )
            if response_metadata is not None:
                payload["metadata"] = response_metadata
            writer(payload)

        span.finish_success(
            outputs={
                "provider": provider_name,
                "model": model_name,
                "assistant_message_chars": len(assistant_message),
                "has_metadata": response_metadata is not None,
            }
        )
        return {"persisted_user_message_id": persisted_exchange.user_message_id}


async def write_memory_node(
    state: ChatWorkflowState,
    runtime: Runtime[ChatWorkflowContext],
) -> dict[str, Any]:
    span = runtime.context["trace_sink"].start_span(
        "workflow.write_memory",
        inputs={"conversation_id": state["conversation_id"]},
        metadata={"node": "write_memory"},
        tags=["workflow", "node:write_memory"],
    )
    with span:
        memory_manager = runtime.context["memory_manager"]
        user_message_id = state["persisted_user_message_id"]
        if memory_manager is None or user_message_id is None:
            span.finish_success(
                outputs={
                    "memory_enabled": memory_manager is not None,
                    "wrote_memory": False,
                }
            )
            return {}

        try:
            await memory_manager.write_after_persist(
                conversation_id=state["conversation_id"],
                user_message=state["user_message"],
                user_metadata=state["user_metadata"],
                user_message_id=user_message_id,
                owner_user_id=state["owner_user_id"],
            )
        except Exception as exc:
            annotate_memory_skipped_on_error(span, exc)
            return {}

        span.finish_success(outputs={"memory_enabled": True, "wrote_memory": True})
        return {}


def annotate_memory_skipped_on_error(span: Any, exc: Exception) -> None:
    span.annotate(
        metadata={
            "memory_enabled": True,
            "outcome": "skipped_on_error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    )


async def route_after_call_model(state: ChatWorkflowState) -> str:
    if state["pending_tool_calls"]:
        return "execute_tools"
    return "persist_response"


class ChatWorkflow:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def run(
        self,
        *,
        conversation_id: str,
        message: str,
        metadata: dict[str, Any] | None,
        owner_user_id: str | None = None,
        provider: ChatProvider,
        repository: ChatRepository,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        output_guard: AsyncGuard | None = None,
        tool_max_rounds: int = 4,
        observability: ObservabilityService | None = None,
        pricing_model: str | None = None,
        input_price_per_1m_tokens: float | None = None,
        output_price_per_1m_tokens: float | None = None,
        trace_sink: TraceSink | None = None,
    ) -> tuple[str, ChatCompletion]:
        resolved_trace_sink = trace_sink or NoopTraceSink()
        started_at = perf_counter()
        outcome = "failed"
        try:
            span = resolved_trace_sink.start_span(
                "workflow.run",
                inputs={
                    "conversation_id": conversation_id,
                    "message": message,
                    "stream": False,
                    "tool_max_rounds": tool_max_rounds,
                },
                metadata={"chat_mode": "sync"},
                tags=["workflow", "sync"],
            )
            with span:
                try:
                    state = cast(
                        ChatWorkflowState,
                        await self._graph.ainvoke(
                            initial_workflow_state(
                                conversation_id=conversation_id,
                                owner_user_id=owner_user_id,
                                message=message,
                                metadata=metadata,
                                stream=False,
                            ),
                            config=workflow_config(conversation_id),
                            context=build_workflow_context(
                                provider=provider,
                                repository=repository,
                                tool_registry=tool_registry,
                                memory_manager=memory_manager,
                                output_guard=output_guard,
                                tool_max_rounds=tool_max_rounds,
                                observability=observability,
                                trace_sink=resolved_trace_sink,
                                pricing_model=pricing_model,
                                input_price_per_1m_tokens=input_price_per_1m_tokens,
                                output_price_per_1m_tokens=output_price_per_1m_tokens,
                            ),
                        ),
                    )

                    assistant_message = state["assistant_message"]
                    provider_name = state["provider_name"]
                    model_name = state["model_name"]
                    response_metadata = deserialize_metadata(state["response_metadata"])
                    if assistant_message is None or provider_name is None or model_name is None:
                        raise ValueError("workflow completed without an assistant response")

                    outcome = "completed"
                    span.finish_success(
                        outputs={
                            "conversation_id": conversation_id,
                            "provider": provider_name,
                            "model": model_name,
                            "assistant_message_chars": len(assistant_message),
                            "tool_rounds": state["tool_rounds"],
                            "model_rounds": state["model_rounds"],
                        }
                    )
                    return conversation_id, ChatCompletion(
                        content=assistant_message,
                        provider=provider_name,
                        model=model_name,
                        metadata=response_metadata,
                        response_id=state["provider_response_id"],
                    )
                except Exception:
                    span.annotate(metadata={"outcome": "failed"})
                    raise
        finally:
            if observability is not None:
                observability.record_chat_workflow(
                    mode="sync",
                    outcome=outcome,
                    duration_seconds=perf_counter() - started_at,
                )

    async def stream(
        self,
        *,
        conversation_id: str,
        message: str,
        metadata: dict[str, Any] | None,
        owner_user_id: str | None = None,
        provider: ChatProvider,
        repository: ChatRepository,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        output_guard: AsyncGuard | None = None,
        tool_max_rounds: int = 4,
        observability: ObservabilityService | None = None,
        pricing_model: str | None = None,
        input_price_per_1m_tokens: float | None = None,
        output_price_per_1m_tokens: float | None = None,
        trace_sink: TraceSink | None = None,
    ) -> AsyncIterator[WorkflowStreamEvent]:
        resolved_trace_sink = trace_sink or NoopTraceSink()
        started_at = perf_counter()
        outcome = "interrupted"
        try:
            span = resolved_trace_sink.start_span(
                "workflow.stream",
                inputs={
                    "conversation_id": conversation_id,
                    "message": message,
                    "stream": True,
                    "tool_max_rounds": tool_max_rounds,
                },
                metadata={"chat_mode": "stream"},
                tags=["workflow", "stream"],
            )
            with span:
                try:
                    async for event in self._graph.astream(
                        initial_workflow_state(
                            conversation_id=conversation_id,
                            owner_user_id=owner_user_id,
                            message=message,
                            metadata=metadata,
                            stream=True,
                        ),
                        config=workflow_config(conversation_id),
                        context=build_workflow_context(
                            provider=provider,
                            repository=repository,
                            tool_registry=tool_registry,
                            memory_manager=memory_manager,
                            output_guard=output_guard,
                            tool_max_rounds=tool_max_rounds,
                            observability=observability,
                            trace_sink=resolved_trace_sink,
                            pricing_model=pricing_model,
                            input_price_per_1m_tokens=input_price_per_1m_tokens,
                            output_price_per_1m_tokens=output_price_per_1m_tokens,
                        ),
                        stream_mode="custom",
                    ):
                        yield cast(WorkflowStreamEvent, event)
                    outcome = "completed"
                    span.finish_success(
                        outputs={
                            "conversation_id": conversation_id,
                            "outcome": outcome,
                        }
                    )
                except Exception:
                    span.annotate(metadata={"outcome": "failed"})
                    outcome = "failed"
                    raise
        finally:
            if observability is not None:
                observability.record_chat_workflow(
                    mode="stream",
                    outcome=outcome,
                    duration_seconds=perf_counter() - started_at,
                )


def initial_workflow_state(
    *,
    conversation_id: str,
    message: str,
    metadata: dict[str, Any] | None,
    stream: bool,
    owner_user_id: str | None = None,
) -> ChatWorkflowState:
    return ChatWorkflowState(
        conversation_id=conversation_id,
        owner_user_id=owner_user_id,
        user_message=message,
        user_metadata=metadata,
        stream=stream,
        history=[],
        provider_messages=[],
        assistant_message=None,
        provider_name=None,
        model_name=None,
        provider_response_id=None,
        pending_tool_calls=[],
        pending_tool_outputs=[],
        response_metadata=None,
        usage_totals=None,
        cost_totals=None,
        persisted_user_message_id=None,
        message_started=False,
        tool_rounds=0,
        model_rounds=0,
    )


def workflow_config(conversation_id: str) -> dict[str, dict[str, str]]:
    return {
        "configurable": {
            "thread_id": conversation_id,
        }
    }


def build_workflow_context(
    *,
    provider: ChatProvider,
    repository: ChatRepository,
    tool_registry: ToolRegistry | None,
    memory_manager: MemoryManager | None,
    output_guard: AsyncGuard | None,
    tool_max_rounds: int,
    observability: ObservabilityService | None,
    trace_sink: TraceSink,
    pricing_model: str | None,
    input_price_per_1m_tokens: float | None,
    output_price_per_1m_tokens: float | None,
) -> ChatWorkflowContext:
    return {
        "provider": provider,
        "repository": repository,
        "tool_registry": tool_registry,
        "memory_manager": memory_manager,
        "output_guard": output_guard,
        "tool_max_rounds": tool_max_rounds,
        "observability": observability,
        "trace_sink": trace_sink,
        "pricing_model": pricing_model,
        "input_price_per_1m_tokens": input_price_per_1m_tokens,
        "output_price_per_1m_tokens": output_price_per_1m_tokens,
    }


def build_chat_workflow(*, checkpointer: Any | None = None) -> ChatWorkflow:
    graph_builder = StateGraph(ChatWorkflowState, context_schema=ChatWorkflowContext)
    graph_builder.add_node("load_context", load_context_node)
    graph_builder.add_node("load_memory", load_memory_node)
    graph_builder.add_node("call_model", call_model_node)
    graph_builder.add_node("execute_tools", execute_tools_node)
    graph_builder.add_node("output_guardrail", output_guardrail_node)
    graph_builder.add_node("persist_response", persist_response_node)
    graph_builder.add_node("write_memory", write_memory_node)
    graph_builder.add_edge(START, "load_context")
    graph_builder.add_edge("load_context", "load_memory")
    graph_builder.add_edge("load_memory", "call_model")
    graph_builder.add_conditional_edges(
        "call_model",
        route_after_call_model,
        {
            "execute_tools": "execute_tools",
            "persist_response": "output_guardrail",
        },
    )
    graph_builder.add_edge("execute_tools", "call_model")
    graph_builder.add_edge("output_guardrail", "persist_response")
    graph_builder.add_edge("persist_response", "write_memory")
    graph_builder.add_edge("write_memory", END)

    compiled_graph = graph_builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        name="chat_workflow",
    )
    return ChatWorkflow(compiled_graph)


def serialize_metadata(metadata: ChatCompletionMetadata | None) -> ChatWorkflowMetadata | None:
    if metadata is None:
        return None

    return build_metadata(
        citations=[serialize_citation(citation) for citation in metadata.citations],
        tool_runs=[serialize_tool_run(tool_run) for tool_run in metadata.tool_runs],
        usage=None if metadata.usage is None else serialize_usage(metadata.usage),
        cost=None if metadata.cost is None else serialize_cost(metadata.cost),
    )


def deserialize_metadata(metadata: ChatWorkflowMetadata | None) -> ChatCompletionMetadata | None:
    if metadata is None:
        return None

    return ChatCompletionMetadata(
        citations=[
            ChatCitation(
                document_id=citation["document_id"],
                filename=citation["filename"],
                chunk_index=citation["chunk_index"],
                start_offset=citation["start_offset"],
                end_offset=citation["end_offset"],
                snippet=citation["snippet"],
            )
            for citation in metadata["citations"]
        ],
        tool_runs=[
            ToolRun(
                tool_call_id=tool_run["tool_call_id"],
                tool_name=tool_run["tool_name"],
                status=tool_run["status"],
                input=tool_run["input"],
                output=tool_run["output"],
                error=tool_run["error"],
            )
            for tool_run in metadata["tool_runs"]
        ],
        usage=deserialize_usage(metadata.get("usage")),
        cost=deserialize_cost(metadata.get("cost")),
    )


def build_metadata(
    citations: list[ChatWorkflowCitation],
    tool_runs: list[ChatWorkflowToolRun],
    *,
    usage: ChatWorkflowUsage | None = None,
    cost: ChatWorkflowCost | None = None,
) -> ChatWorkflowMetadata | None:
    if not citations and not tool_runs and usage is None and cost is None:
        return None
    payload = ChatWorkflowMetadata(
        citations=citations,
        tool_runs=tool_runs,
    )
    if usage is not None:
        payload["usage"] = usage
    if cost is not None:
        payload["cost"] = cost
    return payload


def serialize_citation(citation: ChatCitation) -> ChatWorkflowCitation:
    return ChatWorkflowCitation(
        document_id=citation.document_id,
        filename=citation.filename,
        chunk_index=citation.chunk_index,
        start_offset=citation.start_offset,
        end_offset=citation.end_offset,
        snippet=citation.snippet,
    )


def serialize_tool_call(tool_call: ToolCallRequest) -> ChatWorkflowToolCall:
    return ChatWorkflowToolCall(
        call_id=tool_call.call_id,
        name=tool_call.name,
        arguments=tool_call.arguments,
    )


def deserialize_tool_call(tool_call: ChatWorkflowToolCall) -> ToolCallRequest:
    return ToolCallRequest(
        call_id=tool_call["call_id"],
        name=tool_call["name"],
        arguments=tool_call["arguments"],
    )


def serialize_tool_run(tool_run: ToolRun) -> ChatWorkflowToolRun:
    return ChatWorkflowToolRun(
        tool_call_id=tool_run.tool_call_id,
        tool_name=tool_run.tool_name,
        status=tool_run.status,
        input=tool_run.input,
        output=tool_run.output,
        error=tool_run.error,
    )


def serialize_usage(usage: TokenUsage) -> ChatWorkflowUsage:
    return ChatWorkflowUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def serialize_cost(cost: UsageCost) -> ChatWorkflowCost:
    return ChatWorkflowCost(
        input_cost_usd=cost.input_cost_usd,
        output_cost_usd=cost.output_cost_usd,
        total_cost_usd=cost.total_cost_usd,
        currency=cost.currency,
    )


def extract_completion_usage(completion: ChatCompletion) -> TokenUsage | None:
    if completion.metadata is None:
        return None
    return completion.metadata.usage


def accumulate_usage(
    existing: ChatWorkflowUsage | None,
    addition: TokenUsage | None,
) -> ChatWorkflowUsage | None:
    if addition is None:
        return existing
    if existing is None:
        return serialize_usage(addition)
    return ChatWorkflowUsage(
        input_tokens=existing["input_tokens"] + addition.input_tokens,
        output_tokens=existing["output_tokens"] + addition.output_tokens,
        total_tokens=existing["total_tokens"] + addition.total_tokens,
    )


def accumulate_cost(
    existing: ChatWorkflowCost | None,
    addition: UsageCost | None,
) -> ChatWorkflowCost | None:
    if addition is None:
        return existing
    if existing is None:
        return serialize_cost(addition)
    return ChatWorkflowCost(
        input_cost_usd=round(existing["input_cost_usd"] + addition.input_cost_usd, 12),
        output_cost_usd=round(existing["output_cost_usd"] + addition.output_cost_usd, 12),
        total_cost_usd=round(existing["total_cost_usd"] + addition.total_cost_usd, 12),
        currency="USD",
    )


def calculate_usage_cost(
    usage: TokenUsage | None,
    *,
    model_name: str,
    pricing_model: str | None,
    input_price_per_1m_tokens: float | None,
    output_price_per_1m_tokens: float | None,
) -> UsageCost | None:
    if usage is None:
        return None
    if (
        pricing_model is None
        or input_price_per_1m_tokens is None
        or output_price_per_1m_tokens is None
        or not model_matches_pricing(model_name, pricing_model)
    ):
        return None

    input_cost_usd = round((usage.input_tokens / 1_000_000) * input_price_per_1m_tokens, 12)
    output_cost_usd = round((usage.output_tokens / 1_000_000) * output_price_per_1m_tokens, 12)
    return UsageCost(
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=round(input_cost_usd + output_cost_usd, 12),
    )


def model_matches_pricing(model_name: str, pricing_model: str) -> bool:
    return model_name == pricing_model or model_name.startswith(f"{pricing_model}-")


def record_cost_skip(
    *,
    observability: ObservabilityService | None,
    conversation_id: str,
    provider_name: str,
    model_name: str,
    pricing_model: str | None,
    round_index: int,
) -> None:
    if observability is None:
        return
    observability.log_event(
        "llm.cost.skipped",
        level="warning",
        conversation_id=conversation_id,
        provider=provider_name,
        model=model_name,
        pricing_model=pricing_model,
        round_index=round_index,
    )


def record_llm_round(
    *,
    observability: ObservabilityService | None,
    conversation_id: str,
    provider_name: str,
    model_name: str | None,
    round_index: int,
    tool_round: int,
    outcome: str,
    duration_seconds: float,
    usage: TokenUsage | None,
    cost: UsageCost | None,
) -> None:
    if observability is None:
        return
    resolved_model_name = "unknown" if model_name is None else model_name
    observability.record_llm_request(
        model=resolved_model_name,
        outcome=outcome,
        duration_seconds=duration_seconds,
        usage=usage,
        cost=cost,
    )
    observability.log_event(
        "llm.request.completed" if outcome != "error" else "llm.request.failed",
        level="info" if outcome != "error" else "warning",
        conversation_id=conversation_id,
        provider=provider_name,
        model=resolved_model_name,
        round_index=round_index,
        tool_round=tool_round,
        outcome=outcome,
        duration_ms=duration_seconds * 1000,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        total_tokens=None if usage is None else usage.total_tokens,
        total_cost_usd=None if cost is None else cost.total_cost_usd,
    )


def merge_citations(
    existing: list[ChatWorkflowCitation],
    additions: list[ChatWorkflowCitation],
) -> list[ChatWorkflowCitation]:
    merged = list(existing)
    seen = {
        (citation["document_id"], citation["chunk_index"])
        for citation in existing
    }
    for citation in additions:
        if len(merged) >= MAX_METADATA_CITATIONS:
            break
        key = (citation["document_id"], citation["chunk_index"])
        if key in seen:
            continue
        merged.append(citation)
        seen.add(key)
    return merged
