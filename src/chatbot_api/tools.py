from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from operator import add, floordiv, mod, mul, pow, sub, truediv
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatbot_api.observability import ObservabilityService
from chatbot_api.providers import (
    ChatCitation,
    ProviderToolDefinition,
    ToolCallRequest,
    ToolRun,
)
from chatbot_api.repositories import RetrievedDocumentChunk
from chatbot_api.retrieval import DocumentRetriever, build_citation
from chatbot_api.tracing import NoopTraceSink, TraceSink

ToolStatus = Literal["completed", "failed", "rejected", "timed_out"]
USER_PROFILE_METADATA_KEY = "user_profile"


class CalculatorToolInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    expression: str = Field(min_length=1)


class CalculatorToolOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: float | int


class CurrentUserProfileToolInput(BaseModel):
    model_config = ConfigDict(frozen=True)


class CurrentUserProfilePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    display_name: str | None = None
    email: str | None = None
    plan: str | None = None
    locale: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)


class CurrentUserProfileToolOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    found: bool
    profile: CurrentUserProfilePayload | None = None


class KnowledgeBaseSearchToolInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class KnowledgeBaseSearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    snippet: str
    score: float


class KnowledgeBaseSearchToolOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    hits: list[KnowledgeBaseSearchHit]


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_run: ToolRun
    model_output: str
    citations: list[ChatCitation] = field(default_factory=list)


@dataclass(frozen=True)
class ToolExecutionContext:
    conversation_id: str
    request_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolExecutionContext], Awaitable[BaseModel]]
    timeout_seconds: float

    def provider_definition(self) -> ProviderToolDefinition:
        return ProviderToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )


class ToolRegistry:
    def __init__(
        self,
        tools: Sequence[RegisteredTool],
        *,
        observability: ObservabilityService | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._observability = observability
        self._trace_sink = trace_sink or NoopTraceSink()

    def provider_definitions(self) -> list[ProviderToolDefinition]:
        return [tool.provider_definition() for tool in self._tools.values()]

    async def execute(
        self,
        tool_call: ToolCallRequest,
        *,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        trace_span = self._trace_sink.start_span(
            "tool.execute",
            run_type="tool",
            inputs={
                "tool_call_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "conversation_id": context.conversation_id,
            },
            tags=["tool", f"tool:{tool_call.name}"],
        )
        with trace_span:
            return await self._execute_with_trace(tool_call, context, trace_span)

    async def _execute_with_trace(
        self,
        tool_call: ToolCallRequest,
        context: ToolExecutionContext,
        trace_span,
    ) -> ToolExecutionResult:
        tool = self._tools.get(tool_call.name)
        if self._observability is not None:
            self._observability.log_event(
                "tool.execution.started",
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.name,
            )
        started_at = perf_counter()
        if tool is None:
            result = self._rejected_result(
                tool_call=tool_call,
                error_message=f"tool '{tool_call.name}' is not allowlisted",
            )
            self._record_tool_execution(result, duration_seconds=perf_counter() - started_at)
            trace_span.finish_success(
                outputs={
                    "tool_call_id": result.tool_run.tool_call_id,
                    "tool_name": result.tool_run.tool_name,
                    "status": result.tool_run.status,
                    "error": result.tool_run.error,
                }
            )
            return result

        try:
            validated_input = tool.input_model.model_validate(tool_call.arguments)
        except ValidationError as exc:
            result = self._rejected_result(
                tool_call=tool_call,
                error_message=f"invalid tool input: {exc.errors()[0]['msg']}",
            )
            self._record_tool_execution(result, duration_seconds=perf_counter() - started_at)
            trace_span.finish_success(
                outputs={
                    "tool_call_id": result.tool_run.tool_call_id,
                    "tool_name": result.tool_run.tool_name,
                    "status": result.tool_run.status,
                    "error": result.tool_run.error,
                }
            )
            return result

        try:
            output = await asyncio.wait_for(
                tool.handler(validated_input, context),
                timeout=tool.timeout_seconds,
            )
        except TimeoutError:
            result = ToolExecutionResult(
                tool_run=ToolRun(
                    tool_call_id=tool_call.call_id,
                    tool_name=tool_call.name,
                    status="timed_out",
                    input=validated_input.model_dump(mode="json"),
                    error="tool execution timed out",
                ),
                model_output=json.dumps(
                    {"status": "timed_out", "error": "tool execution timed out"},
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            self._record_tool_execution(result, duration_seconds=perf_counter() - started_at)
            trace_span.finish_success(
                outputs={
                    "tool_call_id": result.tool_run.tool_call_id,
                    "tool_name": result.tool_run.tool_name,
                    "status": result.tool_run.status,
                    "error": result.tool_run.error,
                }
            )
            return result
        except Exception as exc:
            result = ToolExecutionResult(
                tool_run=ToolRun(
                    tool_call_id=tool_call.call_id,
                    tool_name=tool_call.name,
                    status="failed",
                    input=validated_input.model_dump(mode="json"),
                    error=str(exc),
                ),
                model_output=json.dumps(
                    {"status": "failed", "error": str(exc)},
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            self._record_tool_execution(result, duration_seconds=perf_counter() - started_at)
            trace_span.finish_success(
                outputs={
                    "tool_call_id": result.tool_run.tool_call_id,
                    "tool_name": result.tool_run.tool_name,
                    "status": result.tool_run.status,
                    "error": result.tool_run.error,
                }
            )
            return result

        if not isinstance(output, tool.output_model):
            raise TypeError(f"tool '{tool.name}' returned an unexpected output model")

        citations = extract_tool_citations(output)
        output_payload = output.model_dump(mode="json")
        result = ToolExecutionResult(
            tool_run=ToolRun(
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.name,
                status="completed",
                input=validated_input.model_dump(mode="json"),
                output=output_payload,
            ),
            model_output=json.dumps(
                {"status": "completed", "result": output_payload},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            citations=citations,
        )
        self._record_tool_execution(result, duration_seconds=perf_counter() - started_at)
        trace_span.finish_success(
            outputs={
                "tool_call_id": result.tool_run.tool_call_id,
                "tool_name": result.tool_run.tool_name,
                "status": result.tool_run.status,
                "citation_count": len(citations),
            }
        )
        return result

    def _rejected_result(
        self,
        *,
        tool_call: ToolCallRequest,
        error_message: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_run=ToolRun(
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.name,
                status="rejected",
                input=dict(tool_call.arguments),
                error=error_message,
            ),
            model_output=json.dumps(
                {"status": "rejected", "error": error_message},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _record_tool_execution(
        self,
        result: ToolExecutionResult,
        *,
        duration_seconds: float,
    ) -> None:
        if self._observability is None:
            return

        self._observability.record_tool_call(
            tool_name=result.tool_run.tool_name,
            status=result.tool_run.status,
            duration_seconds=duration_seconds,
        )
        self._observability.log_event(
            "tool.execution.completed",
            level="info" if result.tool_run.status == "completed" else "warning",
            tool_call_id=result.tool_run.tool_call_id,
            tool_name=result.tool_run.tool_name,
            status=result.tool_run.status,
            duration_ms=duration_seconds * 1000,
            error=result.tool_run.error,
        )


def build_tool_registry(
    *,
    retriever: DocumentRetriever,
    search_top_k: int,
    timeout_seconds: float,
    observability: ObservabilityService | None = None,
    trace_sink: TraceSink | None = None,
) -> ToolRegistry:
    kb_tool = KnowledgeBaseSearchTool(
        retriever=retriever,
        default_top_k=search_top_k,
        max_top_k=search_top_k,
    )
    return ToolRegistry(
        [
            RegisteredTool(
                name="calculator",
                description="Evaluate a numeric arithmetic expression.",
                input_model=CalculatorToolInput,
                output_model=CalculatorToolOutput,
                handler=run_calculator_tool,
                timeout_seconds=timeout_seconds,
            ),
            RegisteredTool(
                name="get_current_user_profile",
                description="Return the current user profile from request metadata when available.",
                input_model=CurrentUserProfileToolInput,
                output_model=CurrentUserProfileToolOutput,
                handler=run_get_current_user_profile_tool,
                timeout_seconds=timeout_seconds,
            ),
            RegisteredTool(
                name="search_knowledge_base",
                description="Search indexed documents for relevant passages.",
                input_model=KnowledgeBaseSearchToolInput,
                output_model=KnowledgeBaseSearchToolOutput,
                handler=kb_tool.run,
                timeout_seconds=timeout_seconds,
            ),
        ],
        observability=observability,
        trace_sink=trace_sink,
    )


async def run_calculator_tool(
    payload: BaseModel,
    context: ToolExecutionContext,
) -> CalculatorToolOutput:
    del context
    if not isinstance(payload, CalculatorToolInput):
        raise TypeError("calculator tool received invalid payload")
    return CalculatorToolOutput(result=evaluate_expression(payload.expression))


async def run_get_current_user_profile_tool(
    payload: BaseModel,
    context: ToolExecutionContext,
) -> CurrentUserProfileToolOutput:
    if not isinstance(payload, CurrentUserProfileToolInput):
        raise TypeError("current user profile tool received invalid payload")

    metadata = context.request_metadata
    if not isinstance(metadata, dict):
        return CurrentUserProfileToolOutput(found=False, profile=None)

    raw_profile = metadata.get(USER_PROFILE_METADATA_KEY)
    if not isinstance(raw_profile, dict):
        return CurrentUserProfileToolOutput(found=False, profile=None)

    try:
        profile = CurrentUserProfilePayload.model_validate(raw_profile)
    except ValidationError:
        return CurrentUserProfileToolOutput(found=False, profile=None)

    return CurrentUserProfileToolOutput(found=True, profile=profile)


@dataclass(frozen=True)
class KnowledgeBaseSearchTool:
    retriever: DocumentRetriever
    default_top_k: int
    max_top_k: int

    async def run(
        self,
        payload: BaseModel,
        context: ToolExecutionContext,
    ) -> KnowledgeBaseSearchToolOutput:
        del context
        if not isinstance(payload, KnowledgeBaseSearchToolInput):
            raise TypeError("knowledge base search tool received invalid payload")

        top_k = min(payload.top_k or self.default_top_k, self.max_top_k)
        chunks = await self.retriever.retrieve_chunks(
            payload.query,
            top_k=top_k,
            max_chunks_per_document=top_k,
        )
        return KnowledgeBaseSearchToolOutput(
            hits=[chunk_to_search_hit(chunk) for chunk in chunks]
        )


def chunk_to_search_hit(chunk: RetrievedDocumentChunk) -> KnowledgeBaseSearchHit:
    citation = build_citation(chunk)
    return KnowledgeBaseSearchHit(
        document_id=chunk.document_id,
        filename=chunk.filename,
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        snippet=citation.snippet,
        score=chunk.score,
    )


def extract_tool_citations(output: BaseModel) -> list[ChatCitation]:
    if not isinstance(output, KnowledgeBaseSearchToolOutput):
        return []

    return [
        ChatCitation(
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            start_offset=hit.start_offset,
            end_offset=hit.end_offset,
            snippet=hit.snippet,
        )
        for hit in output.hits
    ]


ALLOWED_BINARY_OPERATORS = {
    ast.Add: add,
    ast.Sub: sub,
    ast.Mult: mul,
    ast.Div: truediv,
    ast.FloorDiv: floordiv,
    ast.Mod: mod,
    ast.Pow: pow,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def evaluate_expression(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc

    return _evaluate_ast_node(tree.body)


def _evaluate_ast_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp):
        operator = ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("unsupported arithmetic operator")
        return operator(_evaluate_ast_node(node.left), _evaluate_ast_node(node.right))

    if isinstance(node, ast.UnaryOp):
        operator = ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("unsupported arithmetic operator")
        return operator(_evaluate_ast_node(node.operand))

    raise ValueError("unsupported arithmetic expression")
