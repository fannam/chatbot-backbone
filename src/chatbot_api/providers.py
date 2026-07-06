from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from openai import APIError, APITimeoutError, AsyncOpenAI
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from openai.types.responses.response_input_item import FunctionCallOutput
from openai.types.responses.response_output_message import ResponseOutputMessage
from pydantic import BaseModel, ConfigDict

from chatbot_api.settings import Settings
from chatbot_api.tracing import NoopTraceSink, TraceSink


class ChatProviderError(Exception):
    """Base provider error."""


class ChatProviderConfigurationError(ChatProviderError):
    """Raised when the provider is not configured correctly."""


class ChatProviderTimeoutError(ChatProviderError):
    """Raised when the upstream model times out."""


@dataclass(frozen=True)
class ChatCitation:
    document_id: str
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    snippet: str


@dataclass(frozen=True)
class ToolRun:
    tool_call_id: str
    tool_name: str
    status: Literal["completed", "failed", "rejected", "timed_out"]
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class UsageCost:
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    currency: Literal["USD"] = "USD"


@dataclass(frozen=True)
class ChatCompletionMetadata:
    citations: list[ChatCitation] = field(default_factory=list)
    tool_runs: list[ToolRun] = field(default_factory=list)
    usage: TokenUsage | None = None
    cost: UsageCost | None = None


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    provider: str
    model: str
    metadata: ChatCompletionMetadata | None = None
    response_id: str | None = None


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ProviderToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCallRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResultMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    output: str


@dataclass(frozen=True)
class ToolCallBatch:
    tool_calls: list[ToolCallRequest]
    provider: str
    model: str
    response_id: str
    usage: TokenUsage | None = None


ChatProviderResult = ChatCompletion | ToolCallBatch


class ChatProvider(Protocol):
    async def generate_response(
        self,
        messages: Sequence[ChatTurn],
        *,
        tools: Sequence[ProviderToolDefinition] = (),
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolResultMessage] = (),
    ) -> ChatProviderResult: ...


async def check_message_moderation(client: AsyncOpenAI, text: str, *, model: str) -> bool:
    response = await client.moderations.create(input=text, model=model)
    return any(result.flagged for result in response.results)


class OpenAIChatProvider:
    provider_name = "openai"

    def __init__(self, settings: Settings, *, trace_sink: TraceSink | None = None) -> None:
        if not settings.openai_api_key:
            raise ChatProviderConfigurationError(
                "OPENAI_API_KEY is required to use the OpenAI chat provider"
            )

        self._model = settings.openai_model
        self._trace_sink = trace_sink or NoopTraceSink()
        self._raw_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        self._client = self._trace_sink.wrap_openai_client(self._raw_client)

    async def aclose(self) -> None:
        await self._raw_client.close()

    @property
    def raw_client(self) -> AsyncOpenAI:
        return self._raw_client

    async def generate_response(
        self,
        messages: Sequence[ChatTurn],
        *,
        tools: Sequence[ProviderToolDefinition] = (),
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolResultMessage] = (),
    ) -> ChatProviderResult:
        span = self._trace_sink.start_span(
            "provider.generate_response",
            inputs={
                "message_count": len(messages),
                "tool_count": len(tools),
                "has_previous_response_id": previous_response_id is not None,
                "tool_output_count": len(tool_outputs),
            },
            metadata={
                "provider": self.provider_name,
                "configured_model": self._model,
            },
            tags=["provider:openai"],
        )
        with span:
            try:
                response = await self._client.responses.create(
                    model=self._model,
                    input=self._build_input(messages, tool_outputs, previous_response_id),
                    previous_response_id=previous_response_id,
                    tools=[self._serialize_tool(tool) for tool in tools],
                    parallel_tool_calls=False,
                )
            except APITimeoutError as exc:
                span.annotate(metadata={"outcome": "timeout"})
                raise ChatProviderTimeoutError("LLM request timed out") from exc
            except APIError as exc:
                span.annotate(metadata={"outcome": "error"})
                raise ChatProviderError("LLM provider request failed") from exc

            tool_calls = [
                self._parse_tool_call(item)
                for item in response.output
                if isinstance(item, ResponseFunctionToolCall)
            ]
            usage = extract_usage(response)
            if tool_calls:
                span.finish_success(
                    outputs={
                        "response_id": response.id,
                        "model": response.model,
                        "outcome": "tool_calls",
                        "tool_call_count": len(tool_calls),
                        "usage": None if usage is None else usage.__dict__,
                    }
                )
                return ToolCallBatch(
                    tool_calls=tool_calls,
                    provider=self.provider_name,
                    model=response.model,
                    response_id=response.id,
                    usage=usage,
                )

            content = extract_response_text(response.output).strip()
            if not content:
                span.annotate(metadata={"outcome": "empty_response"})
                raise ChatProviderError("LLM provider returned an empty response")

            span.finish_success(
                outputs={
                    "response_id": response.id,
                    "model": response.model,
                    "outcome": "completed",
                    "content_chars": len(content),
                    "usage": None if usage is None else usage.__dict__,
                }
            )
            return ChatCompletion(
                content=content,
                provider=self.provider_name,
                model=response.model,
                metadata=ChatCompletionMetadata(usage=usage),
                response_id=response.id,
            )

    def _build_input(
        self,
        messages: Sequence[ChatTurn],
        tool_outputs: Sequence[ToolResultMessage],
        previous_response_id: str | None,
    ) -> list[dict[str, Any] | FunctionCallOutput]:
        if previous_response_id is not None:
            return [
                FunctionCallOutput(
                    call_id=tool_output.call_id,
                    output=tool_output.output,
                    type="function_call_output",
                )
                for tool_output in tool_outputs
            ]

        return [
            {
                "type": "message",
                "role": turn.role,
                "content": [
                    {
                        "type": "input_text",
                        "text": turn.content,
                    }
                ],
            }
            for turn in messages
        ]

    def _serialize_tool(self, tool: ProviderToolDefinition) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            strict=True,
        )

    def _parse_tool_call(self, tool_call: ResponseFunctionToolCall) -> ToolCallRequest:
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as exc:
            raise ChatProviderError("LLM provider returned invalid tool arguments") from exc
        if not isinstance(arguments, dict):
            raise ChatProviderError("LLM provider returned invalid tool arguments")

        return ToolCallRequest(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=arguments,
        )


def extract_response_text(items: Sequence[Any]) -> str:
    segments: list[str] = []
    for item in items:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for content in item.content:
            if content.type == "output_text" and content.text:
                segments.append(content.text)
    return "".join(segments)


def extract_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = normalize_usage_value(getattr(usage, "input_tokens", None))
    output_tokens = normalize_usage_value(getattr(usage, "output_tokens", None))
    total_tokens = normalize_usage_value(getattr(usage, "total_tokens", None))

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    resolved_input_tokens = 0 if input_tokens is None else input_tokens
    resolved_output_tokens = 0 if output_tokens is None else output_tokens
    resolved_total_tokens = total_tokens
    if resolved_total_tokens is None:
        resolved_total_tokens = resolved_input_tokens + resolved_output_tokens

    return TokenUsage(
        input_tokens=resolved_input_tokens,
        output_tokens=resolved_output_tokens,
        total_tokens=resolved_total_tokens,
    )


def normalize_usage_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
