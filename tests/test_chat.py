from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from chatbot_api.auth import AuthenticatedUser
from chatbot_api.main import app, get_auth_repository, get_authenticated_user, get_chat_service
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletion,
    ChatCompletionMetadata,
    ChatProviderError,
    ChatProviderTimeoutError,
    TokenUsage,
    ToolRun,
    UsageCost,
)
from chatbot_api.services import (
    ChatService,
    ChatStreamChunk,
    ChatStreamComplete,
    ChatStreamStart,
    ChatStreamToolComplete,
    ChatStreamToolStart,
)
from chatbot_api.settings import Settings, get_settings


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


class StubChatService:
    def __init__(
        self,
        *,
        completion: ChatCompletion | None = None,
        stream_events: list[Any] | None = None,
        chat_error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self._completion = completion
        self._stream_events = stream_events or []
        self._chat_error = chat_error
        self._stream_error = stream_error
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
        owner_user_id: str | None = None,
    ) -> tuple[str, ChatCompletion]:
        self.chat_calls.append(
            {
                "conversation_id": conversation_id,
                "message": message,
                "metadata": metadata,
                "owner_user_id": owner_user_id,
            }
        )
        if self._chat_error is not None:
            raise self._chat_error
        return conversation_id or "generated-conv", self._completion  # type: ignore[return-value]

    async def stream_chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        metadata: dict[str, Any] | None,
        owner_user_id: str | None = None,
    ) -> AsyncIterator[Any]:
        self.stream_calls.append(
            {
                "conversation_id": conversation_id,
                "message": message,
                "metadata": metadata,
                "owner_user_id": owner_user_id,
            }
        )
        for event in self._stream_events:
            yield event
        if self._stream_error is not None:
            raise self._stream_error


def build_chat_service_override(service: StubChatService):
    async def override() -> ChatService:
        return service  # type: ignore[return-value]

    return override


def build_authenticated_user_override(user: AuthenticatedUser | None):
    async def override() -> AuthenticatedUser | None:
        return user

    return override


def build_settings_override(settings: Settings):
    def override() -> Settings:
        return settings

    return override


class StubAuthRepository:
    def __init__(self, user: AuthenticatedUser | None) -> None:
        self.user = user
        self.calls: list[str] = []

    async def authenticate_api_key(self, api_key: str) -> AuthenticatedUser | None:
        self.calls.append(api_key)
        return self.user


def build_auth_repository_override(repository: StubAuthRepository):
    async def override() -> StubAuthRepository:
        return repository

    return override


async def collect_sse_events(response) -> list[SSEEvent]:
    events: list[SSEEvent] = []
    event_name: str | None = None
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if not line:
            if event_name is not None:
                events.append(
                    SSEEvent(
                        event=event_name,
                        data=json.loads("\n".join(data_lines)) if data_lines else {},
                    )
                )
            event_name = None
            data_lines = []
            continue

        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))

    if event_name is not None:
        events.append(
            SSEEvent(
                event=event_name,
                data=json.loads("\n".join(data_lines)) if data_lines else {},
            )
        )

    return events


@pytest.fixture
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_chat_returns_assistant_message_and_tool_metadata(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        completion=ChatCompletion(
            content="Grounded answer",
            provider="openai",
            model="gpt-4.1-mini",
            metadata=ChatCompletionMetadata(
                citations=[
                    ChatCitation(
                        document_id="doc-1",
                        filename="guide.md",
                        chunk_index=0,
                        start_offset=0,
                        end_offset=20,
                        snippet="Guide snippet",
                    )
                ],
                tool_runs=[
                    ToolRun(
                        tool_call_id="tool-1",
                        tool_name="search_knowledge_base",
                        status="completed",
                        input={"query": "guide"},
                        output={"hits": []},
                        error=None,
                    )
                ],
                usage=TokenUsage(input_tokens=120, output_tokens=30, total_tokens=150),
                cost=UsageCost(
                    input_cost_usd=0.000048,
                    output_cost_usd=0.000048,
                    total_cost_usd=0.000096,
                ),
            ),
        )
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    response = await async_client.post(
        "/chat",
        json={"message": "What does the guide say?"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "conversation_id": "generated-conv",
        "message": {
            "role": "assistant",
            "content": "Grounded answer",
        },
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "metadata": {
            "citations": [
                {
                    "document_id": "doc-1",
                    "filename": "guide.md",
                    "chunk_index": 0,
                    "start_offset": 0,
                    "end_offset": 20,
                    "snippet": "Guide snippet",
                }
            ],
            "tool_runs": [
                {
                    "tool_call_id": "tool-1",
                    "tool_name": "search_knowledge_base",
                    "status": "completed",
                    "input": {"query": "guide"},
                    "output": {"hits": []},
                }
            ],
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            },
            "cost": {
                "input_cost_usd": 4.8e-05,
                "output_cost_usd": 4.8e-05,
                "total_cost_usd": 9.6e-05,
                "currency": "USD",
            },
        },
    }


@pytest.mark.anyio
async def test_chat_streams_tool_events_and_completion_metadata(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        stream_events=[
            ChatStreamStart(conversation_id="conv-123"),
            ChatStreamToolStart(
                conversation_id="conv-123",
                tool_call_id="tool-1",
                tool_name="calculator",
                input={"expression": "2 + 2"},
            ),
            ChatStreamToolComplete(
                conversation_id="conv-123",
                tool_call_id="tool-1",
                tool_name="calculator",
                status="completed",
                output={"result": 4},
            ),
            ChatStreamChunk(delta="The answer is 4."),
            ChatStreamComplete(
                conversation_id="conv-123",
                completion=ChatCompletion(
                    content="The answer is 4.",
                    provider="openai",
                    model="gpt-4.1-mini",
                    metadata=ChatCompletionMetadata(
                        citations=[],
                        tool_runs=[
                            ToolRun(
                                tool_call_id="tool-1",
                                tool_name="calculator",
                                status="completed",
                                input={"expression": "2 + 2"},
                                output={"result": 4},
                                error=None,
                            )
                        ],
                        usage=TokenUsage(input_tokens=90, output_tokens=15, total_tokens=105),
                        cost=UsageCost(
                            input_cost_usd=0.000036,
                            output_cost_usd=0.000024,
                            total_cost_usd=0.00006,
                        ),
                    ),
                ),
            ),
        ]
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    async with async_client.stream(
        "POST",
        "/chat",
        json={"message": "What is 2 + 2?", "stream": True},
    ) as response:
        events = await collect_sse_events(response)

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events == [
        SSEEvent(event="message_start", data={"conversation_id": "conv-123"}),
        SSEEvent(
            event="tool_start",
            data={
                "conversation_id": "conv-123",
                "tool_call_id": "tool-1",
                "tool_name": "calculator",
                "input": {"expression": "2 + 2"},
            },
        ),
        SSEEvent(
            event="tool_complete",
            data={
                "conversation_id": "conv-123",
                "tool_call_id": "tool-1",
                "tool_name": "calculator",
                "status": "completed",
                "output": {"result": 4},
            },
        ),
        SSEEvent(event="message_delta", data={"delta": "The answer is 4."}),
        SSEEvent(
            event="message_complete",
            data={
                "conversation_id": "conv-123",
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                },
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "metadata": {
                    "citations": [],
                    "tool_runs": [
                            {
                                "tool_call_id": "tool-1",
                                "tool_name": "calculator",
                                "status": "completed",
                                "input": {"expression": "2 + 2"},
                                "output": {"result": 4},
                            }
                        ],
                    "usage": {
                        "input_tokens": 90,
                        "output_tokens": 15,
                        "total_tokens": 105,
                    },
                    "cost": {
                        "input_cost_usd": 3.6e-05,
                        "output_cost_usd": 2.4e-05,
                        "total_cost_usd": 6e-05,
                        "currency": "USD",
                    },
                    },
                },
        ),
    ]


@pytest.mark.anyio
async def test_chat_stream_emits_error_event_after_tool_activity(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        stream_events=[
            ChatStreamStart(conversation_id="conv-123"),
            ChatStreamToolStart(
                conversation_id="conv-123",
                tool_call_id="tool-1",
                tool_name="search_knowledge_base",
                input={"query": "guide"},
            ),
        ],
        stream_error=ChatProviderError("LLM provider request failed"),
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    async with async_client.stream(
        "POST",
        "/chat",
        json={"message": "What does the guide say?", "stream": True},
    ) as response:
        events = await collect_sse_events(response)

    assert response.status_code == status.HTTP_200_OK
    assert events == [
        SSEEvent(event="message_start", data={"conversation_id": "conv-123"}),
        SSEEvent(
            event="tool_start",
            data={
                "conversation_id": "conv-123",
                "tool_call_id": "tool-1",
                "tool_name": "search_knowledge_base",
                "input": {"query": "guide"},
            },
        ),
        SSEEvent(event="error", data={"detail": "LLM provider request failed"}),
    ]


@pytest.mark.anyio
async def test_chat_rejects_whitespace_only_message(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        completion=ChatCompletion(
            content="unused",
            provider="openai",
            model="gpt-4.1-mini",
        )
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    response = await async_client.post("/chat", json={"message": "   "})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.anyio
async def test_chat_maps_provider_timeout_to_gateway_timeout(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(chat_error=ChatProviderTimeoutError("LLM request timed out"))
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    response = await async_client.post("/chat", json={"message": "Hello"})

    assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
    assert response.json() == {"detail": "LLM request timed out"}


@pytest.mark.anyio
async def test_chat_stream_maps_provider_timeout_before_start_to_gateway_timeout(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(stream_error=ChatProviderTimeoutError("LLM request timed out"))
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)

    response = await async_client.post("/chat", json={"message": "Hello", "stream": True})

    assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
    assert response.json() == {"detail": "LLM request timed out"}


@pytest.mark.anyio
async def test_chat_requires_api_key_when_auth_is_enabled(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        completion=ChatCompletion(
            content="unused",
            provider="openai",
            model="gpt-4.1-mini",
        )
    )
    auth_repository = StubAuthRepository(None)
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)
    app.dependency_overrides[get_settings] = build_settings_override(Settings(auth_enabled=True))
    app.dependency_overrides[get_auth_repository] = build_auth_repository_override(auth_repository)

    response = await async_client.post("/chat", json={"message": "hello"})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "missing API key"}
    assert auth_repository.calls == []


@pytest.mark.anyio
async def test_chat_rejects_invalid_api_key_when_auth_is_enabled(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        completion=ChatCompletion(
            content="unused",
            provider="openai",
            model="gpt-4.1-mini",
        )
    )
    auth_repository = StubAuthRepository(None)
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)
    app.dependency_overrides[get_settings] = build_settings_override(Settings(auth_enabled=True))
    app.dependency_overrides[get_auth_repository] = build_auth_repository_override(auth_repository)

    response = await async_client.post(
        "/chat",
        json={"message": "hello"},
        headers={"X-API-Key": "bad-key"},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "invalid API key"}
    assert auth_repository.calls == ["bad-key"]


@pytest.mark.anyio
async def test_chat_overwrites_reserved_user_profile_metadata_from_authenticated_user(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        completion=ChatCompletion(
            content="Hello Alice",
            provider="openai",
            model="gpt-4.1-mini",
        )
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)
    app.dependency_overrides[get_authenticated_user] = build_authenticated_user_override(
        AuthenticatedUser(
            user_id="user-123",
            display_name="Alice",
            email="alice@example.com",
            plan="pro",
            locale="en-US",
            preferences={"timezone": "UTC"},
        )
    )

    response = await async_client.post(
        "/chat",
        json={
            "message": "hello",
            "metadata": {
                "source": "unit-test",
                "user_profile": {"user_id": "spoofed-user"},
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    assert service.chat_calls == [
        {
            "conversation_id": None,
            "message": "hello",
            "metadata": {
                "source": "unit-test",
                "user_profile": {
                    "user_id": "user-123",
                    "display_name": "Alice",
                    "email": "alice@example.com",
                    "plan": "pro",
                    "locale": "en-US",
                    "preferences": {"timezone": "UTC"},
                },
            },
            "owner_user_id": "user-123",
        }
    ]


@pytest.mark.anyio
async def test_chat_stream_passes_authenticated_owner_user_id(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    service = StubChatService(
        stream_events=[
            ChatStreamStart(conversation_id="conv-auth"),
            ChatStreamComplete(
                conversation_id="conv-auth",
                completion=ChatCompletion(
                    content="done",
                    provider="openai",
                    model="gpt-4.1-mini",
                ),
            ),
        ]
    )
    app.dependency_overrides[get_chat_service] = build_chat_service_override(service)
    app.dependency_overrides[get_authenticated_user] = build_authenticated_user_override(
        AuthenticatedUser(
            user_id="user-456",
            display_name="Bob",
            email=None,
            plan=None,
            locale=None,
            preferences={},
        )
    )

    async with async_client.stream(
        "POST",
        "/chat",
        json={"message": "hello", "stream": True},
    ) as response:
        await collect_sse_events(response)

    assert response.status_code == status.HTTP_200_OK
    assert service.stream_calls == [
        {
            "conversation_id": None,
            "message": "hello",
            "metadata": {
                "user_profile": {
                    "user_id": "user-456",
                    "display_name": "Bob",
                    "email": None,
                    "plan": None,
                    "locale": None,
                    "preferences": {},
                }
            },
            "owner_user_id": "user-456",
        }
    ]
