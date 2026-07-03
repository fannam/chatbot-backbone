from __future__ import annotations

import pytest

from chatbot_api.main import RateLimitMiddleware
from chatbot_api.settings import Settings


def make_http_scope(
    *,
    path: str = "/chat",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("1.2.3.4", 12345),
) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "client": client,
    }


async def dummy_app(scope, receive, send) -> None:
    del scope, receive
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def noop_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


@pytest.mark.anyio
async def test_allows_requests_within_limit() -> None:
    middleware = RateLimitMiddleware(
        dummy_app,
        settings=Settings(rate_limit_requests_per_minute=2),
    )
    scope = make_http_scope()

    for _ in range(2):
        sent_messages: list[dict] = []

        async def send(message, sent_messages=sent_messages) -> None:
            sent_messages.append(message)

        await middleware(scope, noop_receive, send)
        assert sent_messages[0]["status"] == 200


@pytest.mark.anyio
async def test_rejects_requests_past_limit_with_retry_after() -> None:
    middleware = RateLimitMiddleware(
        dummy_app,
        settings=Settings(rate_limit_requests_per_minute=1),
    )
    scope = make_http_scope()

    first_messages: list[dict] = []

    async def first_send(message) -> None:
        first_messages.append(message)

    await middleware(scope, noop_receive, first_send)
    assert first_messages[0]["status"] == 200

    second_messages: list[dict] = []

    async def second_send(message) -> None:
        second_messages.append(message)

    await middleware(scope, noop_receive, second_send)
    assert second_messages[0]["status"] == 429
    headers = dict(second_messages[0]["headers"])
    assert b"retry-after" in headers
    assert int(headers[b"retry-after"]) >= 1


@pytest.mark.anyio
async def test_health_and_metrics_paths_are_exempt() -> None:
    middleware = RateLimitMiddleware(
        dummy_app,
        settings=Settings(rate_limit_requests_per_minute=1),
    )

    for path in ("/health", "/metrics"):
        scope = make_http_scope(path=path)
        for _ in range(3):
            sent_messages: list[dict] = []

            async def send(message, sent_messages=sent_messages) -> None:
                sent_messages.append(message)

            await middleware(scope, noop_receive, send)
            assert sent_messages[0]["status"] == 200


@pytest.mark.anyio
async def test_distinct_api_keys_have_independent_limits() -> None:
    middleware = RateLimitMiddleware(
        dummy_app,
        settings=Settings(rate_limit_requests_per_minute=1),
    )
    scope_a = make_http_scope(headers=[(b"x-api-key", b"key-a")])
    scope_b = make_http_scope(headers=[(b"x-api-key", b"key-b")])

    for scope in (scope_a, scope_b):
        sent_messages: list[dict] = []

        async def send(message, sent_messages=sent_messages) -> None:
            sent_messages.append(message)

        await middleware(scope, noop_receive, send)
        assert sent_messages[0]["status"] == 200


@pytest.mark.anyio
async def test_non_http_scope_passes_through() -> None:
    calls: list[str] = []

    async def lifespan_app(scope, receive, send) -> None:
        del receive, send
        calls.append(scope["type"])

    middleware = RateLimitMiddleware(
        lifespan_app,
        settings=Settings(rate_limit_requests_per_minute=1),
    )
    scope = {"type": "lifespan"}

    async def receive():
        raise AssertionError("receive should not be called for non-http scopes")

    async def send(message) -> None:
        del message
        raise AssertionError("send should not be called for non-http scopes")

    await middleware(scope, receive, send)

    assert calls == ["lifespan"]
