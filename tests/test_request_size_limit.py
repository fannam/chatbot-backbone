from __future__ import annotations

import pytest

from chatbot_api.main import RequestSizeLimitMiddleware, app
from chatbot_api.settings import Settings


def test_middleware_is_registered_on_the_app() -> None:
    registered_classes = [middleware.cls for middleware in app.user_middleware]
    assert RequestSizeLimitMiddleware in registered_classes


def make_http_scope(*, headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/documents",
        "headers": headers,
    }


async def dummy_app(scope, receive, send) -> None:
    del scope
    while True:
        message = await receive()
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.anyio
async def test_rejects_request_with_oversized_content_length_header() -> None:
    middleware = RequestSizeLimitMiddleware(
        dummy_app,
        settings=Settings(request_max_body_bytes=10),
    )
    scope = make_http_scope(headers=[(b"content-length", b"1000")])

    async def receive():
        raise AssertionError(
            "receive should not be called when content-length already exceeds the limit"
        )

    sent_messages: list[dict] = []

    async def send(message) -> None:
        sent_messages.append(message)

    await middleware(scope, receive, send)

    assert sent_messages[0]["status"] == 413


@pytest.mark.anyio
async def test_rejects_streamed_body_exceeding_limit_without_content_length_header() -> None:
    middleware = RequestSizeLimitMiddleware(
        dummy_app,
        settings=Settings(request_max_body_bytes=10),
    )
    scope = make_http_scope(headers=[])
    chunks = [
        {"type": "http.request", "body": b"x" * 6, "more_body": True},
        {"type": "http.request", "body": b"x" * 6, "more_body": False},
    ]

    async def receive():
        return chunks.pop(0)

    sent_messages: list[dict] = []

    async def send(message) -> None:
        sent_messages.append(message)

    await middleware(scope, receive, send)

    assert sent_messages[0]["status"] == 413


@pytest.mark.anyio
async def test_allows_request_within_limit() -> None:
    middleware = RequestSizeLimitMiddleware(
        dummy_app,
        settings=Settings(request_max_body_bytes=100),
    )
    scope = make_http_scope(headers=[(b"content-length", b"2")])
    chunks = [{"type": "http.request", "body": b"ok", "more_body": False}]

    async def receive():
        return chunks.pop(0)

    sent_messages: list[dict] = []

    async def send(message) -> None:
        sent_messages.append(message)

    await middleware(scope, receive, send)

    assert sent_messages[0]["status"] == 200


@pytest.mark.anyio
async def test_non_http_scope_passes_through() -> None:
    calls: list[str] = []

    async def lifespan_app(scope, receive, send) -> None:
        del receive, send
        calls.append(scope["type"])

    middleware = RequestSizeLimitMiddleware(
        lifespan_app,
        settings=Settings(request_max_body_bytes=10),
    )
    scope = {"type": "lifespan"}

    async def receive():
        raise AssertionError("receive should not be called for non-http scopes")

    async def send(message) -> None:
        del message
        raise AssertionError("send should not be called for non-http scopes")

    await middleware(scope, receive, send)

    assert calls == ["lifespan"]
