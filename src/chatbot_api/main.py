from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter, time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import PlainTextResponse
from fastapi.sse import EventSourceResponse, format_sse_event
from langsmith.middleware import TracingMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import Headers, MutableHeaders

from chatbot_api.auth import (
    AuthenticatedUser,
    build_api_key_prefix,
    hash_api_key,
    is_forbidden_owner,
    owner_user_id_of,
)
from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.document_ingestion import (
    DefaultDocumentTextExtractor,
    DocumentContentError,
    DocumentDuplicateError,
    DocumentIngestionService,
    DocumentTextExtractor,
    DocumentTooLargeError,
    TextChunker,
    UnsupportedDocumentTypeError,
)
from chatbot_api.document_tasks import CeleryDocumentTaskQueue, DocumentTaskQueue
from chatbot_api.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderConfigurationError,
    EmbeddingProviderError,
    EmbeddingProviderTimeoutError,
    OpenAIEmbeddingProvider,
)
from chatbot_api.memory import MemoryManager
from chatbot_api.observability import (
    ObservabilityService,
    bind_request_context,
    normalize_route_path,
    reset_request_context,
)
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletion,
    ChatCompletionMetadata,
    ChatProvider,
    ChatProviderConfigurationError,
    ChatProviderError,
    ChatProviderTimeoutError,
    OpenAIChatProvider,
    TokenUsage,
    ToolRun,
    UsageCost,
    check_message_moderation,
)
from chatbot_api.repositories import (
    ChatRepository,
    ConversationSummaryRecord,
    MemoryRecord,
    MemoryRepository,
    OwnershipError,
    SqlAlchemyAuthRepository,
    SqlAlchemyChatRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyMemoryRepository,
    ToolRunRecord,
)
from chatbot_api.retrieval import DocumentRetriever
from chatbot_api.schemas import (
    ChatCitationPayload,
    ChatCostPayload,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMetadataPayload,
    ChatStreamCompletePayload,
    ChatStreamDeltaPayload,
    ChatStreamErrorPayload,
    ChatStreamStartPayload,
    ChatStreamToolCompletePayload,
    ChatStreamToolErrorPayload,
    ChatStreamToolStartPayload,
    ChatToolRunPayload,
    ChatUsagePayload,
    ConversationMemoryResponse,
    ConversationMemorySummaryPayload,
    ConversationToolRunsResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    ToolRunRecordPayload,
    UserMemoriesResponse,
    UserMemoryPayload,
)
from chatbot_api.services import (
    ChatService,
    ChatStreamChunk,
    ChatStreamComplete,
    ChatStreamStart,
    ChatStreamToolComplete,
    ChatStreamToolError,
    ChatStreamToolStart,
)
from chatbot_api.settings import Settings, get_settings
from chatbot_api.tools import ToolRegistry, build_tool_registry
from chatbot_api.tracing import TraceSink, build_trace_sink, is_langsmith_tracing_configured
from chatbot_api.workflow import ChatWorkflow
from chatbot_api.workflow_runtime import ChatWorkflowRuntime


class ObservabilityMiddleware:
    def __init__(self, app, *, settings: Settings) -> None:
        self.app = app
        self._settings = settings

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        app = scope.get("app")
        if app is None:
            await self.app(scope, receive, send)
            return

        observability = get_app_observability(app, self._settings)
        headers = Headers(raw=scope["headers"])
        request_id = headers.get("x-request-id") or uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started_at = perf_counter()
        finalized = False

        observability.log_event(
            "http.request.started",
            method=scope["method"],
            path=scope["path"],
            request_id=request_id,
        )

        async def finalize(status_code: int, *, error: Exception | None = None) -> None:
            nonlocal finalized
            if finalized:
                return

            finalized = True
            route = normalize_route_path(
                getattr(scope.get("route"), "path", None),
                scope["path"],
            )
            duration_seconds = perf_counter() - started_at
            observability.record_http_request(
                method=scope["method"],
                route=route,
                status_code=status_code,
                duration_seconds=duration_seconds,
            )
            if error is None:
                observability.log_event(
                    "http.request.completed",
                    method=scope["method"],
                    route=route,
                    status_code=status_code,
                    duration_ms=duration_seconds * 1000,
                    request_id=request_id,
                )
            else:
                observability.log_event(
                    "http.request.failed",
                    level="error",
                    method=scope["method"],
                    route=route,
                    status_code=status_code,
                    duration_ms=duration_seconds * 1000,
                    error=str(error),
                    error_type=type(error).__name__,
                    request_id=request_id,
                )

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                mutable_headers = MutableHeaders(scope=message)
                mutable_headers["X-Request-ID"] = request_id
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                await finalize(status_code=scope.get("_observability_status_code", 200))
            await send(message)

        async def send_status_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                scope["_observability_status_code"] = message["status"]
            await send_wrapper(message)

        try:
            await self.app(scope, receive, send_status_wrapper)
        except Exception as exc:
            await finalize(status_code=500, error=exc)
            raise


class RequestBodyTooLargeError(Exception):
    """Raised internally when a streamed request body exceeds the configured limit."""


async def send_request_body_too_large(send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": json.dumps({"detail": "request body too large"}).encode("utf-8"),
        }
    )


class RequestSizeLimitMiddleware:
    def __init__(self, app, *, settings: Settings) -> None:
        self.app = app
        self._max_bytes = settings.request_max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(raw=scope["headers"])
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = None
            if declared_bytes is not None and declared_bytes > self._max_bytes:
                await send_request_body_too_large(send)
                return

        received_bytes = 0

        async def receive_wrapper():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body") or b"")
                if received_bytes > self._max_bytes:
                    raise RequestBodyTooLargeError()
            return message

        try:
            await self.app(scope, receive_wrapper, send)
        except RequestBodyTooLargeError:
            await send_request_body_too_large(send)


async def send_rate_limit_exceeded(send, *, retry_after: int) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", str(retry_after).encode("ascii")),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": json.dumps({"detail": "rate limit exceeded"}).encode("utf-8"),
        }
    )


class RateLimitMiddleware:
    EXEMPT_PATHS = frozenset({"/health", "/metrics"})
    WINDOW_SECONDS = 60.0

    def __init__(self, app, *, settings: Settings) -> None:
        self.app = app
        self._limit = settings.rate_limit_requests_per_minute
        self._counters: dict[str, tuple[float, int]] = {}

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] in self.EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        key = self._resolve_key(scope)
        now = time()
        window_start, count = self._counters.get(key, (now, 0))
        if now - window_start >= self.WINDOW_SECONDS:
            window_start, count = now, 0

        count += 1
        self._counters[key] = (window_start, count)

        if count > self._limit:
            retry_after = max(1, int(self.WINDOW_SECONDS - (now - window_start)))
            await send_rate_limit_exceeded(send, retry_after=retry_after)
            return

        await self.app(scope, receive, send)

    def _resolve_key(self, scope) -> str:
        headers = Headers(raw=scope["headers"])
        api_key = headers.get("x-api-key")
        if api_key and api_key.strip():
            return f"key:{hash_api_key(api_key.strip())}"

        client = scope.get("client")
        if client:
            return f"ip:{client[0]}"

        return "ip:unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_database_engine(settings.database_url)
    app.state.db_engine = engine
    app.state.db_session_factory = create_session_factory(engine)
    app.state.chat_workflow_runtime = ChatWorkflowRuntime(settings)
    app.state.observability = ObservabilityService(settings)
    app.state.trace_sink = build_trace_sink(settings)

    try:
        yield
    finally:
        chat_provider = getattr(app.state, "chat_provider", None)
        if chat_provider is not None:
            await chat_provider.aclose()
        embedding_provider = getattr(app.state, "embedding_provider", None)
        if embedding_provider is not None:
            await embedding_provider.aclose()
        app.state.trace_sink.close()
        await app.state.chat_workflow_runtime.close()
        await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(ObservabilityMiddleware, settings=settings)
    app.add_middleware(RequestSizeLimitMiddleware, settings=settings)
    if settings.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware, settings=settings)
    if is_langsmith_tracing_configured(settings):
        app.add_middleware(TracingMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "chatbot-api",
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request) -> PlainTextResponse:
        observability = get_app_observability(request.app, settings)
        return PlainTextResponse(
            observability.render_metrics(),
            media_type="text/plain; version=0.0.4",
        )

    def serialize_stream_event(
        event: (
            ChatStreamStart
            | ChatStreamToolStart
            | ChatStreamToolComplete
            | ChatStreamToolError
            | ChatStreamChunk
            | ChatStreamComplete
        ),
    ) -> bytes:
        if isinstance(event, ChatStreamStart):
            return format_sse_event(
                event="message_start",
                data_str=ChatStreamStartPayload(
                    conversation_id=event.conversation_id
                ).model_dump_json(),
            )

        if isinstance(event, ChatStreamToolStart):
            return format_sse_event(
                event="tool_start",
                data_str=ChatStreamToolStartPayload(
                    conversation_id=event.conversation_id,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    input=event.input,
                ).model_dump_json(),
            )

        if isinstance(event, ChatStreamToolComplete):
            return format_sse_event(
                event="tool_complete",
                data_str=ChatStreamToolCompletePayload(
                    conversation_id=event.conversation_id,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    status="completed",
                    output=event.output,
                ).model_dump_json(),
            )

        if isinstance(event, ChatStreamToolError):
            return format_sse_event(
                event="tool_error",
                data_str=ChatStreamToolErrorPayload(
                    conversation_id=event.conversation_id,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    status=event.status,
                    error=event.error,
                ).model_dump_json(),
            )

        if isinstance(event, ChatStreamChunk):
            return format_sse_event(
                event="message_delta",
                data_str=ChatStreamDeltaPayload(delta=event.delta).model_dump_json(),
            )

        return format_sse_event(
            event="message_complete",
            data_str=ChatStreamCompletePayload(
                conversation_id=event.conversation_id,
                message=ChatMessage(role="assistant", content=event.completion.content),
                provider=event.completion.provider,
                model=event.completion.model,
                metadata=serialize_chat_metadata(event.completion.metadata),
            ).model_dump_json(exclude_none=True),
        )

    @app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
    async def chat(
        request: ChatRequest,
        raw_request: Request,
        service: ChatService = Depends(get_chat_service),
        moderation_provider: ChatProvider | None = Depends(get_optional_moderation_provider),
        settings: Settings = Depends(get_settings),
        observability: ObservabilityService = Depends(get_observability),
        trace_sink: TraceSink = Depends(get_trace_sink),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
    ) -> ChatResponse | EventSourceResponse:
        context_token = bind_request_context(request_id=raw_request.state.request_id)
        try:
            if settings.moderation_enabled:
                raw_client = getattr(moderation_provider, "raw_client", None)
                if raw_client is None:
                    raise HTTPException(
                        status_code=503, detail="moderation check unavailable"
                    )
                flagged = await check_message_moderation(
                    raw_client, request.message, model=settings.moderation_model
                )
                if flagged:
                    observability.record_moderation_check(outcome="blocked")
                    observability.log_event(
                        "moderation.blocked",
                        level="warning",
                        message_chars=len(request.message),
                    )
                    raise HTTPException(
                        status_code=400, detail="message violates content policy"
                    )
                observability.record_moderation_check(outcome="allowed")

            effective_metadata = merge_request_metadata_with_authenticated_user(
                request.metadata,
                authenticated_user,
            )
            metadata_fields = build_request_metadata_fields(observability, effective_metadata)
            observability.log_event(
                "chat.request.started",
                mode="stream" if request.stream else "sync",
                conversation_id=request.conversation_id,
                **metadata_fields,
            )
            started_at = perf_counter()
            request_trace_span = trace_sink.start_request_span(
                "chat.request",
                inputs=build_chat_request_trace_inputs(
                    request,
                    observability,
                    metadata=effective_metadata,
                ),
                metadata=build_chat_request_trace_metadata(
                    request=request,
                    request_id=raw_request.state.request_id,
                    observability=observability,
                    metadata=effective_metadata,
                ),
                tags=["chat", "stream" if request.stream else "sync"],
            )
            if request.stream:
                request_trace_span.__enter__()
                stream = service.stream_chat(
                    conversation_id=request.conversation_id,
                    owner_user_id=owner_user_id_of(authenticated_user),
                    message=request.message,
                    metadata=effective_metadata,
                )

                try:
                    first_event = await anext(stream)
                except StopAsyncIteration as exc:
                    record_failed_chat_request(
                        observability,
                        mode="stream",
                        started_at=started_at,
                        conversation_id=request.conversation_id,
                        error_detail="LLM provider stream ended before completion",
                    )
                    request_trace_span.finish_error(
                        "LLM provider stream ended before completion",
                        metadata={
                            "chat_mode": "stream",
                            "request_id": raw_request.state.request_id,
                        },
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="LLM provider stream ended before completion",
                    ) from exc
                except (
                    ChatProviderTimeoutError,
                    ChatProviderError,
                    EmbeddingProviderTimeoutError,
                    EmbeddingProviderError,
                    OwnershipError,
                ) as exc:
                    record_failed_chat_request(
                        observability,
                        mode="stream",
                        started_at=started_at,
                        conversation_id=request.conversation_id,
                        error_detail=str(exc),
                    )
                    request_trace_span.finish_error(
                        exc,
                        metadata={
                            "chat_mode": "stream",
                            "request_id": raw_request.state.request_id,
                        },
                    )
                    status_code, detail = resolve_chat_error_response(exc)
                    raise HTTPException(status_code=status_code, detail=detail) from exc

                request_trace_span.annotate(
                    metadata={
                        "conversation_id": getattr(
                            first_event,
                            "conversation_id",
                            request.conversation_id,
                        ),
                    }
                )
                request_trace_span.suspend()

                async def event_generator() -> AsyncIterator[bytes]:
                    stream_context_token = bind_request_context(
                        request_id=raw_request.state.request_id
                    )
                    trace_token = request_trace_span.activate()
                    final_outcome: str | None = None
                    final_error_detail: str | None = None
                    stream_conversation_id = getattr(
                        first_event,
                        "conversation_id",
                        request.conversation_id,
                    )
                    completion: ChatCompletion | None = None

                    def finalize_stream(
                        *,
                        outcome: str,
                        error_detail: str | None = None,
                    ) -> None:
                        nonlocal final_error_detail, final_outcome

                        if final_outcome is not None:
                            return

                        final_outcome = outcome
                        final_error_detail = error_detail
                        duration_seconds = perf_counter() - started_at
                        observability.record_chat_request(
                            mode="stream",
                            outcome=outcome,
                            duration_seconds=duration_seconds,
                        )
                        observability.log_event(
                            "chat.request.completed"
                            if outcome == "success"
                            else "chat.request.failed",
                            level="info" if outcome == "success" else "warning",
                            mode="stream",
                            conversation_id=stream_conversation_id,
                            provider=None if completion is None else completion.provider,
                            model=None if completion is None else completion.model,
                            duration_ms=duration_seconds * 1000,
                            outcome=outcome,
                            error=error_detail,
                            **build_completion_observability_fields(
                                None if completion is None else completion.metadata
                            ),
                        )

                    try:
                        yield serialize_stream_event(first_event)
                        if isinstance(first_event, ChatStreamComplete):
                            completion = first_event.completion

                        async for event in stream:
                            if await raw_request.is_disconnected():
                                observability.record_chat_stream_disconnect()
                                finalize_stream(outcome="disconnected")
                                break
                            stream_conversation_id = getattr(
                                event,
                                "conversation_id",
                                stream_conversation_id,
                            )
                            if isinstance(event, ChatStreamComplete):
                                completion = event.completion
                            yield serialize_stream_event(event)
                    except (
                        ChatProviderTimeoutError,
                        ChatProviderError,
                        EmbeddingProviderTimeoutError,
                        EmbeddingProviderError,
                    ) as exc:
                        finalize_stream(outcome="error", error_detail=str(exc))
                        yield format_sse_event(
                            event="error",
                            data_str=ChatStreamErrorPayload(detail=str(exc)).model_dump_json(),
                        )
                    finally:
                        if completion is not None:
                            finalize_stream(outcome="success")
                        request_trace_span.deactivate(trace_token)
                        finish_chat_request_trace(
                            request_trace_span,
                            request_id=raw_request.state.request_id,
                            mode="stream",
                            outcome=final_outcome or "interrupted",
                            conversation_id=stream_conversation_id,
                            completion=completion,
                            error_detail=final_error_detail,
                        )
                        reset_request_context(stream_context_token)
                        await stream.aclose()

                return EventSourceResponse(event_generator())

            with request_trace_span:
                try:
                    conversation_id, completion = await service.chat(
                        conversation_id=request.conversation_id,
                        owner_user_id=owner_user_id_of(authenticated_user),
                        message=request.message,
                        metadata=effective_metadata,
                    )
                except (
                    ChatProviderTimeoutError,
                    ChatProviderError,
                    EmbeddingProviderTimeoutError,
                    EmbeddingProviderError,
                    OwnershipError,
                ) as exc:
                    record_failed_chat_request(
                        observability,
                        mode="sync",
                        started_at=started_at,
                        conversation_id=request.conversation_id,
                        error_detail=str(exc),
                    )
                    request_trace_span.annotate(
                        metadata={
                            "conversation_id": request.conversation_id,
                            "chat_mode": "sync",
                            "request_id": raw_request.state.request_id,
                            "outcome": "error",
                        }
                    )
                    status_code, detail = resolve_chat_error_response(exc)
                    raise HTTPException(status_code=status_code, detail=detail) from exc

                duration_seconds = perf_counter() - started_at
                observability.record_chat_request(
                    mode="sync",
                    outcome="success",
                    duration_seconds=duration_seconds,
                )
                observability.log_event(
                    "chat.request.completed",
                    mode="sync",
                    conversation_id=conversation_id,
                    provider=completion.provider,
                    model=completion.model,
                    duration_ms=duration_seconds * 1000,
                    outcome="success",
                    **build_completion_observability_fields(completion.metadata),
                )
                finish_chat_request_trace(
                    request_trace_span,
                    request_id=raw_request.state.request_id,
                    mode="sync",
                    outcome="success",
                    conversation_id=conversation_id,
                    completion=completion,
                )

                return ChatResponse(
                    conversation_id=conversation_id,
                    message=ChatMessage(role="assistant", content=completion.content),
                    provider=completion.provider,
                    model=completion.model,
                    metadata=serialize_chat_metadata(completion.metadata),
                )
        finally:
            reset_request_context(context_token)

    @app.post(
        "/documents",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        file: UploadFile = File(...),
        service: DocumentIngestionService = Depends(get_document_service),
        repository: SqlAlchemyDocumentRepository = Depends(get_document_repository),
        task_queue: DocumentTaskQueue = Depends(get_document_task_queue),
        observability: ObservabilityService = Depends(get_observability),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
    ) -> DocumentUploadResponse:
        started_at = perf_counter()
        owner_user_id = owner_user_id_of(authenticated_user)
        try:
            document, chunk_count = await service.ingest_document(
                filename=file.filename or "upload",
                content_type=file.content_type,
                content=await file.read(),
                owner_user_id=owner_user_id,
            )
            try:
                task_queue.enqueue_embed_document(document.id)
            except Exception as exc:
                await repository.mark_document_failed(
                    document_id=document.id,
                    failure_reason="enqueue_failed",
                )
                observability.record_document_upload(outcome="enqueue_failed")
                observability.log_event(
                    "document.upload.failed",
                    level="error",
                    filename=document.filename,
                    document_id=document.id,
                    owner_user_id=owner_user_id,
                    duration_ms=(perf_counter() - started_at) * 1000,
                    outcome="enqueue_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise HTTPException(
                    status_code=503,
                    detail="failed to enqueue document embedding task",
                ) from exc
        except DocumentDuplicateError as exc:
            observability.record_document_upload(outcome="duplicate")
            observability.log_event(
                "document.upload.rejected",
                level="warning",
                filename=file.filename or "upload",
                owner_user_id=owner_user_id,
                duration_ms=(perf_counter() - started_at) * 1000,
                outcome="duplicate",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedDocumentTypeError as exc:
            observability.record_document_upload(outcome="validation_error")
            observability.log_event(
                "document.upload.rejected",
                level="warning",
                filename=file.filename or "upload",
                owner_user_id=owner_user_id,
                duration_ms=(perf_counter() - started_at) * 1000,
                outcome="validation_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DocumentContentError as exc:
            observability.record_document_upload(outcome="validation_error")
            observability.log_event(
                "document.upload.rejected",
                level="warning",
                filename=file.filename or "upload",
                owner_user_id=owner_user_id,
                duration_ms=(perf_counter() - started_at) * 1000,
                outcome="validation_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DocumentTooLargeError as exc:
            observability.record_document_upload(outcome="validation_error")
            observability.log_event(
                "document.upload.rejected",
                level="warning",
                filename=file.filename or "upload",
                owner_user_id=owner_user_id,
                duration_ms=(perf_counter() - started_at) * 1000,
                outcome="validation_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except Exception as exc:
            observability.record_document_upload(outcome="error")
            observability.log_event(
                "document.upload.failed",
                level="error",
                filename=file.filename or "upload",
                owner_user_id=owner_user_id,
                duration_ms=(perf_counter() - started_at) * 1000,
                outcome="error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        finally:
            await file.close()

        observability.record_document_upload(outcome="success")
        observability.log_event(
            "document.upload.completed",
            filename=document.filename,
            document_id=document.id,
            owner_user_id=owner_user_id,
            duration_ms=(perf_counter() - started_at) * 1000,
            chunk_count=chunk_count,
            status=document.status,
            outcome="success",
        )
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            byte_size=document.byte_size,
            status=document.status,
            chunk_count=chunk_count,
            created_at=document.created_at,
        )

    @app.get(
        "/conversations/{conversation_id}/tool-runs",
        response_model=ConversationToolRunsResponse,
        response_model_exclude_none=True,
    )
    async def list_conversation_tool_runs(
        conversation_id: str,
        limit: int = Query(default=50, ge=1, le=100),
        repository: ChatRepository = Depends(get_chat_repository),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
    ) -> ConversationToolRunsResponse:
        owner_user_id = owner_user_id_of(authenticated_user)
        if not await repository.conversation_exists(conversation_id, owner_user_id=owner_user_id):
            raise HTTPException(status_code=404, detail="conversation not found")

        tool_runs = await repository.list_tool_runs(
            conversation_id,
            limit=limit,
            owner_user_id=owner_user_id,
        )
        return ConversationToolRunsResponse(
            conversation_id=conversation_id,
            tool_runs=[
                serialize_tool_run_record(tool_run)
                for tool_run in tool_runs
            ],
        )

    @app.get(
        "/conversations/{conversation_id}/memory",
        response_model=ConversationMemoryResponse,
        response_model_exclude_none=True,
    )
    async def get_conversation_memory(
        conversation_id: str,
        chat_repository: ChatRepository = Depends(get_chat_repository),
        memory_repository: MemoryRepository = Depends(get_memory_repository),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
    ) -> ConversationMemoryResponse:
        owner_user_id = owner_user_id_of(authenticated_user)
        if not await chat_repository.conversation_exists(
            conversation_id,
            owner_user_id=owner_user_id,
        ):
            raise HTTPException(status_code=404, detail="conversation not found")

        summary = await memory_repository.get_conversation_summary(
            conversation_id,
            owner_user_id=owner_user_id,
        )
        return ConversationMemoryResponse(
            conversation_id=conversation_id,
            summary=None if summary is None else serialize_conversation_summary(summary),
        )

    @app.get(
        "/users/{user_id}/memories",
        response_model=UserMemoriesResponse,
        response_model_exclude_none=True,
    )
    async def list_user_memories(
        user_id: str,
        memory_repository: MemoryRepository = Depends(get_memory_repository),
        settings: Settings = Depends(get_settings),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
        observability: ObservabilityService = Depends(get_observability),
    ) -> UserMemoriesResponse:
        if is_forbidden_owner(authenticated_user, user_id):
            observability.log_event(
                "memory.access.forbidden",
                level="warning",
                requested_user_id=user_id,
                authenticated_user_id=owner_user_id_of(authenticated_user),
            )
            raise HTTPException(status_code=403, detail="forbidden")
        memories = await memory_repository.list_active_memories(
            user_id,
            limit=settings.memory_max_active_items,
            owner_user_id=owner_user_id_of(authenticated_user),
        )
        return UserMemoriesResponse(
            user_id=user_id,
            memories=[serialize_memory_record(memory) for memory in memories],
        )

    @app.delete(
        "/users/{user_id}/memories/{memory_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_user_memory(
        user_id: str,
        memory_id: int,
        memory_repository: MemoryRepository = Depends(get_memory_repository),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
        observability: ObservabilityService = Depends(get_observability),
    ) -> None:
        if is_forbidden_owner(authenticated_user, user_id):
            observability.log_event(
                "memory.access.forbidden",
                level="warning",
                requested_user_id=user_id,
                authenticated_user_id=owner_user_id_of(authenticated_user),
            )
            raise HTTPException(status_code=403, detail="forbidden")
        deleted = await memory_repository.delete_memory(
            user_id=user_id,
            memory_id=memory_id,
            owner_user_id=owner_user_id_of(authenticated_user),
        )
        observability.log_event(
            "memory.delete.completed" if deleted else "memory.delete.rejected",
            level="info" if deleted else "warning",
            user_id=user_id,
            memory_id=memory_id,
            authenticated_user_id=owner_user_id_of(authenticated_user),
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="memory not found")

    @app.get(
        "/documents/{document_id}",
        response_model=DocumentStatusResponse,
        response_model_exclude_none=True,
    )
    async def get_document(
        document_id: str,
        repository: SqlAlchemyDocumentRepository = Depends(get_document_repository),
        authenticated_user: AuthenticatedUser | None = Depends(get_authenticated_user),
    ) -> DocumentStatusResponse:
        owner_user_id = owner_user_id_of(authenticated_user)
        document = await repository.get_document(document_id, owner_user_id=owner_user_id)
        if document is None:
            raise HTTPException(status_code=404, detail="document not found")

        chunk_count = await repository.count_document_chunks(
            document_id,
            owner_user_id=owner_user_id,
        )
        return DocumentStatusResponse(
            document_id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            byte_size=document.byte_size,
            status=document.status,
            chunk_count=chunk_count,
            created_at=document.created_at,
            updated_at=document.updated_at,
            failure_reason=document.failure_reason,
        )

    return app


async def get_trace_sink(
    request: Request,
) -> TraceSink:
    return get_app_trace_sink(request.app)


async def get_chat_provider(
    request: Request,
    settings: Settings = Depends(get_settings),
    trace_sink: TraceSink = Depends(get_trace_sink),
) -> ChatProvider:
    chat_provider = getattr(request.app.state, "chat_provider", None)
    if chat_provider is None:
        try:
            chat_provider = OpenAIChatProvider(settings, trace_sink=trace_sink)
        except ChatProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        request.app.state.chat_provider = chat_provider

    return chat_provider


async def get_optional_moderation_provider(
    request: Request,
    settings: Settings = Depends(get_settings),
    trace_sink: TraceSink = Depends(get_trace_sink),
) -> ChatProvider | None:
    if not settings.moderation_enabled:
        return None
    return await get_chat_provider(request, settings=settings, trace_sink=trace_sink)


async def get_embedding_provider(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EmbeddingProvider:
    embedding_provider = getattr(request.app.state, "embedding_provider", None)
    if embedding_provider is None:
        try:
            embedding_provider = OpenAIEmbeddingProvider(settings)
        except EmbeddingProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        request.app.state.embedding_provider = embedding_provider

    return embedding_provider


async def get_session_factory(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> async_sessionmaker[AsyncSession]:
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        engine = create_database_engine(settings.database_url)
        request.app.state.db_engine = engine
        session_factory = create_session_factory(engine)
        request.app.state.db_session_factory = session_factory

    return session_factory


async def get_db_session(
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


async def get_auth_repository(
    session: AsyncSession = Depends(get_db_session),
) -> SqlAlchemyAuthRepository:
    return SqlAlchemyAuthRepository(session)


async def get_observability(
    request: Request,
) -> ObservabilityService:
    return get_app_observability(request.app)


async def get_authenticated_user(
    request: Request,
    settings: Settings = Depends(get_settings),
    repository: SqlAlchemyAuthRepository = Depends(get_auth_repository),
    observability: ObservabilityService = Depends(get_observability),
) -> AuthenticatedUser | None:
    if not settings.auth_enabled:
        return None

    api_key = request.headers.get("X-API-Key")
    if api_key is None or not api_key.strip():
        observability.record_auth_attempt(outcome="missing_key")
        observability.log_event(
            "auth.failed",
            level="warning",
            reason="missing_api_key",
        )
        raise HTTPException(status_code=401, detail="missing API key")

    authenticated_user = await repository.authenticate_api_key(api_key.strip())
    if authenticated_user is None:
        observability.record_auth_attempt(outcome="invalid_key")
        observability.log_event(
            "auth.failed",
            level="warning",
            reason="invalid_api_key",
            api_key_prefix=build_api_key_prefix(api_key.strip()),
        )
        raise HTTPException(status_code=401, detail="invalid API key")

    return authenticated_user


async def get_chat_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ChatRepository:
    return SqlAlchemyChatRepository(session)


async def get_memory_repository(
    session: AsyncSession = Depends(get_db_session),
) -> MemoryRepository:
    return SqlAlchemyMemoryRepository(session)


async def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> SqlAlchemyDocumentRepository:
    return SqlAlchemyDocumentRepository(session)


async def get_document_text_extractor() -> DocumentTextExtractor:
    return DefaultDocumentTextExtractor()


async def get_text_chunker(
    settings: Settings = Depends(get_settings),
) -> TextChunker:
    return TextChunker(
        chunk_size=settings.document_chunk_size_chars,
        chunk_overlap=settings.document_chunk_overlap_chars,
    )


async def get_document_service(
    repository: SqlAlchemyDocumentRepository = Depends(get_document_repository),
    extractor: DocumentTextExtractor = Depends(get_document_text_extractor),
    chunker: TextChunker = Depends(get_text_chunker),
    settings: Settings = Depends(get_settings),
) -> DocumentIngestionService:
    return DocumentIngestionService(
        repository,
        extractor,
        chunker,
        max_bytes=settings.document_max_bytes,
    )


async def get_document_task_queue() -> DocumentTaskQueue:
    return CeleryDocumentTaskQueue()


async def get_optional_rerank_provider(
    request: Request,
    settings: Settings = Depends(get_settings),
    trace_sink: TraceSink = Depends(get_trace_sink),
) -> ChatProvider | None:
    if not settings.retrieval_rerank_enabled:
        return None
    return await get_chat_provider(request, settings=settings, trace_sink=trace_sink)


async def get_document_retriever(
    repository: SqlAlchemyDocumentRepository = Depends(get_document_repository),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    settings: Settings = Depends(get_settings),
    observability: ObservabilityService = Depends(get_observability),
    trace_sink: TraceSink = Depends(get_trace_sink),
    rerank_provider: ChatProvider | None = Depends(get_optional_rerank_provider),
) -> DocumentRetriever:
    return DocumentRetriever(
        repository,
        embedding_provider,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        max_chunks_per_document=settings.retrieval_max_chunks_per_document,
        candidate_limit=settings.retrieval_candidate_limit,
        observability=observability,
        trace_sink=trace_sink,
        chat_provider=rerank_provider,
        rerank_enabled=settings.retrieval_rerank_enabled,
    )


async def get_chat_workflow(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ChatWorkflow:
    runtime = getattr(request.app.state, "chat_workflow_runtime", None)
    if runtime is None:
        runtime = ChatWorkflowRuntime(settings)
        request.app.state.chat_workflow_runtime = runtime

    return await runtime.get_workflow()


async def get_tool_registry(
    retriever: DocumentRetriever = Depends(get_document_retriever),
    settings: Settings = Depends(get_settings),
    observability: ObservabilityService = Depends(get_observability),
    trace_sink: TraceSink = Depends(get_trace_sink),
) -> ToolRegistry:
    return build_tool_registry(
        retriever=retriever,
        search_top_k=settings.tool_search_top_k,
        timeout_seconds=settings.tool_execution_timeout_seconds,
        observability=observability,
        trace_sink=trace_sink,
    )


async def get_memory_manager(
    provider: ChatProvider = Depends(get_chat_provider),
    chat_repository: ChatRepository = Depends(get_chat_repository),
    memory_repository: MemoryRepository = Depends(get_memory_repository),
    settings: Settings = Depends(get_settings),
    trace_sink: TraceSink = Depends(get_trace_sink),
) -> MemoryManager | None:
    if not settings.memory_enabled:
        return None

    return MemoryManager(
        provider=provider,
        chat_repository=chat_repository,
        memory_repository=memory_repository,
        settings=settings,
        trace_sink=trace_sink,
    )


async def get_chat_service(
    provider: ChatProvider = Depends(get_chat_provider),
    repository: ChatRepository = Depends(get_chat_repository),
    workflow: ChatWorkflow = Depends(get_chat_workflow),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    memory_manager: MemoryManager | None = Depends(get_memory_manager),
    settings: Settings = Depends(get_settings),
    observability: ObservabilityService = Depends(get_observability),
    trace_sink: TraceSink = Depends(get_trace_sink),
) -> ChatService:
    return ChatService(
        provider,
        repository,
        workflow,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        tool_max_rounds=settings.tool_max_rounds,
        observability=observability,
        pricing_model=settings.openai_model,
        input_price_per_1m_tokens=settings.openai_model_input_price_per_1m_tokens,
        output_price_per_1m_tokens=settings.openai_model_output_price_per_1m_tokens,
        trace_sink=trace_sink,
    )


app = create_app()


def get_app_observability(
    app: FastAPI,
    settings: Settings | None = None,
) -> ObservabilityService:
    observability = getattr(app.state, "observability", None)
    if observability is None:
        observability = ObservabilityService(settings)
        app.state.observability = observability
    return observability


def get_app_trace_sink(
    app: FastAPI,
    settings: Settings | None = None,
) -> TraceSink:
    trace_sink = getattr(app.state, "trace_sink", None)
    if trace_sink is None:
        trace_sink = build_trace_sink(settings)
        app.state.trace_sink = trace_sink
    return trace_sink


def build_request_metadata_fields(
    observability: ObservabilityService,
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    if metadata is None:
        return {"has_metadata": False}
    if observability.include_request_metadata:
        return {
            "has_metadata": True,
            "request_metadata": metadata,
        }
    return {
        "has_metadata": True,
        "metadata_key_count": len(metadata),
    }


def build_chat_request_trace_inputs(
    request: ChatRequest,
    observability: ObservabilityService,
    *,
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "message": request.message,
        "stream": request.stream,
    }
    if request.conversation_id is not None:
        payload["requested_conversation_id"] = request.conversation_id
    if metadata is not None:
        if observability.include_request_metadata:
            payload["request_metadata"] = metadata
        else:
            payload["metadata_key_count"] = len(metadata)
    return payload


def build_chat_request_trace_metadata(
    *,
    request: ChatRequest,
    request_id: str,
    observability: ObservabilityService,
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": request_id,
        "chat_mode": "stream" if request.stream else "sync",
        "has_metadata": metadata is not None,
    }
    if request.conversation_id is not None:
        payload["requested_conversation_id"] = request.conversation_id
    if metadata is not None and not observability.include_request_metadata:
        payload["metadata_key_count"] = len(metadata)
    return payload


def merge_request_metadata_with_authenticated_user(
    metadata: dict[str, object] | None,
    authenticated_user: AuthenticatedUser | None,
) -> dict[str, object] | None:
    if authenticated_user is None:
        return metadata

    merged = {} if metadata is None else dict(metadata)
    merged["user_profile"] = authenticated_user.to_user_profile_metadata()
    return merged


CHAT_ERROR_STATUS_CODES: dict[type[Exception], int] = {
    ChatProviderTimeoutError: 504,
    ChatProviderError: 502,
    EmbeddingProviderTimeoutError: 504,
    EmbeddingProviderError: 502,
    OwnershipError: 404,
}


def resolve_chat_error_response(exc: Exception) -> tuple[int, str]:
    status_code = CHAT_ERROR_STATUS_CODES[type(exc)]
    detail = "conversation not found" if isinstance(exc, OwnershipError) else str(exc)
    return status_code, detail


def record_failed_chat_request(
    observability: ObservabilityService,
    *,
    mode: str,
    started_at: float,
    conversation_id: str | None,
    error_detail: str,
) -> None:
    duration_seconds = perf_counter() - started_at
    observability.record_chat_request(
        mode=mode,
        outcome="error",
        duration_seconds=duration_seconds,
    )
    observability.log_event(
        "chat.request.failed",
        level="warning",
        mode=mode,
        conversation_id=conversation_id,
        duration_ms=duration_seconds * 1000,
        outcome="error",
        error=error_detail,
    )


def finish_chat_request_trace(
    trace_span,
    *,
    request_id: str,
    mode: str,
    outcome: str,
    conversation_id: str | None,
    completion: ChatCompletion | None,
    error_detail: str | None = None,
) -> None:
    metadata: dict[str, object] = {
        "request_id": request_id,
        "chat_mode": mode,
        "outcome": outcome,
    }
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id

    if completion is None:
        if error_detail is not None:
            metadata["error"] = error_detail
            trace_span.finish_error(error_detail, metadata=metadata)
            return
        trace_span.finish_success(outputs=metadata)
        return

    trace_span.finish_success(
        outputs={
            "conversation_id": conversation_id,
            "provider": completion.provider,
            "model": completion.model,
            "assistant_message": completion.content,
            "citation_count": len(completion.metadata.citations)
            if completion.metadata is not None
            else 0,
            "tool_run_count": len(completion.metadata.tool_runs)
            if completion.metadata is not None
            else 0,
            "usage": None if completion.metadata is None or completion.metadata.usage is None else {
                "input_tokens": completion.metadata.usage.input_tokens,
                "output_tokens": completion.metadata.usage.output_tokens,
                "total_tokens": completion.metadata.usage.total_tokens,
            },
            "cost": None if completion.metadata is None or completion.metadata.cost is None else {
                "input_cost_usd": completion.metadata.cost.input_cost_usd,
                "output_cost_usd": completion.metadata.cost.output_cost_usd,
                "total_cost_usd": completion.metadata.cost.total_cost_usd,
                "currency": completion.metadata.cost.currency,
            },
        },
        metadata=metadata,
        tags=[f"provider:{completion.provider}"],
    )


def serialize_chat_metadata(
    metadata: ChatCompletionMetadata | None,
) -> ChatResponseMetadataPayload | None:
    if metadata is None:
        return None

    return ChatResponseMetadataPayload(
        citations=[
            serialize_chat_citation(citation)
            for citation in metadata.citations
        ],
        tool_runs=[
            serialize_tool_run(tool_run)
            for tool_run in metadata.tool_runs
        ],
        usage=None if metadata.usage is None else serialize_usage(metadata.usage),
        cost=None if metadata.cost is None else serialize_cost(metadata.cost),
    )


def serialize_chat_citation(citation: ChatCitation) -> ChatCitationPayload:
    return ChatCitationPayload(
        document_id=citation.document_id,
        filename=citation.filename,
        chunk_index=citation.chunk_index,
        start_offset=citation.start_offset,
        end_offset=citation.end_offset,
        snippet=citation.snippet,
    )


def serialize_tool_run(tool_run: ToolRun) -> ChatToolRunPayload:
    return ChatToolRunPayload(
        tool_call_id=tool_run.tool_call_id,
        tool_name=tool_run.tool_name,
        status=tool_run.status,
        input=tool_run.input,
        output=tool_run.output,
        error=tool_run.error,
    )


def serialize_tool_run_record(tool_run: ToolRunRecord) -> ToolRunRecordPayload:
    return ToolRunRecordPayload(
        id=tool_run.id,
        tool_call_id=tool_run.tool_call_id,
        tool_name=tool_run.tool_name,
        status=tool_run.status,
        input=tool_run.input_payload,
        output=tool_run.output_payload,
        error=tool_run.error_message,
        started_at=tool_run.started_at,
        completed_at=tool_run.completed_at,
    )


def serialize_conversation_summary(
    summary: ConversationSummaryRecord,
) -> ConversationMemorySummaryPayload:
    return ConversationMemorySummaryPayload(
        conversation_id=summary.conversation_id,
        summary_text=summary.summary_text,
        last_summarized_message_id=summary.last_summarized_message_id,
        updated_at=summary.updated_at,
    )


def serialize_memory_record(memory: MemoryRecord) -> UserMemoryPayload:
    return UserMemoryPayload(
        id=memory.id,
        user_id=memory.user_id,
        kind=memory.kind,
        key=memory.key,
        value_json=memory.value_json,
        confidence=memory.confidence,
        source_message_id=memory.source_message_id,
        extraction_method=memory.extraction_method,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def serialize_usage(usage: TokenUsage) -> ChatUsagePayload:
    return ChatUsagePayload(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def serialize_cost(cost: UsageCost) -> ChatCostPayload:
    return ChatCostPayload(
        input_cost_usd=cost.input_cost_usd,
        output_cost_usd=cost.output_cost_usd,
        total_cost_usd=cost.total_cost_usd,
        currency=cost.currency,
    )


def build_completion_observability_fields(
    metadata: ChatCompletionMetadata | None,
) -> dict[str, float | int]:
    if metadata is None:
        return {}

    fields: dict[str, float | int] = {}
    if metadata.usage is not None:
        fields["input_tokens"] = metadata.usage.input_tokens
        fields["output_tokens"] = metadata.usage.output_tokens
        fields["total_tokens"] = metadata.usage.total_tokens
    if metadata.cost is not None:
        fields["input_cost_usd"] = metadata.cost.input_cost_usd
        fields["output_cost_usd"] = metadata.cost.output_cost_usd
        fields["total_cost_usd"] = metadata.cost.total_cost_usd
    return fields
