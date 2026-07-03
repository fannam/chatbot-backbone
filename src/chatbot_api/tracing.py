from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langsmith import Client, trace
from langsmith.run_helpers import get_current_run_tree
from langsmith.wrappers import wrap_openai

from chatbot_api.settings import Settings, get_settings

TraceRunType = Literal["chain", "tool", "retriever", "llm"]

_CURRENT_TRACE_SPAN: ContextVar[TraceSpanHandle | None] = ContextVar(
    "current_trace_span",
    default=None,
)


class TraceSpanHandle(Protocol):
    name: str
    run_type: TraceRunType
    finished: bool

    def start_child_span(
        self,
        name: str,
        *,
        run_type: TraceRunType = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> TraceSpanHandle: ...

    def annotate(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None: ...

    def finish_success(
        self,
        *,
        outputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None: ...

    def finish_error(
        self,
        error: BaseException | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None: ...

    def activate(self) -> Token[TraceSpanHandle | None]: ...

    def deactivate(self, token: Token[TraceSpanHandle | None]) -> None: ...

    def suspend(self) -> None: ...

    def __enter__(self) -> TraceSpanHandle: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool | None: ...


class TraceSink(Protocol):
    enabled: bool

    def start_request_span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> TraceSpanHandle: ...

    def start_span(
        self,
        name: str,
        *,
        run_type: TraceRunType = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        parent: TraceSpanHandle | None = None,
    ) -> TraceSpanHandle: ...

    def wrap_openai_client(self, client: Any) -> Any: ...

    def close(self) -> None: ...


def get_current_trace_span() -> TraceSpanHandle | None:
    return _CURRENT_TRACE_SPAN.get()


def is_langsmith_tracing_configured(settings: Settings) -> bool:
    return settings.langsmith_tracing_enabled and bool(settings.langsmith_api_key)


def build_trace_sink(settings: Settings | None = None) -> TraceSink:
    resolved_settings = settings or get_settings()
    if not is_langsmith_tracing_configured(resolved_settings):
        return NoopTraceSink()
    return LangSmithTraceSink(resolved_settings)


@dataclass
class NoopTraceSpan:
    sink: NoopTraceSink
    name: str
    run_type: TraceRunType
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    finished: bool = False
    _activation_token: Token[TraceSpanHandle | None] | None = None

    def start_child_span(
        self,
        name: str,
        *,
        run_type: TraceRunType = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> TraceSpanHandle:
        return self.sink.start_span(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            tags=tags,
            parent=self,
        )

    def annotate(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        if metadata:
            self.metadata.update(sanitize_trace_payload(dict(metadata)))
        if tags:
            self.tags.extend(str(tag) for tag in tags)

    def finish_success(
        self,
        *,
        outputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        if self.finished:
            return
        self.annotate(metadata=metadata, tags=tags)
        if outputs:
            self.outputs = sanitize_trace_payload(dict(outputs))
        self.finished = True
        self._reset_activation()

    def finish_error(
        self,
        error: BaseException | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        if self.finished:
            return
        self.annotate(metadata=metadata, tags=tags)
        self.metadata["error"] = str(error)
        self.finished = True
        self._reset_activation()

    def activate(self) -> Token[TraceSpanHandle | None]:
        return _CURRENT_TRACE_SPAN.set(self)

    def deactivate(self, token: Token[TraceSpanHandle | None]) -> None:
        _CURRENT_TRACE_SPAN.reset(token)

    def suspend(self) -> None:
        self._reset_activation()

    def __enter__(self) -> TraceSpanHandle:
        if self._activation_token is None:
            self._activation_token = self.activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        if exc_value is None:
            self.finish_success()
        else:
            self.finish_error(exc_value)
        return None

    def _reset_activation(self) -> None:
        if self._activation_token is not None:
            try:
                self.deactivate(self._activation_token)
            except ValueError:
                pass
            self._activation_token = None


@dataclass
class NoopTraceSink:
    enabled: bool = False

    def start_request_span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> TraceSpanHandle:
        return self.start_span(
            name,
            run_type="chain",
            inputs=inputs,
            metadata=metadata,
            tags=tags,
        )

    def start_span(
        self,
        name: str,
        *,
        run_type: TraceRunType = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        parent: TraceSpanHandle | None = None,
    ) -> TraceSpanHandle:
        del parent
        return NoopTraceSpan(
            sink=self,
            name=name,
            run_type=run_type,
            inputs={} if inputs is None else sanitize_trace_payload(dict(inputs)),
            metadata={} if metadata is None else sanitize_trace_payload(dict(metadata)),
            tags=[] if tags is None else [str(tag) for tag in tags],
        )

    def wrap_openai_client(self, client: Any) -> Any:
        return client

    def close(self) -> None:
        return None


@dataclass
class LangSmithTraceSink:
    settings: Settings

    def __post_init__(self) -> None:
        self.enabled = True
        self.project_name = self.settings.langsmith_project
        self.client = Client(
            api_key=self.settings.langsmith_api_key,
            api_url=self.settings.langsmith_endpoint,
        )

    def start_request_span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> TraceSpanHandle:
        return self.start_span(
            name,
            run_type="chain",
            inputs=inputs,
            metadata=metadata,
            tags=tags,
        )

    def start_span(
        self,
        name: str,
        *,
        run_type: TraceRunType = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        parent: TraceSpanHandle | None = None,
    ) -> TraceSpanHandle:
        return LangSmithTraceSpan(
            sink=self,
            name=name,
            run_type=run_type,
            initial_inputs={} if inputs is None else sanitize_trace_payload(dict(inputs)),
            initial_metadata={} if metadata is None else sanitize_trace_payload(dict(metadata)),
            initial_tags=[] if tags is None else [str(tag) for tag in tags],
            explicit_parent=parent,
        )

    def wrap_openai_client(self, client: Any) -> Any:
        return wrap_openai(client)

    def close(self) -> None:
        self.client.flush()
        self.client.close()


@dataclass
class LangSmithTraceSpan:
    sink: LangSmithTraceSink
    name: str
    run_type: TraceRunType
    initial_inputs: dict[str, Any]
    initial_metadata: dict[str, Any]
    initial_tags: list[str]
    explicit_parent: TraceSpanHandle | None = None
    finished: bool = False
    _context_manager: Any | None = None
    _run_tree: Any | None = None
    _activation_token: Token[TraceSpanHandle | None] | None = None

    def start_child_span(
        self,
        name: str,
        *,
        run_type: TraceRunType = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> TraceSpanHandle:
        return self.sink.start_span(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            tags=tags,
            parent=self,
        )

    def annotate(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        if metadata:
            sanitized_metadata = sanitize_trace_payload(dict(metadata))
            if self._run_tree is None:
                self.initial_metadata.update(sanitized_metadata)
            else:
                self._run_tree.add_metadata(sanitized_metadata)
        if tags:
            sanitized_tags = [str(tag) for tag in tags]
            if self._run_tree is None:
                self.initial_tags.extend(sanitized_tags)
            else:
                self._run_tree.add_tags(sanitized_tags)

    def finish_success(
        self,
        *,
        outputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        if self.finished:
            return
        self.annotate(metadata=metadata, tags=tags)
        if self._run_tree is not None:
            self._run_tree.end(
                outputs=None if outputs is None else sanitize_trace_payload(dict(outputs)),
            )
            self._context_manager.__exit__(None, None, None)
        self.finished = True
        self._reset_activation()

    def finish_error(
        self,
        error: BaseException | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        if self.finished:
            return
        self.annotate(metadata=metadata, tags=tags)
        if self._run_tree is not None:
            self._run_tree.end(error=str(error))
            self._context_manager.__exit__(None, None, None)
        self.finished = True
        self._reset_activation()

    def activate(self) -> Token[TraceSpanHandle | None]:
        return _CURRENT_TRACE_SPAN.set(self)

    def deactivate(self, token: Token[TraceSpanHandle | None]) -> None:
        _CURRENT_TRACE_SPAN.reset(token)

    def suspend(self) -> None:
        self._reset_activation()

    def __enter__(self) -> TraceSpanHandle:
        if self._context_manager is None:
            self._context_manager = trace(
                self.name,
                run_type=self.run_type,
                inputs=self.initial_inputs or None,
                project_name=self.sink.project_name,
                parent=self._resolve_parent_run(),
                tags=self.initial_tags or None,
                metadata=self.initial_metadata or None,
                client=self.sink.client,
            )
            self._run_tree = self._context_manager.__enter__()
        if self._activation_token is None:
            self._activation_token = self.activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        if exc_value is None:
            self.finish_success()
        else:
            self.finish_error(exc_value)
        return None

    def _resolve_parent_run(self) -> Any | None:
        if (
            isinstance(self.explicit_parent, LangSmithTraceSpan)
            and self.explicit_parent._run_tree is not None
        ):
            return self.explicit_parent._run_tree

        current_span = get_current_trace_span()
        if isinstance(current_span, LangSmithTraceSpan) and current_span._run_tree is not None:
            return current_span._run_tree

        return get_current_run_tree()

    def _reset_activation(self) -> None:
        if self._activation_token is not None:
            try:
                self.deactivate(self._activation_token)
            except ValueError:
                pass
            self._activation_token = None


def sanitize_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): sanitize_trace_value(value)
        for key, value in payload.items()
        if value is not None
    }


def sanitize_trace_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return sanitize_trace_payload(value)
    if isinstance(value, (list, tuple)):
        return [sanitize_trace_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
