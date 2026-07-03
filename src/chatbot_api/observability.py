from __future__ import annotations

import json
import logging
import sys
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from chatbot_api.providers import TokenUsage, UsageCost
from chatbot_api.settings import Settings, get_settings

MetricKind = Literal["counter", "histogram"]
LogLevel = Literal["debug", "info", "warning", "error"]

_REQUEST_CONTEXT: ContextVar[dict[str, str]] = ContextVar("request_context", default={})
_LOGGER = logging.getLogger("chatbot_api")
_LOGGER_CONFIGURED = False
_PROCESS_OBSERVABILITY: ObservabilityService | None = None

DEFAULT_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
DEFAULT_COUNT_BUCKETS = (0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0)
DEFAULT_SCORE_BUCKETS = (0.1, 0.25, 0.35, 0.5, 0.75, 0.9, 1.0)


def bind_request_context(*, request_id: str) -> Token[dict[str, str]]:
    return _REQUEST_CONTEXT.set({"request_id": request_id})


def reset_request_context(token: Token[dict[str, str]]) -> None:
    _REQUEST_CONTEXT.reset(token)


def get_request_id() -> str | None:
    return _REQUEST_CONTEXT.get({}).get("request_id")


def configure_json_logger() -> logging.Logger:
    global _LOGGER_CONFIGURED

    if _LOGGER_CONFIGURED:
        return _LOGGER

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    _LOGGER.handlers.clear()
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    _LOGGER_CONFIGURED = True
    return _LOGGER


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "logger": record.name,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()

        request_id = get_request_id()
        if request_id is not None and "request_id" not in payload:
            payload["request_id"] = request_id

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            if exc_type is not None:
                payload["exception_type"] = exc_type.__name__
            if exc_value is not None:
                payload["exception_message"] = str(exc_value)

        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    kind: MetricKind
    description: str
    labels: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()


@dataclass
class HistogramValue:
    count: int
    sum: float
    bucket_counts: list[int]


class MetricsRegistry:
    def __init__(self, definitions: list[MetricDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}
        self._lock = threading.Lock()
        self._counter_values: dict[str, dict[tuple[str, ...], float]] = {}
        self._histogram_values: dict[str, dict[tuple[str, ...], HistogramValue]] = {}

    def increment(self, name: str, *, labels: dict[str, str], amount: float = 1.0) -> None:
        definition = self._definitions[name]
        if definition.kind != "counter":
            raise ValueError(f"metric '{name}' is not a counter")

        label_values = self._label_values(definition, labels)
        with self._lock:
            metric_values = self._counter_values.setdefault(name, {})
            metric_values[label_values] = metric_values.get(label_values, 0.0) + amount

    def observe(self, name: str, *, labels: dict[str, str], value: float) -> None:
        definition = self._definitions[name]
        if definition.kind != "histogram":
            raise ValueError(f"metric '{name}' is not a histogram")

        label_values = self._label_values(definition, labels)
        with self._lock:
            metric_values = self._histogram_values.setdefault(name, {})
            histogram = metric_values.get(label_values)
            if histogram is None:
                histogram = HistogramValue(
                    count=0,
                    sum=0.0,
                    bucket_counts=[0 for _ in definition.buckets],
                )
                metric_values[label_values] = histogram

            histogram.count += 1
            histogram.sum += value
            for index, bucket in enumerate(definition.buckets):
                if value <= bucket:
                    histogram.bucket_counts[index] += 1

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for definition in self._definitions.values():
                lines.append(f"# HELP {definition.name} {definition.description}")
                lines.append(f"# TYPE {definition.name} {definition.kind}")
                if definition.kind == "counter":
                    metric_values = self._counter_values.get(definition.name, {})
                    for label_values, value in sorted(metric_values.items()):
                        lines.append(
                            (
                                f"{definition.name}"
                                f"{render_label_set(definition.labels, label_values)} "
                                f"{value:g}"
                            )
                        )
                    continue

                metric_values = self._histogram_values.get(definition.name, {})
                for label_values, histogram in sorted(metric_values.items()):
                    cumulative = 0
                    for index, bucket in enumerate(definition.buckets):
                        cumulative += histogram.bucket_counts[index]
                        bucket_labels = render_histogram_labels(
                            definition.labels,
                            label_values,
                            le=str(bucket),
                        )
                        lines.append(f"{definition.name}_bucket{bucket_labels} {cumulative}")

                    inf_labels = render_histogram_labels(
                        definition.labels,
                        label_values,
                        le="+Inf",
                    )
                    lines.append(f"{definition.name}_bucket{inf_labels} {histogram.count}")
                    label_set = render_label_set(definition.labels, label_values)
                    lines.append(f"{definition.name}_count{label_set} {histogram.count}")
                    lines.append(f"{definition.name}_sum{label_set} {histogram.sum:g}")

            return "\n".join(lines) + ("\n" if lines else "")

    def reset(self) -> None:
        with self._lock:
            self._counter_values.clear()
            self._histogram_values.clear()

    def _label_values(
        self,
        definition: MetricDefinition,
        labels: dict[str, str],
    ) -> tuple[str, ...]:
        missing = [label for label in definition.labels if label not in labels]
        if missing:
            raise ValueError(f"metric '{definition.name}' is missing labels: {', '.join(missing)}")
        return tuple(str(labels[label]) for label in definition.labels)


class ObservabilityService:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved_settings = settings or get_settings()
        self._json_logs_enabled = resolved_settings.observability_json_logs
        self._metrics_enabled = resolved_settings.observability_metrics_enabled
        self._include_request_metadata = resolved_settings.observability_include_request_metadata
        self._logger = configure_json_logger()
        self._metrics = MetricsRegistry(build_metric_definitions())

    @property
    def include_request_metadata(self) -> bool:
        return self._include_request_metadata

    def log_event(
        self,
        event: str,
        *,
        level: LogLevel = "info",
        **fields: Any,
    ) -> None:
        if not self._json_logs_enabled:
            return

        payload = {
            "event": event,
            **sanitize_log_fields(fields),
        }
        self._logger.log(resolve_log_level(level), payload)

    def increment(self, name: str, *, labels: dict[str, str], amount: float = 1.0) -> None:
        if not self._metrics_enabled:
            return
        self._metrics.increment(name, labels=labels, amount=amount)

    def observe(self, name: str, *, labels: dict[str, str], value: float) -> None:
        if not self._metrics_enabled:
            return
        self._metrics.observe(name, labels=labels, value=value)

    def render_metrics(self) -> str:
        if not self._metrics_enabled:
            return "# metrics disabled\n"
        return self._metrics.render()

    def reset_for_tests(self) -> None:
        self._metrics.reset()

    def record_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        labels = {
            "method": method,
            "route": route,
            "status_code": str(status_code),
        }
        self.increment("http_requests_total", labels=labels)
        self.observe("http_request_duration_seconds", labels=labels, value=duration_seconds)

    def record_chat_request(
        self,
        *,
        mode: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        labels = {"mode": mode, "outcome": outcome}
        self.increment("chat_requests_total", labels=labels)
        self.observe("chat_request_duration_seconds", labels=labels, value=duration_seconds)

    def record_chat_stream_disconnect(self) -> None:
        self.increment("chat_stream_disconnects_total", labels={})

    def record_chat_workflow(
        self,
        *,
        mode: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        labels = {"mode": mode, "outcome": outcome}
        self.increment("chat_workflow_runs_total", labels=labels)
        self.observe("chat_workflow_duration_seconds", labels=labels, value=duration_seconds)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        labels = {"tool_name": tool_name, "status": status}
        self.increment("tool_calls_total", labels=labels)
        self.observe("tool_execution_duration_seconds", labels=labels, value=duration_seconds)

    def record_retrieval(
        self,
        *,
        outcome: str,
        selected_chunk_count: int,
        top_score: float | None,
    ) -> None:
        labels = {"outcome": outcome}
        self.increment("retrieval_requests_total", labels=labels)
        self.observe(
            "retrieval_selected_chunks",
            labels=labels,
            value=float(selected_chunk_count),
        )
        if top_score is not None:
            self.observe("retrieval_top_score", labels=labels, value=top_score)

    def record_llm_request(
        self,
        *,
        model: str,
        outcome: str,
        duration_seconds: float,
        usage: TokenUsage | None,
        cost: UsageCost | None,
    ) -> None:
        labels = {"model": model, "outcome": outcome}
        self.increment("llm_requests_total", labels=labels)
        self.observe("llm_request_duration_seconds", labels=labels, value=duration_seconds)
        if usage is not None:
            self.increment(
                "llm_input_tokens_total",
                labels={"model": model},
                amount=float(usage.input_tokens),
            )
            self.increment(
                "llm_output_tokens_total",
                labels={"model": model},
                amount=float(usage.output_tokens),
            )
            self.increment(
                "llm_total_tokens_total",
                labels={"model": model},
                amount=float(usage.total_tokens),
            )
        if cost is not None:
            self.increment(
                "llm_request_cost_usd_total",
                labels={"model": model},
                amount=cost.total_cost_usd,
            )

    def record_document_upload(
        self,
        *,
        outcome: str,
    ) -> None:
        self.increment("document_upload_requests_total", labels={"outcome": outcome})

    def record_auth_attempt(
        self,
        *,
        outcome: str,
    ) -> None:
        self.increment("auth_attempts_total", labels={"outcome": outcome})

    def record_document_embedding_job(
        self,
        *,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        labels = {"outcome": outcome}
        self.increment("document_embedding_jobs_total", labels=labels)
        self.observe(
            "document_embedding_job_duration_seconds",
            labels=labels,
            value=duration_seconds,
        )


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            sanitized[key] = round(value, 6)
            continue
        sanitized[key] = value
    return sanitized


def resolve_log_level(level: LogLevel) -> int:
    if level == "debug":
        return logging.DEBUG
    if level == "warning":
        return logging.WARNING
    if level == "error":
        return logging.ERROR
    return logging.INFO


def render_label_set(label_names: tuple[str, ...], label_values: tuple[str, ...]) -> str:
    if not label_names:
        return ""

    serialized = ",".join(
        f'{name}="{escape_label_value(value)}"'
        for name, value in zip(label_names, label_values, strict=True)
    )
    return "{" + serialized + "}"


def render_histogram_labels(
    label_names: tuple[str, ...],
    label_values: tuple[str, ...],
    *,
    le: str,
) -> str:
    combined_names = (*label_names, "le")
    combined_values = (*label_values, le)
    return render_label_set(combined_names, combined_values)


def escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def build_metric_definitions() -> list[MetricDefinition]:
    return [
        MetricDefinition(
            name="http_requests_total",
            kind="counter",
            description="Total HTTP requests handled by the API.",
            labels=("method", "route", "status_code"),
        ),
        MetricDefinition(
            name="http_request_duration_seconds",
            kind="histogram",
            description="Latency for handled HTTP requests.",
            labels=("method", "route", "status_code"),
            buckets=DEFAULT_DURATION_BUCKETS,
        ),
        MetricDefinition(
            name="chat_requests_total",
            kind="counter",
            description="Total chat requests by mode and outcome.",
            labels=("mode", "outcome"),
        ),
        MetricDefinition(
            name="chat_request_duration_seconds",
            kind="histogram",
            description="Latency for chat requests by mode and outcome.",
            labels=("mode", "outcome"),
            buckets=DEFAULT_DURATION_BUCKETS,
        ),
        MetricDefinition(
            name="chat_stream_disconnects_total",
            kind="counter",
            description="Total disconnected chat streams.",
        ),
        MetricDefinition(
            name="chat_workflow_runs_total",
            kind="counter",
            description="Total workflow runs by mode and outcome.",
            labels=("mode", "outcome"),
        ),
        MetricDefinition(
            name="chat_workflow_duration_seconds",
            kind="histogram",
            description="Workflow execution duration by mode and outcome.",
            labels=("mode", "outcome"),
            buckets=DEFAULT_DURATION_BUCKETS,
        ),
        MetricDefinition(
            name="tool_calls_total",
            kind="counter",
            description="Total tool calls by tool and terminal status.",
            labels=("tool_name", "status"),
        ),
        MetricDefinition(
            name="tool_execution_duration_seconds",
            kind="histogram",
            description="Tool execution duration by tool and terminal status.",
            labels=("tool_name", "status"),
            buckets=DEFAULT_DURATION_BUCKETS,
        ),
        MetricDefinition(
            name="retrieval_requests_total",
            kind="counter",
            description="Total retrieval requests by outcome.",
            labels=("outcome",),
        ),
        MetricDefinition(
            name="retrieval_selected_chunks",
            kind="histogram",
            description="Number of selected retrieval chunks.",
            labels=("outcome",),
            buckets=DEFAULT_COUNT_BUCKETS,
        ),
        MetricDefinition(
            name="retrieval_top_score",
            kind="histogram",
            description="Top retrieval score when results exist.",
            labels=("outcome",),
            buckets=DEFAULT_SCORE_BUCKETS,
        ),
        MetricDefinition(
            name="llm_requests_total",
            kind="counter",
            description="Total provider LLM requests by model and outcome.",
            labels=("model", "outcome"),
        ),
        MetricDefinition(
            name="llm_request_duration_seconds",
            kind="histogram",
            description="Latency for provider LLM requests by model and outcome.",
            labels=("model", "outcome"),
            buckets=DEFAULT_DURATION_BUCKETS,
        ),
        MetricDefinition(
            name="llm_input_tokens_total",
            kind="counter",
            description="Total input tokens sent to the LLM by model.",
            labels=("model",),
        ),
        MetricDefinition(
            name="llm_output_tokens_total",
            kind="counter",
            description="Total output tokens received from the LLM by model.",
            labels=("model",),
        ),
        MetricDefinition(
            name="llm_total_tokens_total",
            kind="counter",
            description="Total LLM tokens by model.",
            labels=("model",),
        ),
        MetricDefinition(
            name="llm_request_cost_usd_total",
            kind="counter",
            description="Estimated total LLM request cost in USD by model.",
            labels=("model",),
        ),
        MetricDefinition(
            name="document_upload_requests_total",
            kind="counter",
            description="Total document upload requests by outcome.",
            labels=("outcome",),
        ),
        MetricDefinition(
            name="auth_attempts_total",
            kind="counter",
            description="Total API key authentication attempts by outcome.",
            labels=("outcome",),
        ),
        MetricDefinition(
            name="document_embedding_jobs_total",
            kind="counter",
            description="Total document embedding jobs by outcome.",
            labels=("outcome",),
        ),
        MetricDefinition(
            name="document_embedding_job_duration_seconds",
            kind="histogram",
            description="Embedding job duration by outcome.",
            labels=("outcome",),
            buckets=DEFAULT_DURATION_BUCKETS,
        ),
    ]


def get_process_observability(settings: Settings | None = None) -> ObservabilityService:
    global _PROCESS_OBSERVABILITY

    if _PROCESS_OBSERVABILITY is None:
        _PROCESS_OBSERVABILITY = ObservabilityService(settings)
    return _PROCESS_OBSERVABILITY


def reset_process_observability() -> None:
    global _PROCESS_OBSERVABILITY
    if _PROCESS_OBSERVABILITY is not None:
        _PROCESS_OBSERVABILITY.reset_for_tests()
    _PROCESS_OBSERVABILITY = None


def normalize_route_path(route: str | None, fallback_path: str) -> str:
    if route:
        return route
    return fallback_path or "/"
