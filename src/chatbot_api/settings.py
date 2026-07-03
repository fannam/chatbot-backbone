from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_name: str = "chatbot-api"
    database_url: str = "postgresql+psycopg://chatbot:chatbot@postgres:5432/chatbot"
    langgraph_checkpoint_database_url: str | None = (
        "postgresql://chatbot:chatbot@postgres:5432/chatbot"
    )
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_model_input_price_per_1m_tokens: float | None = 0.40
    openai_model_output_price_per_1m_tokens: float | None = 1.60
    llm_timeout_seconds: float = 30.0
    document_max_bytes: int = 5_242_880
    request_max_body_bytes: int = 10_485_760
    document_chunk_size_chars: int = 1200
    document_chunk_overlap_chars: int = 200
    document_embedding_dimensions: int = 1536
    document_embedding_batch_size: int = 32
    document_embedding_task_max_retries: int = 3
    document_embedding_task_retry_backoff_seconds: int = 30
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.35
    retrieval_max_chunks_per_document: int = 1
    retrieval_candidate_limit: int = 12
    tool_max_rounds: int = 4
    tool_execution_timeout_seconds: float = 15.0
    tool_search_top_k: int = 3
    auth_enabled: bool = False
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 60
    memory_enabled: bool = True
    memory_recent_message_window: int = 6
    memory_summary_trigger_messages: int = 12
    memory_max_summary_chars: int = 2000
    memory_max_active_items: int = 8
    memory_long_term_enabled: bool = True
    observability_json_logs: bool = True
    observability_metrics_enabled: bool = True
    observability_include_request_metadata: bool = False
    langsmith_tracing_enabled: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "chatbot-api"
    langsmith_endpoint: str | None = None
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"


def derive_langgraph_checkpoint_database_url(database_url: str) -> str | None:
    url = make_url(database_url)
    if not url.drivername.startswith("postgresql"):
        return None

    drivername = url.drivername.split("+", maxsplit=1)[0]
    return url.set(drivername=drivername).render_as_string(hide_password=False)


def parse_optional_float_env(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    if value == "":
        return None
    return float(value)


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


@lru_cache
def get_settings() -> Settings:
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://chatbot:chatbot@postgres:5432/chatbot",
    )
    checkpoint_database_url = os.getenv("LANGGRAPH_CHECKPOINT_DATABASE_URL")
    if checkpoint_database_url == "":
        checkpoint_database_url = None
    elif checkpoint_database_url is None:
        checkpoint_database_url = derive_langgraph_checkpoint_database_url(database_url)

    return Settings(
        app_name=os.getenv("APP_NAME", "chatbot-api"),
        database_url=database_url,
        langgraph_checkpoint_database_url=checkpoint_database_url,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        openai_model_input_price_per_1m_tokens=parse_optional_float_env(
            "OPENAI_MODEL_INPUT_PRICE_PER_1M_TOKENS",
            0.40,
        ),
        openai_model_output_price_per_1m_tokens=parse_optional_float_env(
            "OPENAI_MODEL_OUTPUT_PRICE_PER_1M_TOKENS",
            1.60,
        ),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        document_max_bytes=int(os.getenv("DOCUMENT_MAX_BYTES", "5242880")),
        request_max_body_bytes=int(os.getenv("REQUEST_MAX_BODY_BYTES", "10485760")),
        document_chunk_size_chars=int(os.getenv("DOCUMENT_CHUNK_SIZE_CHARS", "1200")),
        document_chunk_overlap_chars=int(
            os.getenv("DOCUMENT_CHUNK_OVERLAP_CHARS", "200")
        ),
        document_embedding_dimensions=int(
            os.getenv("DOCUMENT_EMBEDDING_DIMENSIONS", "1536")
        ),
        document_embedding_batch_size=int(
            os.getenv("DOCUMENT_EMBEDDING_BATCH_SIZE", "32")
        ),
        document_embedding_task_max_retries=int(
            os.getenv("DOCUMENT_EMBEDDING_TASK_MAX_RETRIES", "3")
        ),
        document_embedding_task_retry_backoff_seconds=int(
            os.getenv("DOCUMENT_EMBEDDING_TASK_RETRY_BACKOFF_SECONDS", "30")
        ),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
        retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.35")),
        retrieval_max_chunks_per_document=int(
            os.getenv("RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", "1")
        ),
        retrieval_candidate_limit=int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "12")),
        tool_max_rounds=int(os.getenv("TOOL_MAX_ROUNDS", "4")),
        tool_execution_timeout_seconds=float(
            os.getenv("TOOL_EXECUTION_TIMEOUT_SECONDS", "15")
        ),
        tool_search_top_k=int(os.getenv("TOOL_SEARCH_TOP_K", "3")),
        auth_enabled=parse_bool_env("AUTH_ENABLED", False),
        rate_limit_enabled=parse_bool_env("RATE_LIMIT_ENABLED", False),
        rate_limit_requests_per_minute=int(
            os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
        ),
        memory_enabled=parse_bool_env("MEMORY_ENABLED", True),
        memory_recent_message_window=int(os.getenv("MEMORY_RECENT_MESSAGE_WINDOW", "6")),
        memory_summary_trigger_messages=int(
            os.getenv("MEMORY_SUMMARY_TRIGGER_MESSAGES", "12")
        ),
        memory_max_summary_chars=int(os.getenv("MEMORY_MAX_SUMMARY_CHARS", "2000")),
        memory_max_active_items=int(os.getenv("MEMORY_MAX_ACTIVE_ITEMS", "8")),
        memory_long_term_enabled=parse_bool_env("MEMORY_LONG_TERM_ENABLED", True),
        observability_json_logs=os.getenv("OBSERVABILITY_JSON_LOGS", "true").lower()
        not in {"0", "false", "no"},
        observability_metrics_enabled=os.getenv(
            "OBSERVABILITY_METRICS_ENABLED",
            "true",
        ).lower()
        not in {"0", "false", "no"},
        observability_include_request_metadata=os.getenv(
            "OBSERVABILITY_INCLUDE_REQUEST_METADATA",
            "false",
        ).lower()
        in {"1", "true", "yes"},
        langsmith_tracing_enabled=parse_bool_env("LANGSMITH_TRACING", False),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "chatbot-api"),
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT") or None,
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        celery_broker_url=os.getenv(
            "CELERY_BROKER_URL",
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
        ),
    )
