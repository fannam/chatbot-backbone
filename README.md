# Chatbot API

Phase 0 bootstrap for the chatbot backend. This scaffold provides:

- FastAPI application with a `GET /health` endpoint
- Chat API with `POST /chat` backed by an OpenAI provider abstraction
- Async-first document ingestion API with `POST /documents` and `GET /documents/{id}`
- Tool-aware chat with allowlisted structured actions and tool trace metadata
- Request-scoped profile lookup via the `get_current_user_profile` tool
- Knowledge-base search tool backed by chunk embeddings and source metadata
- Optional SSE streaming for `POST /chat` via `stream: true`
- Local-first observability with structured logs, `X-Request-ID`, and `GET /metrics`
- Optional LangSmith tracing for `POST /chat`, LangGraph workflow, tool calls, and retrieval
- PostgreSQL-backed persistence for conversations and messages
- PostgreSQL-backed persistence for tool execution history
- PostgreSQL-backed persistence for document metadata and chunks
- Celery worker for background document embedding jobs
- LangGraph workflow orchestration with durable checkpoints on PostgreSQL
- Python project management with `uv`
- pytest and Ruff configuration
- Docker Compose services for API, PostgreSQL, and Redis
- Alembic migration for the initial chat tables

## Prerequisites

- Python 3.12+
- `uv`
- Docker and Docker Compose

## Local development

Create a local environment file:

```bash
cp .env.example .env
```

Set the OpenAI credentials in `.env` before using `POST /chat`.

Install dependencies:

```bash
uv sync --extra dev
```

Run the API in reload mode:

```bash
uv run uvicorn chatbot_api.main:app --reload --app-dir src
```

Run the background worker in a second terminal:

```bash
uv run celery -A chatbot_api.tasks.celery_app:celery_app worker --loglevel=info
```

Example chat request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello",
    "metadata": {"source": "manual-test"}
  }'
```

Expected response shape:

```json
{
  "conversation_id": "generated-or-supplied-id",
  "message": {
    "role": "assistant",
    "content": "..."
  },
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "metadata": {
    "citations": [
      {
        "document_id": "stored-document-id",
        "filename": "notes.md",
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": 120,
        "snippet": "Relevant excerpt..."
      }
    ],
    "tool_runs": [
      {
        "tool_call_id": "fc_123",
        "tool_name": "search_knowledge_base",
        "status": "completed",
        "input": {
          "query": "What does the guide say?"
        },
        "output": {
          "hits": []
        }
      }
    ],
    "usage": {
      "input_tokens": 120,
      "output_tokens": 30,
      "total_tokens": 150
    },
    "cost": {
      "input_cost_usd": 0.000048,
      "output_cost_usd": 0.000048,
      "total_cost_usd": 0.000096,
      "currency": "USD"
    }
  }
}
```

`POST /chat` persists both the user and assistant turns. When a caller sends an
existing `conversation_id`, the stored history is loaded and replayed to the LLM
provider before generating the next answer. When retrieval finds relevant chunks,
the model can call the allowlisted `search_knowledge_base` tool and the response
includes both citation metadata and executed `tool_runs` without altering
`message.content`. Tool execution is conservative by default: tools are
allowlisted, validated with schemas, and time-bounded.

The request `metadata` object is also the source for request-scoped tool context.
The reserved key `metadata.user_profile` is used by the allowlisted
`get_current_user_profile` tool. The current payload shape is:

```json
{
  "user_profile": {
    "user_id": "user-123",
    "display_name": "Alice",
    "email": "alice@example.com",
    "plan": "pro",
    "locale": "en-US",
    "preferences": {
      "timezone": "UTC"
    }
  }
}
```

If `user_profile` is missing or malformed, the tool still completes normally and
returns `{"found": false, "profile": null}` instead of failing the tool call.

## API key auth

Phase 6 adds an optional API key auth layer for stateful endpoints. When
`AUTH_ENABLED=true`, `/chat`, `/documents`, `/documents/{id}`,
`/conversations/{conversation_id}/tool-runs`,
`/conversations/{conversation_id}/memory`, and `/users/{user_id}/memories`
require `X-API-Key`.

Provision a user plus one API key:

```bash
uv run python -m chatbot_api.tasks.create_api_key \
  --user-id user-123 \
  --name dev \
  --display-name "Alice" \
  --email alice@example.com \
  --plan pro \
  --locale en-US \
  --preferences-json '{"timezone":"UTC"}'
```

The command prints the plaintext API key once. The database stores only its hash
plus a short prefix for debugging.

When auth is enabled, the server overwrites `metadata.user_profile` with the
authenticated user profile before the workflow, tools, and memory layer read it.
Client-supplied `user_profile.user_id` is therefore not trusted.

Example chat request with request-scoped profile data:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "message": "What plan am I on?",
    "metadata": {
      "user_profile": {
        "user_id": "user-123",
        "display_name": "Alice",
        "plan": "pro",
        "preferences": {"timezone": "UTC"}
      }
    }
  }'
```

Example document upload:

```bash
curl -X POST http://localhost:8000/documents \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@/path/to/notes.md"
```

Expected response shape:

```json
{
  "document_id": "generated-id",
  "filename": "notes.md",
  "content_type": "text/markdown",
  "byte_size": 1234,
  "status": "processing",
  "chunk_count": 2,
  "created_at": "2026-06-04T00:00:00Z"
}
```

`POST /documents` accepts UTF-8 `txt`/`md` files and text-based PDFs. The API
extracts text, chunks it deterministically, persists the document with
`status=processing`, then enqueues a Celery job to generate embeddings in the
background. Retrieval only uses documents after they are marked `ready`.
Uploading byte-identical content already stored for the same owner (or globally
when auth is disabled) is rejected with `409 Conflict` referencing the existing
document's ID, based on a SHA-256 checksum computed before text extraction.

Poll the document status:

```bash
curl http://localhost:8000/documents/generated-id
```

Example status response:

```json
{
  "document_id": "generated-id",
  "filename": "notes.md",
  "content_type": "text/markdown",
  "byte_size": 1234,
  "status": "ready",
  "chunk_count": 2,
  "created_at": "2026-06-04T00:00:00Z",
  "updated_at": "2026-06-04T00:00:05Z",
  "failure_reason": null
}
```

Inspect tool execution history for a conversation:

```bash
curl "http://localhost:8000/conversations/generated-or-supplied-id/tool-runs?limit=50"
```

Expected response shape:

```json
{
  "conversation_id": "generated-or-supplied-id",
  "tool_runs": [
    {
      "id": 2,
      "tool_call_id": "fc_123",
      "tool_name": "search_knowledge_base",
      "status": "completed",
      "input": {
        "query": "What does the guide say?"
      },
      "output": {
        "hits": []
      },
      "error": null,
      "started_at": "2026-06-08T00:00:00Z",
      "completed_at": "2026-06-08T00:00:01Z"
    }
  ]
}
```

`GET /conversations/{conversation_id}/tool-runs` is a read-only debug endpoint.
It returns the newest tool runs first, supports `limit` from `1` to `100`, and
returns `404` when the conversation does not exist or belongs to a different
authenticated user.

## Request limits

Every request passes through a general request size limit middleware before it
reaches any route handler. Requests whose declared or actual body size exceeds
`REQUEST_MAX_BODY_BYTES` (default 10 MiB) are rejected with `413 Request Entity
Too Large`. This is independent of `DOCUMENT_MAX_BYTES`, which is a friendlier,
document-specific validation error applied after `POST /documents` has already
read the file.

Set `RATE_LIMIT_ENABLED=true` to enable per-API-key (or per-client-IP when
unauthenticated) rate limiting, capped at `RATE_LIMIT_REQUESTS_PER_MINUTE`
requests per rolling 60-second window. Requests over the limit receive `429 Too
Many Requests` with a `Retry-After` header. `GET /health` and `GET /metrics` are
always exempt. Rate limit state is in-process only; it resets on restart and is
not shared across multiple API replicas.

Set `MODERATION_ENABLED=true` to screen every `POST /chat` message through the
OpenAI Moderation API before it reaches the workflow. Flagged messages receive
`400 Bad Request` and never reach the LLM, the retriever, or persistence. This
is off by default to avoid the extra network call/cost on every chat turn; when
enabled, a moderation-API failure fails closed (`503`) rather than letting an
unscreened message through.

## Memory debug endpoints

Phase 5 keeps `/chat` unchanged and exposes memory through separate debug
endpoints:

```bash
curl -H "X-API-Key: <your-api-key>" \
  http://localhost:8000/conversations/generated-or-supplied-id/memory
curl -H "X-API-Key: <your-api-key>" \
  http://localhost:8000/users/user-123/memories
curl -X DELETE -H "X-API-Key: <your-api-key>" \
  http://localhost:8000/users/user-123/memories/1
```

`GET /conversations/{conversation_id}/memory` returns the current rolling
conversation summary snapshot when it exists. `GET /users/{user_id}/memories`
returns active long-term user memories, and
`DELETE /users/{user_id}/memories/{memory_id}` soft-deletes one memory item.
When auth is enabled, those endpoints are scoped to the authenticated user.

## Observability

Every HTTP response includes an `X-Request-ID` header. If the client sends one,
the API preserves it; otherwise the API generates one. Structured JSON logs use
the same request ID so a single chat, upload, or SSE stream can be correlated
across request logs and internal tool/retrieval events.

`POST /chat` can also emit structured traces through a trace sink abstraction.
The default sink is a no-op. When LangSmith is enabled, the API traces the chat
request root span plus nested workflow, model round, tool, and retrieval spans
without changing the HTTP or SSE response contracts.

Prometheus-style metrics are available at:

```bash
curl http://localhost:8000/metrics
```

Current metrics include HTTP request totals/latency, chat request totals/latency,
workflow totals, provider LLM request totals/latency, LLM token/cost counters,
tool execution totals/latency, retrieval hit counters, document upload totals,
document embedding job totals/latency, API key authentication attempt totals,
and content moderation check totals (`moderation_checks_total`, when
`MODERATION_ENABLED=true`).

Sensitive actions are logged as structured audit events through the same JSON log
stream (filterable by `event` name): `auth.failed` (missing/invalid API key, with
`reason` and, for invalid keys, `api_key_prefix`), `memory.access.forbidden`
(cross-user access rejected on the memory debug endpoints), and
`memory.delete.completed`/`memory.delete.rejected`. Document upload events
(`document.upload.*`) also carry `owner_user_id` when auth is enabled, and
include a `duplicate` outcome when a byte-identical document already exists for
the same owner. `moderation.blocked` is logged whenever a message is rejected by
the moderation gate.

Observability-related environment variables:

```bash
OBSERVABILITY_JSON_LOGS=true
OBSERVABILITY_LOG_LEVEL=info
OBSERVABILITY_METRICS_ENABLED=true
OBSERVABILITY_INCLUDE_REQUEST_METADATA=false
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=chatbot-api
LANGSMITH_ENDPOINT=
OPENAI_MODEL_INPUT_PRICE_PER_1M_TOKENS=0.40
OPENAI_MODEL_OUTPUT_PRICE_PER_1M_TOKENS=1.60
```

Despite the name, `OBSERVABILITY_JSON_LOGS` is a full logging kill-switch, not
a JSON-vs-plain-text format toggle — there is no alternate plain-text
formatter, so setting it to `false` disables all structured logging entirely
(metrics/tracing are unaffected). `OBSERVABILITY_LOG_LEVEL` controls the
minimum level actually emitted (`debug`/`info`/`warning`/`error`).
`OBSERVABILITY_INCLUDE_REQUEST_METADATA` stays off by default to avoid logging
arbitrary caller metadata. If pricing env vars are blank, the API still reports
token usage but omits estimated cost.

To enable LangSmith tracing, set `LANGSMITH_TRACING=true` and provide
`LANGSMITH_API_KEY`. `LANGSMITH_PROJECT` defaults to `chatbot-api`, and
`LANGSMITH_ENDPOINT` is only needed for non-default or self-hosted deployments.

Both the sync and SSE paths now go through the same LangGraph workflow:
`load_context -> call_model -> execute_tools? -> call_model -> persist_response`.

To stream the assistant response over SSE, send `stream: true`:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello",
    "stream": true
  }'
```

Streaming responses use `text/event-stream` and emit these events:

- `message_start`
- `tool_start`
- `tool_complete`
- `tool_error`
- `message_delta`
- `message_complete`
- `error`

The endpoint keeps the existing JSON behavior when `stream` is omitted or `false`.
When tool execution succeeds, the `message_complete` payload can include both
`metadata.citations` and `metadata.tool_runs`. The terminal JSON and SSE payloads
can also include aggregated `metadata.usage` and `metadata.cost` for the full
chat request, including extra model rounds triggered by tool calling.

## Guardrails

Guardrails are built on [Guardrails AI](https://github.com/guardrails-ai/guardrails)
(`AsyncGuard`), using only locally-defined custom validators
(`src/chatbot_api/workflow/guardrails.py`) — no Guardrails Hub account or API key is
required, and no validator ever calls out to a third party (the Hub's usage
telemetry is explicitly disabled at import time; see the module docstring for
details). Four independent, opt-in checks are layered on `POST /chat`:

- **Input moderation** (`MODERATION_ENABLED`): the existing OpenAI Moderation
  API check on the user's message, unchanged from Phase 6.
- **Input jailbreak heuristic** (`JAILBREAK_DETECTION_ENABLED`): a regex-based
  scan of the user's message for common jailbreak/prompt-injection phrasing
  (e.g. "ignore previous instructions", "you are now DAN"). A match returns
  `400 Bad Request` before the workflow runs.
- **Input PII detection** (`PII_DETECTION_ENABLED`): a regex/heuristic scan
  for email, phone, credit-card (Luhn-checked), and generic long-digit IDs in
  the user's message. This is **log-only** — it never blocks or alters the
  message, only records a `guardrail.input.pii_detected` audit event.
- **Output guardrails** (`OUTPUT_GUARDRAILS_ENABLED`): runs after the LLM
  produces its final response and before it is persisted or streamed back to
  the client (a new `output_guardrail` node between `call_model` and
  `persist_response` in the LangGraph workflow). PII in the response is
  redacted in place (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`,
  `[REDACTED_CARD]`, `[REDACTED_ID]`); a flagged moderation result substitutes
  a canned refusal message instead of the model's actual response.

Audit log events: `guardrail.input.blocked`, `guardrail.input.pii_detected`,
`guardrail.output.redacted`, `guardrail.output.blocked`. Metric:
`guardrail_checks_total{direction,check,outcome}`.

Run the jailbreak/PII regression eval to sanity-check the heuristics against a
curated set of true positives, deliberate false-positive traps, and PII cases:

```bash
uv run python -m chatbot_api.evals.guardrails_eval --dataset evals/guardrails_jailbreak.json
```

**What this does NOT cover:**

- The jailbreak heuristic is pattern-based, not adversarially robust — a
  paraphrased or novel attack can still get through. Treat it as one signal
  among several (system prompt, moderation, output guardrails), not a silver
  bullet.
- Tool output (e.g. `search_knowledge_base` results) is **not** re-checked by
  the output guard before being fed back into the next model round. This is
  explicitly out of scope because all current tools only operate on the
  organization's own trusted, pre-ingested documents, not arbitrary external
  content — this would need revisiting if a future tool ever fetches
  untrusted external content (e.g. web browsing).
- PII detection is regex/heuristic-based, not an NER model, so it will miss
  unusual formats and can false-positive on PII-shaped-but-not-PII numbers
  (only credit cards are precision-checked, via a Luhn checksum).

## Quality checks

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

## Docker Compose

Start the local stack:

```bash
docker compose up --build
```

The stack exposes:

- API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- Worker: Celery process consuming document embedding jobs

## Database migrations

Apply the current schema with:

```bash
uv run alembic upgrade head
```

Create a future revision with:

```bash
uv run alembic revision -m "describe change"
```

Enqueue embedding jobs for existing documents that still have missing chunk
embeddings:

```bash
uv run python -m chatbot_api.tasks.reindex_embeddings
```

## Retrieval eval baseline

The repo includes an offline retrieval baseline to measure document selection
quality before changing retrieval thresholds, ANN indexes, or reranking:

```bash
uv run python -m chatbot_api.evals.rag_eval \
  --dataset evals/rag_retrieval_baseline.json \
  --output .artifacts/rag-eval-report.json
```

The checked-in dataset uses stable `filename` expectations because uploaded
`document_id` values are generated at ingest time. Each case declares
`expected_sources`, where every source can match by `filename`, `document_id`,
or both, plus optional `chunk_indexes` for chunk-level checks.

If you want a reproducible starter corpus, upload the files in
`evals/fixtures/` first, wait until each document is `ready`, then run the eval
command. The summary prints document hit rate, average expected-document
coverage, optional chunk hit rate, and failed case IDs. The JSON report includes
per-case retrieved chunks and the active retrieval config.

## Chat/tool regression eval

The repo also includes a deterministic service-level regression eval for the
`ChatService` + LangGraph + tool-calling path:

```bash
uv run python -m chatbot_api.evals.chat_eval \
  --dataset evals/chat_tool_regression.json \
  --output .artifacts/chat-eval-report.json
```

This eval does not call a live chat model. Instead, it uses a scripted provider
for the model rounds while keeping the real workflow, tool registry, retrieval,
repository persistence, citation propagation, and usage/cost aggregation in the
execution path. That makes it stable enough for regression checks without
changing the `/chat` API contract.

The checked-in dataset covers:

- `search_knowledge_base` grounded answer flow
- cross-document disambiguation with seeded conversation history
- `calculator` happy path
- rejected non-allowlisted tool calls
- request-scoped `get_current_user_profile` success and missing-profile flows

If you want a reproducible starter corpus for the search cases, upload the
files in `evals/fixtures/` first and wait until each document is `ready`. The
summary prints pass rate plus tool/source/answer match rates, and the JSON
report includes per-case failures, tool runs, citations, and aggregated
usage/cost metadata.

## Memory regression eval

The repo also includes a deterministic memory-focused regression eval for the
real `ChatService` + `MemoryManager` path:

```bash
uv run python -m chatbot_api.evals.memory_eval \
  --dataset evals/memory_regression.json \
  --output .artifacts/memory-eval-report.json
```

This eval keeps the real workflow, persistence, short-term summary refresh, and
long-term memory extraction/injection logic, but replaces the model with a
scripted provider. That makes it stable enough to regression-test prompt
injection, summary writes, allowlisted long-term extraction, and failure paths
without calling a live model.

The checked-in dataset covers:

- injecting seeded conversation summaries and active memories into the chat prompt
- refreshing the rolling conversation summary when unsummarized history exceeds
  the configured window
- rule-based extraction for durable preferences such as preferred name,
  language, timezone, and response style
- LLM-based extraction for the allowlisted profile keys
  `profile.role/profile.company/profile.team`
- invalid extraction payloads that must not persist memory
- requests without `metadata.user_profile.user_id`, which must skip long-term
  writes

The summary prints pass rate plus prompt/summary/memory/answer match rates, and
the JSON report includes per-case provider prompts, stored summary state, and
active memories after the run.

## Combined eval suite

For a single local/dev quality gate that checks corpus readiness and runs the
retrieval, chat/tool, and memory evals together:

```bash
uv run python -m chatbot_api.evals.eval_suite \
  --rag-dataset evals/rag_retrieval_baseline.json \
  --chat-dataset evals/chat_tool_regression.json \
  --memory-dataset evals/memory_regression.json \
  --output-dir .artifacts
```

The suite first checks that every expected filename referenced by the checked-in
datasets exists in the database and has at least one document in `ready` state.
If preflight fails, the suite writes `.artifacts/eval-suite-report.json` and
exits non-zero without running the child evals.

When preflight passes, the suite writes:

- `.artifacts/rag-eval-report.json`
- `.artifacts/chat-eval-report.json`
- `.artifacts/memory-eval-report.json`
- `.artifacts/eval-suite-report.json`

Default quality gates are strict for local/dev regression use:

- retrieval `document_hit_rate >= 1.0`
- retrieval `chunk_hit_rate >= 1.0` when chunk expectations exist
- chat/tool `pass_rate >= 1.0`
- memory `pass_rate >= 1.0`

You can relax them with `--min-rag-document-hit-rate`,
`--min-rag-chunk-hit-rate`, `--min-chat-pass-rate`, and
`--min-memory-pass-rate`.

## Environment variables

- `OPENAI_API_KEY`: required for `POST /chat`
- `OPENAI_MODEL`: defaults to `gpt-4.1-mini`
- `OPENAI_EMBEDDING_MODEL`: defaults to `text-embedding-3-small`
- `LLM_TIMEOUT_SECONDS`: request timeout for the upstream LLM call
- `DOCUMENT_MAX_BYTES`: upload size limit for `POST /documents`
- `REQUEST_MAX_BODY_BYTES`: general request body size limit enforced by middleware
  for every endpoint, ahead of any per-endpoint validation
- `DOCUMENT_CHUNK_SIZE_CHARS`: target chunk size for stored document chunks
- `DOCUMENT_CHUNK_OVERLAP_CHARS`: overlap between adjacent stored document chunks
- `DOCUMENT_EMBEDDING_DIMENSIONS`: embedding vector size for stored chunks
- `DOCUMENT_EMBEDDING_BATCH_SIZE`: number of chunks fetched and embedded per
  embedding-provider call in the embedding worker
- `DOCUMENT_REINDEX_PAGE_SIZE`: number of documents fetched per DB page when
  scanning for documents missing embeddings (`reindex_embeddings.py`)
- `DOCUMENT_EMBEDDING_TASK_MAX_RETRIES`: maximum Celery retries for temporary embedding failures
- `DOCUMENT_EMBEDDING_TASK_RETRY_BACKOFF_SECONDS`: base exponential backoff for embedding retries
- `RETRIEVAL_TOP_K`: number of retrieved chunks injected into chat
- `RETRIEVAL_MIN_SCORE`: minimum similarity score required before a chunk is used
- `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT`: per-document cap applied after ranking
- `RETRIEVAL_CANDIDATE_LIMIT`: number of ranked candidates fetched before filtering and dedupe
- `RETRIEVAL_RERANK_ENABLED`: rerank the candidate pool with an LLM call before
  final selection; off by default (adds one extra LLM call per retrieval)
- `TOOL_MAX_ROUNDS`: maximum provider rounds allowed for tool execution
- `TOOL_EXECUTION_TIMEOUT_SECONDS`: timeout applied to one tool execution
- `TOOL_SEARCH_TOP_K`: default top-k used by the knowledge-base search tool
- `TOOL_SEARCH_MAX_TOP_K`: upper bound the knowledge-base search tool clamps a
  caller-supplied `top_k` to
- `AUTH_ENABLED`: require `X-API-Key` for stateful API endpoints
- `RATE_LIMIT_ENABLED`: enable per-API-key (or per-IP when unauthenticated)
  request rate limiting middleware
- `RATE_LIMIT_REQUESTS_PER_MINUTE`: request budget per rolling 60-second window
  when rate limiting is enabled
- `MODERATION_ENABLED`: check each `POST /chat` message against the OpenAI
  Moderation API before it reaches the workflow; off by default
- `MODERATION_MODEL`: moderation model used when `MODERATION_ENABLED=true`,
  defaults to `omni-moderation-latest`
- `JAILBREAK_DETECTION_ENABLED`: scan `POST /chat` messages for common
  jailbreak/prompt-injection phrasing before the workflow runs; off by default
- `PII_DETECTION_ENABLED`: log (never block) detected PII in `POST /chat`
  messages; off by default
- `OUTPUT_GUARDRAILS_ENABLED`: redact PII and hard-block flagged responses
  from the LLM before they are persisted or streamed back to the client; off
  by default
- `MEMORY_ENABLED`: enables short-term and long-term memory nodes in the workflow
- `MEMORY_RECENT_MESSAGE_WINDOW`: number of newest raw messages kept outside the rolling summary
- `MEMORY_SUMMARY_TRIGGER_MESSAGES`: unsummarized message count required before refreshing the rolling summary
- `MEMORY_MAX_SUMMARY_CHARS`: maximum summary size produced by the summary updater
- `MEMORY_MAX_ACTIVE_ITEMS`: maximum active long-term memories injected into the prompt
- `MEMORY_LONG_TERM_ENABLED`: enables long-term memory extraction and injection
- `DATABASE_URL`: SQLAlchemy connection string used for chat persistence
- `LANGGRAPH_CHECKPOINT_DATABASE_URL`: optional Postgres DSN for LangGraph
  checkpoints; defaults to a driverless form derived from `DATABASE_URL`
- `REDIS_URL`: Redis connection used locally
- `CELERY_BROKER_URL`: broker URL for Celery; defaults to `REDIS_URL`
