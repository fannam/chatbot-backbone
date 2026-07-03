# Chatbot Project Status

## Trang Thai Hien Tai

Ngay cap nhat: 2026-06-10

Trang thai tong quan: Phase 0 den Phase 4 da co nen tang hoan chinh cho chat + document ingestion + embedding async + retrieval/tool loop; Phase 5 da co memory v1 local/dev voi regression gate rieng; Phase 7 local/dev observability + evaluation van on dinh; va Phase 6 da bat dau voi lat dau tien cho API key auth + user scoping. `POST /chat` van giu JSON sync response va SSE streaming qua `stream=true`, nhung neu `AUTH_ENABLED=true` thi cac endpoint stateful doc `X-API-Key`, server-side inject/overwrite reserved key `metadata.user_profile`, va workflow van di theo `load_context -> load_memory -> call_model -> execute_tools? -> call_model -> persist_response -> write_memory`. Model van duoc phep goi cac tool allowlist `calculator`, `search_knowledge_base`, va `get_current_user_profile`; `tool_runs` van duoc persist vao database; response JSON va SSE `message_complete` van co the tra ve `metadata.citations`, `metadata.tool_runs`, `metadata.usage`, va `metadata.cost` ma khong doi contract; SSE van giu `tool_start`, `tool_complete`, `tool_error`; va he thong da co rolling conversation summary + conservative long-term memory cho `user_id`. Retrieval va document status hien da co owner scoping theo authenticated user; debug endpoints `GET /conversations/{conversation_id}/memory`, `GET /conversations/{conversation_id}/tool-runs`, `GET /users/{user_id}/memories`, `DELETE /users/{user_id}/memories/{memory_id}`, va `GET /documents/{document_id}` deu ton trong user boundary khi auth bat. Repo van giu `python -m chatbot_api.rag_eval`, `python -m chatbot_api.chat_eval`, `python -m chatbot_api.memory_eval`, va `python -m chatbot_api.eval_suite` cho retrieval/chat-tool/memory local-dev gating; buoc tiep theo la verify auth-enabled dogfood/dev-staging, bo sung rate limiting/request size/audit logging, va tiep tuc memory tuning tren use case that.

Project location:

- Thu muc du kien: `/home/namph32/workspace/chatbot`
- Ghi chu: Yeu cau ban dau la `/workspace/chatbot`, nhung moi truong hien tai khong cho tao `/workspace`. Da dung duong dan co quyen ghi tuong ung trong workspace cua user.

## Dang O Buoc Nao

Roadmap phase hien tai: Phase 6 dang lam voi first cut cho API key auth + user scoping tren chat/documents/retrieval/memory debug surfaces, trong khi Phase 5 memory v1 van giu summary ngan han + long-term memory conservative va regression gate deterministic. Buoc tiep theo hop ly la verify auth-enabled dogfood/dev-staging, bo sung hardening cho rate limiting/request size/audit logging, va tune memory/retrieval dua tren failed cases that thay vi tiep tuc phu thuoc `metadata.user_profile.user_id` o runtime production.

Trang thai Phase 0:

- Da tao project structure theo `src` layout.
- Da tao `pyproject.toml` va `uv.lock`.
- Da tao FastAPI app trong `src/chatbot_api`.
- Da them health check endpoint `GET /health`.
- Da them test cho health endpoint.
- Da cau hinh Ruff va pytest.
- Da tao Dockerfile.
- Da tao `docker-compose.yml` voi API, PostgreSQL va Redis.
- Da tao `.env.example`.
- Da tao Alembic migration scaffold.
- Chua smoke-test duoc `docker compose` trong moi truong hien tai.

Trang thai Phase 1:

- Da them endpoint `POST /chat`.
- Da tao LLM provider abstraction voi OpenAI implementation dau tien.
- Da them config `OPENAI_API_KEY`, `OPENAI_MODEL`, `LLM_TIMEOUT_SECONDS`.
- Da them test cho happy path, validation error, provider timeout, provider failure, va misconfiguration.
- Da them conversation/message persistence bang SQLAlchemy.
- Da them Alembic revision dau tien cho chat data model.
- Da replay stored history vao provider khi caller gui lai `conversation_id`.
- Da them SSE streaming cho `POST /chat` qua `stream=true`.
- Da chot event schema toi thieu: `message_start`, `message_delta`, `message_complete`, `error`.
- Da giu backward compatibility cho JSON response khi `stream` khong duoc bat.
- Da giu persistence semantics an toan: chi commit exchange khi assistant stream hoan tat thanh cong.

Trang thai Phase 2:

- Da dinh nghia workflow state serializable de chay qua LangGraph va checkpoint duoc.
- Da tao cac node toi thieu `load_context`, `call_model`, `persist_response`.
- Da tach graph construction khoi FastAPI handler; handler chi con request/response orchestration.
- Da giu `ChatService` lam thin facade, con chat orchestration nam trong workflow.
- Da cho ca sync path va SSE streaming path chay qua cung mot workflow.
- Da them LangGraph checkpoint runtime voi Postgres saver khi cau hinh checkpoint DSN la PostgreSQL.
- Da fallback sang in-memory checkpointer cho moi truong test/non-PostgreSQL de giu testability.
- Da them unit test cho tung node va integration test graph end-to-end.

Trang thai Phase 3:

- Da them endpoint `POST /documents` dung `multipart/form-data`.
- Da them document ingestion service rieng cho validate file, extract text, chunk text va persist metadata/chunks.
- Da ho tro `txt`, `md` va `pdf` text-based cho MVP ingestion.
- Da them model `documents` va `document_chunks` cung Alembic migration tuong ung.
- Da them config `DOCUMENT_MAX_BYTES`, `DOCUMENT_CHUNK_SIZE_CHARS`, `DOCUMENT_CHUNK_OVERLAP_CHARS`.
- Da them unit/API/repository/migration tests cho document ingestion thin slice.
- Da them embedding boundary voi OpenAI embedding implementation dau tien.
- Da mo rong schema/migration voi `document_chunks.embedding`, bat `pgvector` tren PostgreSQL va fallback JSON cho SQLite tests.
- Da them retriever service va retrieval node vao LangGraph workflow chat.
- Da mo rong `POST /chat` JSON va SSE `message_complete` voi citation metadata khi retrieval co ket qua.
- Da them command backfill `python -m chatbot_api.reindex_embeddings` cho cac chunk cu chua co embedding.
- Da chuyen embedding/reindex sang background worker voi Celery + Redis.
- Da them endpoint `GET /documents/{document_id}` de poll trang thai ingestion.
- Da doi `POST /documents` sang async-first response voi `status=processing`.
- Da them retrieval guardrails theo huong conservative: `RETRIEVAL_MIN_SCORE`, `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT`, `RETRIEVAL_CANDIDATE_LIMIT`.
- Da loc chunk retrieval theo nguong score, dedupe theo document, va prompt tune de model chi dua vao source khi context thuc su ho tro cau tra loi.
- Da giu nguyen contract `/chat` va SSE; khi khong con candidate hop le sau filter thi response tro ve plain chat khong metadata.
- Da them test retrieval unit/API cho filter low-score, dedupe theo document, va prompt instructions moi.
- Da them offline retrieval eval baseline de do hit-rate/coverage truoc khi toi uu ANN, reranking hoac hybrid retrieval.
- Da them command `python -m chatbot_api.rag_eval` voi JSON report output tuy chon, active retrieval config, va failed-case summary.
- Da them checked-in eval dataset `evals/rag_retrieval_baseline.json` va fixture corpus mau trong `evals/fixtures/`.
- Da mo rong retrieval service voi internal path lay selected chunks cho eval ma khong doi HTTP API contract.

Trang thai Phase 4:

- Da dinh nghia tool interface noi bo voi schema input/output va provider-facing tool definitions.
- Da them 3 tool allowlist cho scope hien tai: `calculator`, `search_knowledge_base`, va `get_current_user_profile`.
- Da refactor OpenAI Responses adapter de ho tro tool calls + `previous_response_id` + `function_call_output`.
- Da thay `retrieve_context` node bang vong lap tool-calling trong LangGraph workflow.
- Da them persistence `tool_runs` vao database va Alembic migration tuong ung.
- Da mo rong `POST /chat` JSON va SSE `message_complete` voi `metadata.tool_runs`.
- Da them SSE event moi `tool_start`, `tool_complete`, `tool_error`.
- Da them config `TOOL_MAX_ROUNDS`, `TOOL_EXECUTION_TIMEOUT_SECONDS`, `TOOL_SEARCH_TOP_K`.
- Da them test cho tools, workflow tool loop, API tool trace, repository tool run lifecycle, va migration schema.
- Da them read-only query API `GET /conversations/{conversation_id}/tool-runs` voi `limit` de xem tool run lifecycle theo conversation.
- Da them repository query cho `conversation_exists` va `list_tool_runs`, schema response rieng cho tool run records, va test API/repository cho `404`, empty-list, ordering va limit.
- Da them request-scoped tool execution context de tool co the doc `conversation_id` va request metadata an toan.
- Da chot contract reserved `metadata.user_profile` cho profile lookup truoc khi co auth/user store o Phase 6.

Trang thai Phase 5:

- Da them model/migration `conversation_summaries` va `memories`.
- Da them config `MEMORY_ENABLED`, `MEMORY_RECENT_MESSAGE_WINDOW`, `MEMORY_SUMMARY_TRIGGER_MESSAGES`, `MEMORY_MAX_SUMMARY_CHARS`, `MEMORY_MAX_ACTIVE_ITEMS`, va `MEMORY_LONG_TERM_ENABLED`.
- Da them `MemoryManager` rieng de xu ly prompt injection, summary refresh, va long-term extraction.
- Da refactor workflow thanh `load_context -> load_memory -> call_model -> execute_tools? -> call_model -> persist_response -> write_memory`.
- Da giu backward compatibility cho `/chat`; memory khong lam doi JSON response shape hay SSE event schema.
- Da them rolling conversation summary de nen history cu va giu recent-window messages trong prompt.
- Da them long-term memory conservative theo `metadata.user_profile.user_id`; neu khong co `user_id` thi bo qua long-term write.
- Da them hybrid extraction cho long-term memory: rule-based cho `profile.preferred_name`, `preferences.language`, `preferences.timezone`, `preferences.response_style`; LLM-based co validation chat cho allowlist `profile.role`, `profile.company`, `profile.team`.
- Da them debug endpoints `GET /conversations/{conversation_id}/memory`, `GET /users/{user_id}/memories`, va `DELETE /users/{user_id}/memories/{memory_id}`.
- Da them test migration/repository/workflow/API cho memory v1 va xac nhan `uv run pytest tests/test_migrations.py tests/test_repositories.py tests/test_tool_runs_api.py tests/test_memory_api.py tests/test_workflow.py`, `uv run ruff check src/chatbot_api tests`, va `uv run python -m py_compile src/chatbot_api/*.py tests/*.py` deu pass.
- Auth/runtime ownership cho memory/doc debug surfaces da duoc bat dau o Phase 6, nhung memory eval tren corpus/use case that van chua chay o dev/staging.
- Da them memory-specific eval suite deterministic `python -m chatbot_api.memory_eval` voi checked-in dataset `evals/memory_regression.json` de cover summary injection, summary refresh, rule-based extraction, allowlisted LLM extraction, invalid payload, va missing-user-id path.
- Da mo rong `python -m chatbot_api.eval_suite` voi memory dataset/threshold va output `.artifacts/memory-eval-report.json`.
- Chua chay memory eval tren corpus/use case that o dev/staging.

Trang thai Phase 6:

- Da them model/migration `users`, `api_keys`, `conversations.owner_user_id`, va `documents.owner_user_id`.
- Da them config `AUTH_ENABLED` va dependency doc `X-API-Key`; khi auth bat, stateful endpoints se xac thuc API key truoc khi vao business flow.
- Da them script provision `python -m chatbot_api.create_api_key` de tao/update user va mint plaintext API key mot lan cho dev/staging.
- Da server-side inject/overwrite `metadata.user_profile` tu authenticated user; client khong con duoc tin de set `user_id` cho flow chat/tool/memory khi auth bat.
- Da mo rong `ChatService`/workflow/repository/tool context voi `owner_user_id` de scope conversation history, tool run lifecycle, document reads, va KB retrieval.
- Da scope `POST /documents`, `GET /documents/{document_id}`, `GET /conversations/{conversation_id}/tool-runs`, `GET /conversations/{conversation_id}/memory`, `GET /users/{user_id}/memories`, va `DELETE /users/{user_id}/memories/{memory_id}` theo authenticated user.
- Da giu backward compatibility local/dev: khi `AUTH_ENABLED=false`, `/chat` va cac surface hien tai van co the chay ma khong can header auth, va request metadata cu van tiep tuc hoat dong.
- Chua co rate limiting, request size middleware tong quat, audit logging cho sensitive actions, hay tenant scoping cho worker/admin flows; do van la phan tiep theo cua Phase 6.

Trang thai Phase 7:

- Da co structured logs, `X-Request-ID`, va `GET /metrics`.
- Da them provider-round metrics `llm_requests_total`, `llm_request_duration_seconds`, `llm_input_tokens_total`, `llm_output_tokens_total`, `llm_total_tokens_total`, va `llm_request_cost_usd_total`.
- Da mo rong provider/workflow de thu token usage qua tung model round va cong don usage/cost qua tool loop.
- Da mo rong `POST /chat` JSON va SSE `message_complete` voi `metadata.usage` va `metadata.cost`.
- Da them config pricing cho active chat model qua `OPENAI_MODEL_INPUT_PRICE_PER_1M_TOKENS` va `OPENAI_MODEL_OUTPUT_PRICE_PER_1M_TOKENS`.
- Da them trace sink abstraction voi `Noop` mac dinh va `LangSmith` backend tuy chon cho chat path.
- Da trace duoc root `chat.request`, workflow sync/stream, workflow nodes, provider span, tool execution, va retrieval trong cung request.
- Da them config `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`.
- Da co regression eval service-level cho end-to-end chat/tool path voi scripted provider + tool/retrieval/persistence that.
- Da them eval suite hop nhat `python -m chatbot_api.eval_suite` voi corpus preflight, report JSON tong hop, va exit code pass/fail cho local/dev gating.
- Da ghi report rieng cho `rag_eval`, `chat_eval`, `memory_eval`, va report tong hop trong `.artifacts/`.

## Vua Giai Quyet

Da hoan thanh:

- Bat dau Phase 6 voi lat dau tien cho auth/user store thay vi tiep tuc de `metadata.user_profile.user_id` la identity runtime production.
- Them module `src/chatbot_api/auth.py`, repository auth, va script `src/chatbot_api/create_api_key.py` cho API key lifecycle local/dev-staging.
- Them migration `0007_add_auth_and_owner_scoping` va mo rong models cho `users`, `api_keys`, conversation/document ownership.
- Refactor `src/chatbot_api/main.py`, `services.py`, `workflow.py`, `retrieval.py`, va `tools.py` de propagate `owner_user_id`, overwrite server-side `metadata.user_profile`, va filter documents/retrieval theo authenticated user.
- Them test API/repository/migration cho API key auth, metadata override, cross-user `404/403`, document ownership, va auth repository path.

- Tao `pyproject.toml` su dung `uv` va commit `uv.lock`.
- Tao package `src/chatbot_api` va app FastAPI voi endpoint `GET /health`.
- Tao test `tests/test_health.py` va xac nhan `pytest` pass.
- Cau hinh Ruff va xac nhan `ruff check .` pass.
- Tao `.gitignore`, `.dockerignore`, `README.md`, `.env.example`.
- Tao `Dockerfile` va `docker-compose.yml` cho API + PostgreSQL + Redis.
- Tao `alembic.ini` va thu muc `alembic/` de san cho migrations o phase sau.
- Them `POST /chat` voi request/response schema toi thieu.
- Them `ChatService` va provider boundary de tach API handler khoi OpenAI SDK.
- Them OpenAI adapter dung Responses API.
- Them test `tests/test_chat.py` va xac nhan `pytest`/`ruff` pass sau khi mo Phase 1.
- Them SQLAlchemy async runtime, model `conversations`/`messages`, va repository boundary cho chat persistence.
- Them Alembic migration dau tien cho chat schema.
- Luu user message + assistant message khi `POST /chat` thanh cong.
- Nap stored history va gui lai provider khi tiep tuc mot conversation hien co.
- Them test cho persistence behavior, repository behavior, va migration smoke test.
- Them SSE streaming cho `POST /chat` bang `EventSourceResponse` khi request co `stream=true`.
- Them OpenAI provider streaming path va ChatService streaming orchestration.
- Them test cho streaming happy path, history replay trong stream mode, timeout truoc khi stream bat dau, va loi giua stream khong persist partial exchange.
- Them dependency `langgraph` va `langgraph-checkpoint-postgres`.
- Them workflow module rieng cho chat orchestration bang LangGraph.
- Them runtime lazy-init cho compiled graph va checkpointer.
- Refactor `ChatService` de delegate sang LangGraph workflow thay vi tu orchestration truc tiep.
- Giu nguyen contract `POST /chat` cho ca JSON response va SSE event schema.
- Them test `tests/test_workflow.py` cho node-level behavior va graph end-to-end.
- Cap nhat `.env.example` va `README.md` voi config LangGraph checkpoint.
- Them `POST /documents` cho upload 1 file bang `multipart/form-data`.
- Them module document ingestion cho text extraction (`txt`/`md`/`pdf`) va deterministic chunking.
- Them SQLAlchemy model va Alembic migration cho `documents` va `document_chunks`.
- Them `SqlAlchemyDocumentRepository` va persistence transaction cho document metadata + chunks.
- Them dependencies `pypdf` va `python-multipart`.
- Them test cho upload `txt`/`md`/`pdf`, unsupported type, oversized file, empty text, chunking behavior, repository persistence va migration schema.
- Chay lai `uv run pytest` va `uv run ruff check .` thanh cong sau khi mo Phase 3 thin slice.
- Them dependency `pgvector` va migration `0003_add_chunk_embeddings` cho vector storage.
- Them OpenAI embedding provider, config embedding/retrieval, va synchronous chunk embedding luc upload document.
- Them retrieval service de embed current query, lay top-k chunk, va tao citation metadata on dinh.
- Refactor LangGraph workflow thanh `load_context -> retrieve_context -> call_model -> persist_response`.
- Them citation metadata vao `POST /chat` JSON response va SSE `message_complete` ma van giu backward compatibility khi khong co retrieval.
- Them reindex command cho chunk chua co embedding va test cho backfill path.
- Chay lai `uv sync --extra dev`, `uv run pytest`, va `uv run ruff check .` thanh cong sau khi mo retrieval MVP.
- Refactor document ingestion de chi validate/extract/chunk/persist trong request va tao document o `status=processing`.
- Them `failure_reason` vao model/migration `documents` va repository methods de doc/poll/update trang thai.
- Them `DocumentEmbeddingService`, Celery app/task, va queue adapter de embed chunks trong background voi retry/backoff.
- Them endpoint `GET /documents/{document_id}` va mo rong response schema cho `processing | ready | failed`.
- Doi `python -m chatbot_api.reindex_embeddings` thanh enqueue embedding jobs cho documents con thieu embeddings.
- Cap nhat `docker-compose.yml` voi service `worker`, cap nhat `.env.example`/`README.md` cho worker va flow ingestion async.
- Them test cho document status API, enqueue failure path, embedding service/task retry logic, reindex enqueue flow, repository status updates, va migration moi.
- Xac nhan lai `uv run --no-sync pytest` va `uv run --no-sync ruff check .` pass sau khi chuyen sang async embedding pipeline.
- Them module `src/chatbot_api/rag_eval.py` cho offline retrieval evaluation chay bang `python -m chatbot_api.rag_eval`.
- Them schema dataset eval ho tro `expected_sources` match theo `filename`, `document_id`, va `chunk_indexes`.
- Them report aggregate/per-case cho retrieval baseline, ghi duoc ra JSON file neu can.
- Them test `tests/test_rag_eval.py` cho dataset validation, metric logic, va integration path SQLite + stub embeddings.
- Cap nhat `README.md` voi huong dan upload fixture corpus va chay retrieval eval baseline.
- Chay lai `uv run pytest` va `uv run ruff check .` thanh cong sau khi them retrieval eval baseline.
- Them module `src/chatbot_api/tools.py` voi tool registry, safe calculator evaluator, va KB search tool dung retriever hien co.
- Mo rong `src/chatbot_api/providers.py` de tra ve tool call batch hoac final completion tu OpenAI Responses API.
- Refactor `src/chatbot_api/workflow.py` thanh workflow vong lap co `execute_tools` va gioi han `TOOL_MAX_ROUNDS`.
- Them model/migration `tool_runs` va repository methods de persist lifecycle `running/completed/failed/rejected/timed_out`.
- Mo rong schema/API `/chat` de tra `metadata.tool_runs` va stream `tool_start/tool_complete/tool_error`.
- Cap nhat `.env.example` va `README.md` cho config/tool event moi.
- Xac nhan `uv run pytest` va `uv run ruff check .` pass sau khi them Phase 4 MVP.
- Them endpoint `GET /conversations/{conversation_id}/tool-runs` de doc tool execution history theo conversation ma khong doi contract `/chat`.
- Them response schema co `id`, timestamps, va support day du status `running/completed/failed/rejected/timed_out`.
- Them test repository/API cho query tool runs va cap nhat `README.md` voi vi du endpoint moi.
- Them `TokenUsage`/`UsageCost` metadata va propagate usage qua provider boundary, workflow, JSON response, va SSE terminal event.
- Them estimate cost theo active OpenAI model pricing config, cong don qua cac provider rounds trong tool-calling workflow.
- Them provider-round metrics/log cho token usage, estimated cost, va latency; cap nhat `README.md` va `.env.example` cho config/response shape moi.
- Them test cho sync JSON, SSE metadata, workflow aggregation, pricing-disabled fallback, va observability metrics moi.
- Them module `src/chatbot_api/tracing.py` voi trace sink abstraction, `NoopTraceSink`, va `LangSmithTraceSink`.
- Mo rong dependency wiring de provider/retriever/tool registry/workflow/chat handler cung dung chung trace sink.
- Wrap OpenAI client bang LangSmith wrapper khi tracing duoc bat, va giu root trace bao phu ca sync path va SSE lifecycle.
- Them test `tests/test_tracing.py` cho sink selection, stream request trace completion, va workflow trace hierarchy tool/retrieval.
- Xac nhan `uv run pytest` (73 tests) va `uv run ruff check src/chatbot_api tests` pass sau khi them tracing.
- Them module `src/chatbot_api/eval_common.py` de dung chung source-matching helpers cho retrieval eval va chat/tool eval.
- Them module `src/chatbot_api/chat_eval.py` cho regression eval service-level qua `ChatService`/LangGraph/tool registry ma khong can di qua HTTP.
- Them `ScriptedEvalProvider`, dataset schema cho chat/tool eval, report summary per-case, va CLI `python -m chatbot_api.chat_eval`.
- Them helper seed history cho eval, matcher cho expected tool runs / citations / answer substrings, va JSON report writer.
- Them checked-in dataset `evals/chat_tool_regression.json` cover KB search, cross-document disambiguation, calculator, va rejected tool path.
- Them test `tests/test_chat_eval.py` cho dataset validation, provider sequencing, history seeding, va integration path ghi report.
- Xac nhan `uv run pytest` (78 tests), `uv run ruff check src/chatbot_api tests`, va `uv run python -m chatbot_api.chat_eval --help` deu pass sau khi them chat/tool regression eval.
- Them `ToolExecutionContext` cho workflow -> tool registry va them tool `get_current_user_profile` doc reserved key `metadata.user_profile`.
- Mo rong `chat_eval` dataset schema voi `request_metadata` per-case de profile tool di qua duoc regression path that.
- Cap nhat checked-in dataset `evals/chat_tool_regression.json` voi ca success/missing path cho profile tool.
- Them module `src/chatbot_api/eval_suite.py` de preflight corpus, chay `rag_eval` + `chat_eval`, ghi report tong hop, va tra exit code non-zero khi quality gate fail.
- Them test `tests/test_eval_suite.py` cho preflight failure, success path, va quality gate failure.
- Xac nhan `uv run pytest tests/test_tools.py tests/test_workflow.py tests/test_chat_eval.py tests/test_eval_suite.py tests/test_observability.py` pass sau khi them profile tool + eval suite.
- Them module `src/chatbot_api/memory.py` cho short-term summary + conservative long-term memory management.
- Them migration `0006_add_memory_tables` va mo rong persistence voi `conversation_summaries`, `memories`, `list_message_records`, va `append_exchange` tra ve persisted message ids.
- Refactor LangGraph workflow de bo sung `load_memory` va `write_memory` quanh flow chat hien co ma van giu contract `/chat`.
- Them debug endpoints memory trong FastAPI va cap nhat `.env.example` / `README.md` voi config + endpoint moi.
- Them hybrid extraction strategy cho memory: rule-based cho preferences ro rang va LLM JSON extraction cho mot allowlist profile keys.
- Them test `tests/test_memory_api.py` va mo rong `tests/test_workflow.py`, `tests/test_repositories.py`, `tests/test_migrations.py` cho memory v1.
- Xac nhan `uv run pytest tests/test_migrations.py tests/test_repositories.py tests/test_tool_runs_api.py tests/test_memory_api.py tests/test_workflow.py`, `uv run ruff check src/chatbot_api tests`, va `uv run python -m py_compile src/chatbot_api/*.py tests/*.py` pass sau khi them Phase 5 memory v1.
- Them module `src/chatbot_api/memory_eval.py` cho regression eval deterministic tren `ChatService` + `MemoryManager` ma van di qua workflow/persistence that.
- Them checked-in dataset `evals/memory_regression.json` cover prompt injection, summary refresh, rule-based extraction, allowlisted LLM extraction, invalid extraction payload, va missing-user-id path.
- Mo rong `src/chatbot_api/eval_suite.py` voi `memory_dataset`, `min_memory_pass_rate`, `memory_summary`, va output `.artifacts/memory-eval-report.json`.
- Them test `tests/test_memory_eval.py` va mo rong `tests/test_eval_suite.py` cho success/failure path cua memory eval + suite integration.
- Cap nhat `README.md` voi huong dan `python -m chatbot_api.memory_eval` va combined suite moi co memory gate.
- Xac nhan `uv run pytest tests/test_memory_eval.py tests/test_eval_suite.py` va `uv run ruff check src/chatbot_api/memory_eval.py src/chatbot_api/eval_suite.py tests/test_memory_eval.py tests/test_eval_suite.py` deu pass.

## Viec Se Giai Quyet Tiep Theo

Buoc tiep theo nen lam:

1. Chay `python -m chatbot_api.memory_eval` va `python -m chatbot_api.eval_suite` tren use case that o dev/staging de xem memory co lam lech response, summary, hoac long-term extraction hay khong.
2. Bat `AUTH_ENABLED=true` tren dev/staging, provision API keys that, va smoke-test chat/documents/memory/tool-runs trong dogfood flow that.
3. Bo sung rate limiting, request size limit tong quat, va audit logging cho sensitive auth/memory actions de day tiep Phase 6 security hardening.
4. Tinh chinh prompt/threshold cho summary refresh va allowlist extraction dua tren failed cases dogfood/staging thuc te.
5. Neu retrieval van la bottleneck tren corpus that, quay lai retrieval tuning voi ANN, reranking, hoac hybrid retrieval sau khi memory quality on dinh.
6. Neu moi truong co Docker, chay smoke test `docker compose up --build` de verify local stack day du ca API + worker + auth-enabled memory chat path.
7. Can nhac trace propagation sang document upload/worker path va memory write path neu can correlate ingestion, chat, retrieval, memory, va auth context trong cung mot he observability.

## Tien Do Theo Roadmap

| Phase | Ten phase | Trang thai |
| --- | --- | --- |
| Phase 0 | Project Bootstrap | Hoan thanh |
| Phase 1 | Chat API Toi Thieu | Hoan thanh |
| Phase 2 | LangGraph Workflow | Hoan thanh |
| Phase 3 | RAG Ingestion Va Retrieval | Hoan thanh |
| Phase 4 | Tools Va Structured Actions | Hoan thanh |
| Phase 5 | Memory | Dang lam |
| Phase 6 | Auth, Multi-Tenant Va Security | Dang lam |
| Phase 7 | Observability Va Evaluation | Hoan thanh |
| Phase 8 | Production Hardening | Chua bat dau |

## Quy Uoc Cap Nhat Status

Sau moi lan lam viec, cap nhat file nay voi cac muc:

- `Trang Thai Hien Tai`: tinh trang tong quan cua codebase.
- `Dang O Buoc Nao`: phase va task hien tai.
- `Vua Giai Quyet`: cac thay doi vua hoan thanh.
- `Viec Se Giai Quyet Tiep Theo`: task tiep theo co thu tu uu tien.
- `Tien Do Theo Roadmap`: cap nhat trang thai tung phase.

Trang thai hop le:

- `Chua bat dau`
- `Dang cho implement`
- `Dang lam`
- `Bi chan`
- `Hoan thanh`

## Rủi Ro / Ghi Chu

- Da chon OpenAI lam provider dau tien; neu can da provider, can mo rong provider registry o Phase 1/2.
- Neu dung pgvector, can PostgreSQL extension duoc enable trong migration.
- Memory dai han khong nen implement qua som; nen lam sau khi chat va RAG on dinh.
- Tool calling can allowlist va timeout ngay tu dau de tranh rui ro hanh vi ngoai y muon.
