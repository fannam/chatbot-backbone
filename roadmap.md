# Chatbot Project Roadmap

## 1. Muc tieu du an

Xay dung backend chatbot bang Python co kha nang:

- Chat realtime voi streaming response.
- RAG tren tai lieu nguoi dung hoac knowledge base noi bo.
- Tool calling cho cac tac vu co cau truc.
- Memory ngan han cho conversation state.
- Memory dai han cho user facts, preferences va summaries.
- Observability de debug agent, prompt, retrieval va chi phi LLM.
- Kien truc de mo rong sang frontend web, worker ingestion va production deployment.

## 2. Tech Stack De Xuat

### Backend API

- Python 3.12+
- FastAPI
- Pydantic v2
- Uvicorn/Gunicorn
- Server-Sent Events hoac WebSocket cho streaming

### Agent va LLM Orchestration

- LangGraph lam workflow/state machine chinh
- LangChain cho integrations voi LLM, embeddings, retrievers, tools
- LangMem cho long-term memory khi da co use case ro rang
- LangSmith cho tracing/debug LLM workflow

### Storage

- PostgreSQL lam database chinh
- pgvector cho vector search MVP
- Redis cho cache, session, rate limit va queue lightweight
- Object storage local/S3-compatible cho file uploads

### Background Jobs

- Celery, RQ hoac Dramatiq cho ingestion, chunking, embedding va indexing
- Worker rieng voi retry va dead-letter strategy

### Quality va DevOps

- Docker Compose cho local development
- Alembic cho database migrations
- pytest cho test
- Ruff cho lint/format
- MyPy hoac Pyright cho static typing neu can chat hon
- GitHub Actions cho CI

## 3. Kien Truc Tong Quan

```
Client
  |
  | HTTP/SSE/WebSocket
  v
FastAPI
  |
  | request validation, auth, rate limit
  v
LangGraph Chat Workflow
  |
  |-- intent/router node
  |-- memory load node
  |-- retrieval node
  |-- tool execution node
  |-- LLM response node
  |-- memory write node
  v
Streaming Response
```

Du lieu nen tach thanh cac nhom:

- `users`: thong tin user va auth identity.
- `conversations`: metadata cua hoi thoai.
- `messages`: message history va trace references.
- `documents`: metadata tai lieu.
- `document_chunks`: chunks, embeddings va retrieval metadata.
- `memories`: long-term memory co source, confidence va timestamps.
- `tool_runs`: lich su tool execution.

## 4. Phase 0: Project Bootstrap

Muc tieu: Tao skeleton backend co the chay local va co nen tang test/lint.

Checklist:

- Tao repo/project structure.
- Tao `pyproject.toml`.
- Cau hinh FastAPI app.
- Them health check endpoint `GET /health`.
- Them Dockerfile.
- Them `docker-compose.yml` voi API, PostgreSQL, Redis.
- Cau hinh `.env.example`.
- Cau hinh Ruff va pytest.
- Tao script dev run.

Ket qua mong doi:

- Chay duoc API local.
- Test health check pass.
- Docker Compose start duoc cac service can thiet.

## 5. Phase 1: Chat API Toi Thieu

Muc tieu: Co endpoint chat co the goi LLM va stream response.

Checklist:

- Tao LLM provider abstraction.
- Cau hinh provider qua environment variables.
- Them endpoint `POST /chat`.
- Ho tro request body gom `conversation_id`, `message`, `metadata`.
- Luu user message va assistant message vao database.
- Ho tro streaming bang SSE.
- Them error handling cho timeout, provider error va invalid input.
- Them basic tests cho API schema va happy path.

Ket qua mong doi:

- User gui message va nhan response streaming.
- Conversation history duoc persist.
- Provider LLM co the thay doi ma khong sua business logic.

## 6. Phase 2: LangGraph Workflow

Muc tieu: Chuyen chat logic sang workflow ro rang, test duoc tung node.

Checklist:

- Dinh nghia `ChatState`.
- Tao cac node toi thieu:
  - `load_context`
  - `call_model`
  - `persist_response`
- Them checkpoint cho conversation state.
- Tach graph construction khoi API handler.
- Them unit tests cho tung node.
- Them integration test cho graph end-to-end.

Ket qua mong doi:

- API handler chi orchestration request/response.
- Chat behavior nam trong LangGraph workflow.
- Co nen tang de them retrieval, tools va memory.

## 7. Phase 3: RAG Ingestion Va Retrieval

Muc tieu: Chatbot tra loi dua tren knowledge base.

Checklist:

- Tao document upload endpoint.
- Tao pipeline ingestion:
  - validate file
  - extract text
  - chunk text
  - generate embeddings
  - save chunks vao pgvector
- Tao retriever service.
- Them retrieval node vao LangGraph.
- Them prompt template co context va citation/source metadata.
- Them tests cho chunking va retrieval.
- Them re-index command/script.

Ket qua mong doi:

- Upload tai lieu.
- Chatbot retrieve duoc context lien quan.
- Response co the tra ve source references.

## 8. Phase 4: Tools Va Structured Actions

Muc tieu: Cho phep chatbot goi tools an toan va quan sat duoc.

Checklist:

- Dinh nghia tool interface noi bo.
- Tao tools mau:
  - calculator
  - search knowledge base
  - get current user profile
- Them tool routing node.
- Luu `tool_runs` vao database.
- Them timeout va allowlist cho tools.
- Them validation input/output bang Pydantic.
- Them tests cho tool execution va failure path.

Ket qua mong doi:

- Model co the goi tool khi can.
- Tool execution co log, trace va error handling.
- Khong cho phep tool tuy tien ngoai allowlist.

## 9. Phase 5: Memory

Muc tieu: Them memory ngan han va dai han ma khong lam chatbot bi nhiem thong tin sai.

Checklist:

- Memory ngan han:
  - load recent messages
  - summarize conversation khi qua dai
  - checkpoint state theo conversation
- Memory dai han:
  - xac dinh memory schema
  - trich xuat facts/preferences co confidence
  - luu source message id
  - them co che update/delete memory
- Tich hop LangMem neu can extraction va management layer san co.
- Them memory load/write nodes vao LangGraph.
- Them guardrails de khong luu thong tin nhay cam neu khong can.

Ket qua mong doi:

- Chatbot nho ngu canh gan day.
- Chatbot co the ghi nho preferences/facts dai han co kiem soat.
- Memory co trace nguon va co the xoa/sua.

## 10. Phase 6: Auth, Multi-Tenant Va Security

Muc tieu: San sang cho nhieu user va moi truong production.

Checklist:

- Them auth middleware.
- Them user scoping cho conversations, documents va memories.
- Them rate limiting.
- Them request size limit.
- Them audit logging cho sensitive actions.
- Them input sanitization cho document ingestion.
- Them secret management strategy.
- Them CORS config ro rang.

Ket qua mong doi:

- Du lieu user duoc isolate.
- API co gioi han tai nguyen.
- Cac tac vu nhay cam co audit trail.

## 11. Phase 7: Observability Va Evaluation

Muc tieu: Debug duoc chat quality, latency, cost va retrieval accuracy.

Checklist:

- Tich hop LangSmith tracing.
- Them structured logs.
- Them metrics:
  - latency
  - token usage
  - retrieval hit count
  - tool call count
  - error rate
- Tao evaluation dataset nho.
- Them regression eval cho prompts/RAG.
- Them dashboard hoac report co ban.

Ket qua mong doi:

- Co the debug tung request.
- Co baseline chat quality.
- Co canh bao khi latency/error tang.

## 12. Phase 8: Production Hardening

Muc tieu: Dong goi va van hanh on dinh.

Checklist:

- Tach config theo environment.
- Them database migration workflow.
- Them backup/restore strategy.
- Them retry policy cho provider va jobs.
- Them graceful shutdown.
- Them load test co ban.
- Them CI pipeline:
  - lint
  - test
  - type check neu bat
  - build image
- Them deployment manifests neu can:
  - Docker Compose production
  - Kubernetes
  - Fly.io/Render/Railway

Ket qua mong doi:

- Co the deploy repeatable.
- Co migration va rollback strategy.
- He thong chiu duoc loi provider/job tam thoi.

## 13. Thu Tu Uu Tien De Implement

1. Bootstrap FastAPI + Docker Compose.
2. Chat API streaming toi thieu.
3. LangGraph workflow.
4. PostgreSQL persistence.
5. RAG ingestion + retrieval.
6. Tool calling.
7. Memory ngan han.
8. Memory dai han voi LangMem.
9. Auth va multi-tenant.
10. Observability/evaluation.
11. Production hardening.

## 14. Nguyen Tac Thiet Ke

- Khong de API handler chua business logic phuc tap.
- Khong hard-code LLM provider vao graph.
- Moi tool phai co schema, timeout va log.
- Memory dai han phai co source va co the xoa.
- RAG response nen gan source metadata de debug.
- Prompt va graph changes can co evaluation test.
- Bat dau don gian, chi them multi-agent khi workflow thuc su can.

## 15. Definition Of Done Cho MVP

MVP duoc xem la hoan thanh khi:

- API chay local bang Docker Compose.
- Co endpoint health check.
- Co endpoint chat streaming.
- Conversation history duoc luu.
- LangGraph dieu phoi chat workflow.
- Upload va index duoc tai lieu text/PDF co ban.
- Retrieval duoc tich hop vao response.
- Co test cho core paths.
- Co tracing/logging du de debug request loi.

