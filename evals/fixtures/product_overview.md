# Product Overview

The chatbot API exposes a FastAPI backend for realtime chat, document ingestion,
and retrieval-augmented generation. The local stack uses PostgreSQL for
persistence, Redis for background work, and a Celery worker for embedding jobs.

The system is designed so the API layer validates requests, the LangGraph
workflow orchestrates chat behavior, and retrieval injects conservative context
before the model generates an answer.
