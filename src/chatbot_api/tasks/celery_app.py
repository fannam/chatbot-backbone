from __future__ import annotations

from celery import Celery

from chatbot_api.settings import Settings, get_settings


def create_celery_app(settings: Settings | None = None) -> Celery:
    resolved_settings = settings or get_settings()
    app = Celery(
        "chatbot_api",
        broker=resolved_settings.celery_broker_url,
        include=["chatbot_api.tasks.embedding_jobs"],
    )
    app.conf.task_ignore_result = True
    return app


celery_app = create_celery_app()
