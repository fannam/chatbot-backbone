from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config

from alembic import command


def test_alembic_upgrade_creates_chat_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migrations.db"
    config = Config("alembic.ini")
    config.set_main_option("script_location", str(Path("alembic").resolve()))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    command.upgrade(config, "head")

    engine = sa.create_engine(f"sqlite:///{database_path}")
    inspector = sa.inspect(engine)

    assert {"conversations", "messages", "documents", "document_chunks", "tool_runs"} <= set(
        inspector.get_table_names()
    )
    assert {column["name"] for column in inspector.get_columns("messages")} >= {
        "id",
        "conversation_id",
        "role",
        "content",
        "metadata",
        "created_at",
    }
    assert "ix_messages_conversation_id_id" in {
        index["name"] for index in inspector.get_indexes("messages")
    }
    assert {column["name"] for column in inspector.get_columns("documents")} >= {
        "id",
        "filename",
        "content_type",
        "byte_size",
        "checksum_sha256",
        "status",
        "failure_reason",
        "created_at",
        "updated_at",
    }
    assert {column["name"] for column in inspector.get_columns("document_chunks")} >= {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "embedding",
        "start_offset",
        "end_offset",
        "metadata",
        "created_at",
    }
    assert "ix_document_chunks_document_id_chunk_index" in {
        index["name"] for index in inspector.get_indexes("document_chunks")
    }
    assert {column["name"] for column in inspector.get_columns("tool_runs")} >= {
        "id",
        "conversation_id",
        "tool_call_id",
        "tool_name",
        "status",
        "input_payload",
        "output_payload",
        "error_message",
        "started_at",
        "completed_at",
    }
    assert "ix_tool_runs_conversation_id_id" in {
        index["name"] for index in inspector.get_indexes("tool_runs")
    }
