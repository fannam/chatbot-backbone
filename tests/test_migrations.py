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

    assert {
        "conversations",
        "messages",
        "documents",
        "document_chunks",
        "tool_runs",
        "conversation_summaries",
        "memories",
        "users",
        "api_keys",
    } <= set(
        inspector.get_table_names()
    )
    assert {column["name"] for column in inspector.get_columns("conversations")} >= {
        "id",
        "owner_user_id",
        "created_at",
        "updated_at",
    }
    assert "ix_conversations_owner_user_id_id" in {
        index["name"] for index in inspector.get_indexes("conversations")
    }
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
        "owner_user_id",
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
    assert {column["name"] for column in inspector.get_columns("conversation_summaries")} >= {
        "conversation_id",
        "summary_text",
        "last_summarized_message_id",
        "created_at",
        "updated_at",
    }
    assert "ix_conversation_summaries_last_summarized_message_id" in {
        index["name"] for index in inspector.get_indexes("conversation_summaries")
    }
    assert {column["name"] for column in inspector.get_columns("memories")} >= {
        "id",
        "user_id",
        "kind",
        "key",
        "value_json",
        "confidence",
        "source_message_id",
        "extraction_method",
        "created_at",
        "updated_at",
        "deleted_at",
    }
    assert {"ix_memories_user_id_id", "ix_memories_user_id_key"} <= {
        index["name"] for index in inspector.get_indexes("memories")
    }
    assert {column["name"] for column in inspector.get_columns("users")} >= {
        "id",
        "display_name",
        "email",
        "plan",
        "locale",
        "preferences_json",
        "created_at",
        "updated_at",
    }
    assert "ix_users_email" in {
        index["name"] for index in inspector.get_indexes("users")
    }
    assert {column["name"] for column in inspector.get_columns("api_keys")} >= {
        "id",
        "user_id",
        "name",
        "key_prefix",
        "key_hash",
        "created_at",
        "last_used_at",
        "revoked_at",
    }
    assert {
        "ix_api_keys_user_id_id",
        "ix_api_keys_key_hash",
        "ix_api_keys_key_prefix",
    } <= {
        index["name"] for index in inspector.get_indexes("api_keys")
    }
