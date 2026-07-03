"""add memory tables

Revision ID: 0006_add_memory_tables
Revises: 0005_create_tool_runs_table
Create Date: 2026-06-10 09:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_add_memory_tables"
down_revision = "0005_create_tool_runs_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_summaries",
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("last_summarized_message_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversation_summaries_last_summarized_message_id",
        "conversation_summaries",
        ["last_summarized_message_id"],
        unique=False,
    )

    op.create_table(
        "memories",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "source_message_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column("extraction_method", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('profile', 'preference')",
            name="ck_memories_kind",
        ),
        sa.CheckConstraint(
            "extraction_method IN ('rule', 'llm')",
            name="ck_memories_extraction_method",
        ),
    )
    op.create_index("ix_memories_user_id_id", "memories", ["user_id", "id"], unique=False)
    op.create_index("ix_memories_user_id_key", "memories", ["user_id", "key"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_memories_user_id_key", table_name="memories")
    op.drop_index("ix_memories_user_id_id", table_name="memories")
    op.drop_table("memories")
    op.drop_index(
        "ix_conversation_summaries_last_summarized_message_id",
        table_name="conversation_summaries",
    )
    op.drop_table("conversation_summaries")
