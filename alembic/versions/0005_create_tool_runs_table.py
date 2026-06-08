"""create tool runs table

Revision ID: 0005_create_tool_runs_table
Revises: 0004_add_document_failure_reason
Create Date: 2026-06-05 08:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_create_tool_runs_table"
down_revision = "0004_add_document_failure_reason"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_runs",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("tool_call_id", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=False),
        sa.Column("output_payload", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'rejected', 'timed_out')",
            name="ck_tool_runs_status",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_tool_runs_conversation_id_id",
        "tool_runs",
        ["conversation_id", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_tool_runs_conversation_id_id", table_name="tool_runs")
    op.drop_table("tool_runs")
