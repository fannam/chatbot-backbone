"""add document failure reason

Revision ID: 0004_add_document_failure_reason
Revises: 0003_add_chunk_embeddings
Create Date: 2026-06-04 10:30:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_add_document_failure_reason"
down_revision = "0003_add_chunk_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("failure_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "failure_reason")
