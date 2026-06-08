"""add chunk embeddings

Revision ID: 0003_add_chunk_embeddings
Revises: 0002_create_document_tables
Create Date: 2026-06-04 09:30:00
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_add_chunk_embeddings"
down_revision = "0002_create_document_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dimensions = int(os.getenv("DOCUMENT_EMBEDDING_DIMENSIONS", "1536"))

    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.add_column(
            "document_chunks",
            sa.Column("embedding", Vector(dimensions), nullable=True),
        )
        return

    op.add_column(
        "document_chunks",
        sa.Column("embedding", sa.JSON(none_as_null=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "embedding")
