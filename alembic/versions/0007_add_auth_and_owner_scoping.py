"""add auth and owner scoping

Revision ID: 0007_add_auth_and_owner_scoping
Revises: 0006_add_memory_tables
Create Date: 2026-06-10 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_add_auth_and_owner_scoping"
down_revision = "0006_add_memory_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.create_table(
        "users",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("plan", sa.Text(), nullable=True),
        sa.Column("locale", sa.Text(), nullable=True),
        sa.Column(
            "preferences_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_api_keys_user_id_id", "api_keys", ["user_id", "id"], unique=False)
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"], unique=False)

    op.add_column("conversations", sa.Column("owner_user_id", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("owner_user_id", sa.Text(), nullable=True))
    op.create_index(
        "ix_conversations_owner_user_id_id",
        "conversations",
        ["owner_user_id", "id"],
        unique=False,
    )
    op.create_index(
        "ix_documents_owner_user_id_id",
        "documents",
        ["owner_user_id", "id"],
        unique=False,
    )

    if not is_sqlite:
        op.create_foreign_key(
            "fk_conversations_owner_user_id_users",
            "conversations",
            "users",
            ["owner_user_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.create_foreign_key(
            "fk_documents_owner_user_id_users",
            "documents",
            "users",
            ["owner_user_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if not is_sqlite:
        op.drop_constraint(
            "fk_documents_owner_user_id_users",
            "documents",
            type_="foreignkey",
        )
        op.drop_constraint(
            "fk_conversations_owner_user_id_users",
            "conversations",
            type_="foreignkey",
        )

    op.drop_index("ix_documents_owner_user_id_id", table_name="documents")
    op.drop_index("ix_conversations_owner_user_id_id", table_name="conversations")
    op.drop_column("documents", "owner_user_id")
    op.drop_column("conversations", "owner_user_id")

    op.drop_index("ix_api_keys_key_prefix", table_name="api_keys")
    op.drop_index("ix_api_keys_key_hash", table_name="api_keys")
    op.drop_index("ix_api_keys_user_id_id", table_name="api_keys")
    op.drop_table("api_keys")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
