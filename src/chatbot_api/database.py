from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def create_database_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(
        database_url,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        expire_on_commit=False,
    )


@asynccontextmanager
async def session_scope(database_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Build a short-lived engine/session factory for one call, disposing the
    engine on exit. For per-request session lifecycles that persist across an
    app's lifetime, use create_database_engine/create_session_factory directly
    and manage disposal via the app's own lifespan instead.
    """
    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        yield session_factory
    finally:
        await engine.dispose()
