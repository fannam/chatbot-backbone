from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from chatbot_api.auth import AuthenticatedUser
from chatbot_api.main import app, get_authenticated_user, get_chat_repository, get_memory_repository
from chatbot_api.models import utcnow
from chatbot_api.repositories import ConversationSummaryRecord, MemoryRecord, PersistedExchange


@dataclass
class StubChatRepository:
    existing_conversation_ids: set[str]

    async def list_messages(self, conversation_id: str, *, owner_user_id: str | None = None):
        del owner_user_id
        return []

    async def list_message_records(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ):
        del owner_user_id
        return []

    async def conversation_exists(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> bool:
        del owner_user_id
        return conversation_id in self.existing_conversation_ids

    async def list_tool_runs(
        self,
        conversation_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ):
        del owner_user_id
        return []

    async def create_tool_run(self, **kwargs):
        raise NotImplementedError

    async def complete_tool_run(self, **kwargs):
        raise NotImplementedError

    async def fail_tool_run(self, **kwargs):
        raise NotImplementedError

    async def append_exchange(self, **kwargs) -> PersistedExchange:
        return PersistedExchange(
            conversation_id=kwargs["conversation_id"],
            user_message_id=1,
            assistant_message_id=2,
            created_at=utcnow(),
        )


@dataclass
class StubMemoryRepository:
    summaries: dict[str, ConversationSummaryRecord] = field(default_factory=dict)
    memories_by_user: dict[str, list[MemoryRecord]] = field(default_factory=dict)

    async def get_conversation_summary(
        self,
        conversation_id: str,
        *,
        owner_user_id: str | None = None,
    ):
        del owner_user_id
        return self.summaries.get(conversation_id)

    async def upsert_conversation_summary(self, **kwargs):
        raise NotImplementedError

    async def list_active_memories(
        self,
        user_id: str,
        *,
        limit: int,
        owner_user_id: str | None = None,
    ):
        if owner_user_id is not None and owner_user_id != user_id:
            return []
        return list(self.memories_by_user.get(user_id, []))[:limit]

    async def upsert_memory(self, **kwargs):
        raise NotImplementedError

    async def delete_memory(
        self,
        *,
        user_id: str,
        memory_id: int,
        owner_user_id: str | None = None,
    ) -> bool:
        if owner_user_id is not None and owner_user_id != user_id:
            return False
        memories = self.memories_by_user.get(user_id, [])
        for index, memory in enumerate(memories):
            if memory.id != memory_id:
                continue
            del memories[index]
            return True
        return False


def build_chat_repository_override(repository: StubChatRepository):
    async def override():
        return repository

    return override


def build_memory_repository_override(repository: StubMemoryRepository):
    async def override():
        return repository

    return override


def build_authenticated_user_override(user: AuthenticatedUser | None):
    async def override() -> AuthenticatedUser | None:
        return user

    return override


@pytest.fixture
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_get_conversation_memory_returns_summary(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    now = utcnow()
    chat_repository = StubChatRepository(existing_conversation_ids={"conv-123"})
    memory_repository = StubMemoryRepository(
        summaries={
            "conv-123": ConversationSummaryRecord(
                conversation_id="conv-123",
                summary_text="User wants concise answers.",
                last_summarized_message_id=14,
                created_at=now,
                updated_at=now,
            )
        }
    )
    app.dependency_overrides[get_chat_repository] = build_chat_repository_override(
        chat_repository
    )
    app.dependency_overrides[get_memory_repository] = build_memory_repository_override(
        memory_repository
    )

    response = await async_client.get("/conversations/conv-123/memory")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "conversation_id": "conv-123",
        "summary": {
            "conversation_id": "conv-123",
            "summary_text": "User wants concise answers.",
            "last_summarized_message_id": 14,
            "updated_at": now.isoformat().replace("+00:00", "Z"),
        },
    }


@pytest.mark.anyio
async def test_get_conversation_memory_returns_not_found_for_missing_conversation(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    app.dependency_overrides[get_chat_repository] = build_chat_repository_override(
        StubChatRepository(existing_conversation_ids=set())
    )
    app.dependency_overrides[get_memory_repository] = build_memory_repository_override(
        StubMemoryRepository()
    )

    response = await async_client.get("/conversations/conv-missing/memory")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "conversation not found"}


@pytest.mark.anyio
async def test_list_user_memories_returns_active_items(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    now = utcnow()
    memory_repository = StubMemoryRepository(
        memories_by_user={
            "user-123": [
                MemoryRecord(
                    id=3,
                    user_id="user-123",
                    kind="preference",
                    key="preferences.language",
                    value_json={"value": "Vietnamese"},
                    confidence=0.92,
                    source_message_id=11,
                    extraction_method="rule",
                    created_at=now,
                    updated_at=now,
                    deleted_at=None,
                )
            ]
        }
    )
    app.dependency_overrides[get_memory_repository] = build_memory_repository_override(
        memory_repository
    )

    response = await async_client.get("/users/user-123/memories")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "user_id": "user-123",
        "memories": [
            {
                "id": 3,
                "user_id": "user-123",
                "kind": "preference",
                "key": "preferences.language",
                "value_json": {"value": "Vietnamese"},
                "confidence": 0.92,
                "source_message_id": 11,
                "extraction_method": "rule",
                "created_at": now.isoformat().replace("+00:00", "Z"),
                "updated_at": now.isoformat().replace("+00:00", "Z"),
            }
        ],
    }


@pytest.mark.anyio
async def test_delete_user_memory_returns_no_content_and_removes_item(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    now = utcnow()
    memory_repository = StubMemoryRepository(
        memories_by_user={
            "user-123": [
                MemoryRecord(
                    id=9,
                    user_id="user-123",
                    kind="profile",
                    key="profile.company",
                    value_json={"value": "Example"},
                    confidence=0.7,
                    source_message_id=21,
                    extraction_method="llm",
                    created_at=now,
                    updated_at=now,
                    deleted_at=None,
                )
            ]
        }
    )
    app.dependency_overrides[get_memory_repository] = build_memory_repository_override(
        memory_repository
    )

    response = await async_client.delete("/users/user-123/memories/9")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert memory_repository.memories_by_user["user-123"] == []


@pytest.mark.anyio
async def test_list_user_memories_forbids_access_to_other_user_when_authenticated(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    app.dependency_overrides[get_memory_repository] = build_memory_repository_override(
        StubMemoryRepository()
    )
    app.dependency_overrides[get_authenticated_user] = build_authenticated_user_override(
        AuthenticatedUser(
            user_id="user-self",
            display_name=None,
            email=None,
            plan=None,
            locale=None,
            preferences={},
        )
    )

    response = await async_client.get("/users/user-other/memories")

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json() == {"detail": "forbidden"}


@pytest.mark.anyio
async def test_get_conversation_memory_returns_not_found_for_other_users_conversation(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    app.dependency_overrides[get_chat_repository] = build_chat_repository_override(
        StubChatRepository(existing_conversation_ids=set())
    )
    app.dependency_overrides[get_memory_repository] = build_memory_repository_override(
        StubMemoryRepository()
    )
    app.dependency_overrides[get_authenticated_user] = build_authenticated_user_override(
        AuthenticatedUser(
            user_id="user-self",
            display_name=None,
            email=None,
            plan=None,
            locale=None,
            preferences={},
        )
    )

    response = await async_client.get("/conversations/conv-foreign/memory")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "conversation not found"}
