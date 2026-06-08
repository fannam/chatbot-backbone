from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from chatbot_api.main import app, get_chat_repository
from chatbot_api.repositories import ToolRunRecord


@dataclass
class StubToolRunRepository:
    existing_conversation_ids: set[str]
    tool_runs_by_conversation: dict[str, list[ToolRunRecord]]

    async def list_messages(self, conversation_id: str):
        return []

    async def conversation_exists(self, conversation_id: str) -> bool:
        return conversation_id in self.existing_conversation_ids

    async def list_tool_runs(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[ToolRunRecord]:
        return list(self.tool_runs_by_conversation.get(conversation_id, []))[:limit]

    async def create_tool_run(self, **kwargs):
        raise NotImplementedError

    async def complete_tool_run(self, **kwargs):
        raise NotImplementedError

    async def fail_tool_run(self, **kwargs):
        raise NotImplementedError

    async def append_exchange(self, **kwargs) -> None:
        raise NotImplementedError


def build_chat_repository_override(repository: StubToolRunRepository):
    async def override() -> StubToolRunRepository:
        return repository

    return override


def make_tool_run(
    *,
    id: int,
    conversation_id: str,
    tool_call_id: str,
    tool_name: str,
    status: str,
    started_at: datetime,
    completed_at: datetime | None,
    input_payload: dict,
    output_payload: dict | None = None,
    error_message: str | None = None,
) -> ToolRunRecord:
    return ToolRunRecord(
        id=id,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        status=status,
        input_payload=input_payload,
        output_payload=output_payload,
        error_message=error_message,
        started_at=started_at,
        completed_at=completed_at,
    )


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
async def test_list_tool_runs_returns_records_for_conversation(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    now = datetime.now(UTC)
    repository = StubToolRunRepository(
        existing_conversation_ids={"conv-123"},
        tool_runs_by_conversation={
            "conv-123": [
                make_tool_run(
                    id=2,
                    conversation_id="conv-123",
                    tool_call_id="tool-2",
                    tool_name="search_knowledge_base",
                    status="failed",
                    started_at=now + timedelta(seconds=5),
                    completed_at=now + timedelta(seconds=7),
                    input_payload={"query": "guide"},
                    error_message="search failed",
                ),
                make_tool_run(
                    id=1,
                    conversation_id="conv-123",
                    tool_call_id="tool-1",
                    tool_name="calculator",
                    status="completed",
                    started_at=now,
                    completed_at=now + timedelta(seconds=1),
                    input_payload={"expression": "2 + 2"},
                    output_payload={"result": 4},
                ),
            ]
        },
    )
    app.dependency_overrides[get_chat_repository] = build_chat_repository_override(
        repository
    )

    response = await async_client.get("/conversations/conv-123/tool-runs?limit=1")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "conversation_id": "conv-123",
        "tool_runs": [
            {
                "id": 2,
                "tool_call_id": "tool-2",
                "tool_name": "search_knowledge_base",
                "status": "failed",
                "input": {"query": "guide"},
                "error": "search failed",
                "started_at": (now + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
                "completed_at": (now + timedelta(seconds=7)).isoformat().replace("+00:00", "Z"),
            }
        ],
    }


@pytest.mark.anyio
async def test_list_tool_runs_returns_empty_list_for_existing_conversation_without_runs(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = StubToolRunRepository(
        existing_conversation_ids={"conv-empty"},
        tool_runs_by_conversation={},
    )
    app.dependency_overrides[get_chat_repository] = build_chat_repository_override(
        repository
    )

    response = await async_client.get("/conversations/conv-empty/tool-runs")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "conversation_id": "conv-empty",
        "tool_runs": [],
    }


@pytest.mark.anyio
async def test_list_tool_runs_returns_not_found_for_missing_conversation(
    async_client: AsyncClient,
    clear_dependency_overrides: None,
) -> None:
    repository = StubToolRunRepository(
        existing_conversation_ids=set(),
        tool_runs_by_conversation={},
    )
    app.dependency_overrides[get_chat_repository] = build_chat_repository_override(
        repository
    )

    response = await async_client.get("/conversations/conv-missing/tool-runs")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "conversation not found"}
