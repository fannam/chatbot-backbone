from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chatbot_api.models import Conversation, Message, utcnow


class ExpectedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str | None = None
    document_id: str | None = None
    chunk_indexes: list[int] = Field(default_factory=list)

    @field_validator("filename", "document_id")
    @classmethod
    def validate_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        if not normalized:
            raise ValueError("expected source identifiers must not be blank")
        return normalized

    @field_validator("chunk_indexes")
    @classmethod
    def validate_chunk_indexes(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("chunk indexes must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_source_selector(self) -> ExpectedSource:
        if self.filename is None and self.document_id is None:
            raise ValueError("expected source must include filename or document_id")
        return self


def safe_ratio(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def source_matches_reference(
    expected_source: ExpectedSource,
    *,
    filename: str,
    document_id: str,
) -> bool:
    if expected_source.filename is not None and expected_source.filename != filename:
        return False
    if expected_source.document_id is not None and expected_source.document_id != document_id:
        return False
    return True


def is_deep_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and is_deep_subset(expected_value, actual[key])
            for key, expected_value in expected.items()
        )

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(expected) > len(actual):
            return False
        return all(
            is_deep_subset(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )

    return expected == actual


def write_report(path: str | Path, report: BaseModel) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


class HistoryTurnLike(Protocol):
    role: str
    content: str


async def seed_history(
    session: Any,
    *,
    conversation_id: str,
    history: Sequence[HistoryTurnLike],
    owner_user_id: str | None = None,
) -> None:
    if not history:
        return

    timestamp = utcnow()
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        conversation = Conversation(
            id=conversation_id,
            owner_user_id=owner_user_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        session.add(conversation)
    else:
        conversation.updated_at = timestamp

    session.add_all(
        [
            Message(
                conversation_id=conversation_id,
                role=turn.role,
                content=turn.content,
                metadata_=None,
                created_at=timestamp,
            )
            for turn in history
        ]
    )
    await session.commit()
