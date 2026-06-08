from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
