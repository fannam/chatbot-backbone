from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1)
    metadata: dict[str, Any] | None = None
    stream: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty or whitespace")
        return value


class ChatMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCitationPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    snippet: str


class ChatToolRunPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    status: Literal["completed", "failed", "rejected", "timed_out"]
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None


class ChatUsagePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatCostPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    currency: Literal["USD"]


class ToolRunRecordPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    tool_call_id: str
    tool_name: str
    status: Literal["running", "completed", "failed", "rejected", "timed_out"]
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class ChatResponseMetadataPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    citations: list[ChatCitationPayload] = Field(default_factory=list)
    tool_runs: list[ChatToolRunPayload] = Field(default_factory=list)
    usage: ChatUsagePayload | None = None
    cost: ChatCostPayload | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    message: ChatMessage
    provider: str
    model: str
    metadata: ChatResponseMetadataPayload | None = None


class ChatStreamStartPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str


class ChatStreamDeltaPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: str


class ChatStreamToolStartPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]


class ChatStreamToolCompletePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: Literal["completed"]
    output: dict[str, Any]


class ChatStreamToolErrorPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    tool_call_id: str
    tool_name: str
    status: Literal["failed", "rejected", "timed_out"]
    error: str


class ChatStreamCompletePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    message: ChatMessage
    provider: str
    model: str
    metadata: ChatResponseMetadataPayload | None = None


class ChatStreamErrorPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    content_type: str
    byte_size: int
    status: Literal["processing", "ready", "failed"]
    chunk_count: int
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    content_type: str
    byte_size: int
    status: Literal["processing", "ready", "failed"]
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    failure_reason: str | None = None


class ConversationToolRunsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    tool_runs: list[ToolRunRecordPayload] = Field(default_factory=list)
