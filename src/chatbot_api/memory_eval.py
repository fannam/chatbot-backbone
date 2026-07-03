from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.eval_common import is_deep_subset, safe_ratio, seed_history, write_report
from chatbot_api.memory import MemoryManager, extract_user_id
from chatbot_api.providers import ChatCompletion, ChatProvider, ChatTurn, ToolCallBatch
from chatbot_api.repositories import (
    ConversationSummaryRecord,
    MemoryRecord,
    SqlAlchemyChatRepository,
    SqlAlchemyMemoryRepository,
)
from chatbot_api.services import ChatService
from chatbot_api.settings import Settings, get_settings
from chatbot_api.tracing import NoopTraceSink

DEFAULT_DATASET_PATH = Path("evals/memory_regression.json")


class MemoryEvalHistoryTurn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("history content must not be blank")
        return normalized


class SeededConversationSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary_text: str
    last_summarized_message_id: int = Field(ge=1)

    @field_validator("summary_text")
    @classmethod
    def validate_summary_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("summary_text must not be blank")
        return normalized


class SeededActiveMemory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["profile", "preference"]
    key: str
    value_json: dict[str, Any]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_method: Literal["rule", "llm"] = "rule"

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory key must not be blank")
        return normalized


class MemoryEvalSettingsOverride(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_recent_message_window: int | None = Field(default=None, ge=1)
    memory_summary_trigger_messages: int | None = Field(default=None, ge=2)
    memory_max_summary_chars: int | None = Field(default=None, ge=32)
    memory_max_active_items: int | None = Field(default=None, ge=1)
    memory_long_term_enabled: bool | None = None


class MemoryEvalStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["chat", "summary", "memory_extraction"]
    response_id: str
    content: str
    expected_prompt_substrings: list[str] = Field(default_factory=list)
    forbidden_prompt_substrings: list[str] = Field(default_factory=list)

    @field_validator("response_id", "content")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("step fields must not be blank")
        return normalized

    @field_validator("expected_prompt_substrings", "forbidden_prompt_substrings")
    @classmethod
    def validate_string_list(cls, values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("prompt expectations must not contain blank strings")
            normalized_values.append(normalized)
        return normalized_values


class ExpectedPersistedSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_text: str | None = None
    text_substrings: list[str] = Field(default_factory=list)
    last_summarized_message_id: int | None = Field(default=None, ge=1)

    @field_validator("exact_text")
    @classmethod
    def validate_exact_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("exact_text must not be blank")
        return normalized

    @field_validator("text_substrings")
    @classmethod
    def validate_substrings(cls, values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("summary substrings must not be blank")
            normalized_values.append(normalized)
        return normalized_values


class ExpectedPersistedMemory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    kind: Literal["profile", "preference"] | None = None
    extraction_method: Literal["rule", "llm"] | None = None
    value_subset: dict[str, Any] | None = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("expected memory key must not be blank")
        return normalized


class MemoryEvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    message: str
    request_metadata: dict[str, Any] | None = None
    history: list[MemoryEvalHistoryTurn] = Field(default_factory=list)
    seeded_summary: SeededConversationSummary | None = None
    seeded_memories: list[SeededActiveMemory] = Field(default_factory=list)
    settings_overrides: MemoryEvalSettingsOverride | None = None
    script: list[MemoryEvalStep] = Field(min_length=1)
    expected_answer_substrings: list[str] = Field(default_factory=list)
    forbidden_answer_substrings: list[str] = Field(default_factory=list)
    expect_summary_present: bool | None = None
    expected_summary: ExpectedPersistedSummary | None = None
    expected_active_memory_count: int | None = Field(default=None, ge=0)
    expected_active_memories: list[ExpectedPersistedMemory] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("id", "message")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case fields must not be blank")
        return normalized

    @field_validator("expected_answer_substrings", "forbidden_answer_substrings")
    @classmethod
    def validate_answer_substrings(cls, values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("answer expectations must not contain blank strings")
            normalized_values.append(normalized)
        return normalized_values

    @field_validator("notes")
    @classmethod
    def validate_optional_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_case(self) -> MemoryEvalCase:
        if self.script[0].kind != "chat":
            raise ValueError("the first script step must be chat")

        seen_response_ids: set[str] = set()
        previous_order = -1
        kind_order = {"chat": 0, "summary": 1, "memory_extraction": 2}
        seen_kinds: set[str] = set()

        for step in self.script:
            if step.response_id in seen_response_ids:
                raise ValueError(f"duplicate script response_id: {step.response_id}")
            seen_response_ids.add(step.response_id)

            if step.kind in seen_kinds:
                raise ValueError(f"duplicate script step kind: {step.kind}")
            seen_kinds.add(step.kind)

            current_order = kind_order[step.kind]
            if current_order < previous_order:
                raise ValueError("script steps must follow chat -> summary -> memory_extraction")
            previous_order = current_order

        return self


class MemoryEvalDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[MemoryEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> MemoryEvalDataset:
        seen_case_ids: set[str] = set()
        for case in self.cases:
            if case.id in seen_case_ids:
                raise ValueError(f"duplicate memory eval case id: {case.id}")
            seen_case_ids.add(case.id)
        return self


class MemoryEvalProviderCallMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class MemoryEvalProviderCallReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["chat", "summary", "memory_extraction"]
    messages: list[MemoryEvalProviderCallMessage]


class MemoryEvalSummaryReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    summary_text: str
    last_summarized_message_id: int
    updated_at: datetime


class MemoryEvalRecordReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    user_id: str
    kind: str
    key: str
    value_json: dict[str, Any]
    confidence: float
    source_message_id: int
    extraction_method: str
    created_at: datetime
    updated_at: datetime


class MemoryEvalCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    passed: bool
    failure_reasons: list[str]
    assistant_message: str | None
    provider_calls: list[MemoryEvalProviderCallReport]
    summary: MemoryEvalSummaryReport | None
    active_memories: list[MemoryEvalRecordReport]
    prompt_match: bool
    summary_match: bool
    memory_match: bool
    answer_match: bool


class MemoryEvalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cases: int
    passed_cases: int
    pass_rate: float
    failed_case_ids: list[str]
    prompt_expectation_case_count: int
    prompt_match_rate: float
    summary_expectation_case_count: int
    summary_match_rate: float
    memory_expectation_case_count: int
    memory_match_rate: float
    answer_expectation_case_count: int
    answer_match_rate: float


class MemoryEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    dataset_path: str
    summary: MemoryEvalSummary
    cases: list[MemoryEvalCaseReport]


class ScriptedMemoryEvalProviderError(RuntimeError):
    """Raised when the scripted memory eval provider sees unexpected execution."""


class ProviderCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["chat", "summary", "memory_extraction"]
    messages: list[ChatTurn]


class ScriptedMemoryEvalProvider(ChatProvider):
    provider_name = "scripted-memory-eval"

    def __init__(self, steps: Sequence[MemoryEvalStep]) -> None:
        self._steps = list(steps)
        self.calls: list[ProviderCallRecord] = []

    async def generate_response(
        self,
        messages: Sequence[ChatTurn],
        *,
        tools: Sequence[Any] = (),
        previous_response_id: str | None = None,
        tool_outputs: Sequence[Any] = (),
    ) -> ChatCompletion | ToolCallBatch:
        if tools:
            raise ScriptedMemoryEvalProviderError("memory eval does not expect tool definitions")
        if previous_response_id is not None:
            raise ScriptedMemoryEvalProviderError(
                "memory eval does not expect previous_response_id"
            )
        if tool_outputs:
            raise ScriptedMemoryEvalProviderError("memory eval does not expect tool outputs")
        if not self._steps:
            raise ScriptedMemoryEvalProviderError(
                "provider called more times than scripted memory steps"
            )

        step = self._steps.pop(0)
        self.calls.append(ProviderCallRecord(kind=step.kind, messages=list(messages)))
        return ChatCompletion(
            content=step.content,
            provider=self.provider_name,
            model="gpt-4.1-mini",
            response_id=step.response_id,
        )

    def assert_exhausted(self) -> None:
        if self._steps:
            raise ScriptedMemoryEvalProviderError(
                f"{len(self._steps)} scripted memory step(s) were not consumed"
            )


def load_memory_eval_dataset(path: str | Path) -> MemoryEvalDataset:
    dataset_path = Path(path)
    payload = dataset_path.read_text(encoding="utf-8")
    try:
        parsed = MemoryEvalDataset.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid memory eval dataset: {exc}") from exc
    return parsed


def apply_memory_settings_overrides(
    settings: Settings,
    overrides: MemoryEvalSettingsOverride | None,
) -> Settings:
    if overrides is None:
        return settings

    update_payload = {
        key: value
        for key, value in overrides.model_dump().items()
        if value is not None
    }
    return settings.model_copy(update=update_payload)


async def evaluate_memory_dataset(
    cases: Sequence[MemoryEvalCase],
    *,
    settings: Settings,
) -> list[MemoryEvalCaseReport]:
    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    reports: list[MemoryEvalCaseReport] = []

    try:
        for case in cases:
            reports.append(
                await evaluate_memory_case(
                    case,
                    session_factory=session_factory,
                    settings=settings,
                )
            )
    finally:
        await engine.dispose()

    return reports


async def evaluate_memory_case(
    case: MemoryEvalCase,
    *,
    session_factory,
    settings: Settings,
) -> MemoryEvalCaseReport:
    conversation_id = f"memory-eval-{case.id}-{uuid4()}"
    case_settings = apply_memory_settings_overrides(settings, case.settings_overrides)

    async with session_factory() as session:
        chat_repository = SqlAlchemyChatRepository(session)
        memory_repository = SqlAlchemyMemoryRepository(session)
        provider = ScriptedMemoryEvalProvider(case.script)
        memory_manager = MemoryManager(
            provider=provider,
            chat_repository=chat_repository,
            memory_repository=memory_repository,
            settings=case_settings,
            trace_sink=NoopTraceSink(),
        )
        service = ChatService(
            provider,
            chat_repository,
            memory_manager=memory_manager,
            trace_sink=NoopTraceSink(),
        )

        user_id = extract_user_id(case.request_metadata)

        await seed_history(
            session,
            conversation_id=conversation_id,
            history=case.history,
            owner_user_id=user_id,
        )
        if case.seeded_summary is not None:
            await memory_repository.upsert_conversation_summary(
                conversation_id=conversation_id,
                summary_text=case.seeded_summary.summary_text,
                last_summarized_message_id=case.seeded_summary.last_summarized_message_id,
                owner_user_id=user_id,
            )

        if user_id is not None:
            for seeded_memory in case.seeded_memories:
                await memory_repository.upsert_memory(
                    user_id=user_id,
                    kind=seeded_memory.kind,
                    key=seeded_memory.key,
                    value_json=seeded_memory.value_json,
                    confidence=seeded_memory.confidence,
                    source_message_id=1,
                    extraction_method=seeded_memory.extraction_method,
                    owner_user_id=user_id,
                )

        request_metadata = {} if case.request_metadata is None else dict(case.request_metadata)
        request_metadata["eval_case_id"] = case.id

        try:
            _, completion = await service.chat(
                conversation_id=conversation_id,
                owner_user_id=user_id,
                message=case.message,
                metadata=request_metadata,
            )
            provider.assert_exhausted()
        except Exception as exc:
            return await build_execution_failure_case_report(
                case,
                conversation_id=conversation_id,
                memory_repository=memory_repository,
                provider=provider,
                user_id=user_id,
                case_settings=case_settings,
                error=exc,
            )

        summary = await memory_repository.get_conversation_summary(
            conversation_id,
            owner_user_id=user_id,
        )
        active_memories = (
            []
            if user_id is None
            else await memory_repository.list_active_memories(
                user_id,
                limit=max(case_settings.memory_max_active_items, 64),
                owner_user_id=user_id,
            )
        )

    return build_case_report(
        case,
        completion=completion,
        provider=provider,
        summary=summary,
        active_memories=active_memories,
    )


async def build_execution_failure_case_report(
    case: MemoryEvalCase,
    *,
    conversation_id: str,
    memory_repository: SqlAlchemyMemoryRepository,
    provider: ScriptedMemoryEvalProvider,
    user_id: str | None,
    case_settings: Settings,
    error: Exception,
) -> MemoryEvalCaseReport:
    summary = await memory_repository.get_conversation_summary(
        conversation_id,
        owner_user_id=user_id,
    )
    active_memories = (
        []
        if user_id is None
        else await memory_repository.list_active_memories(
            user_id,
            limit=max(case_settings.memory_max_active_items, 64),
            owner_user_id=user_id,
        )
    )
    failure_reasons = [f"case execution failed: {error}"]
    return MemoryEvalCaseReport(
        case_id=case.id,
        passed=False,
        failure_reasons=failure_reasons,
        assistant_message=None,
        provider_calls=[serialize_provider_call(call) for call in provider.calls],
        summary=None if summary is None else serialize_summary(summary),
        active_memories=[serialize_memory(memory) for memory in active_memories],
        prompt_match=False if case_has_prompt_expectations(case) else True,
        summary_match=False if case_has_summary_expectations(case) else True,
        memory_match=False if case_has_memory_expectations(case) else True,
        answer_match=False if has_answer_expectations(case) else True,
    )


def build_case_report(
    case: MemoryEvalCase,
    *,
    completion: ChatCompletion,
    provider: ScriptedMemoryEvalProvider,
    summary: ConversationSummaryRecord | None,
    active_memories: Sequence[MemoryRecord],
) -> MemoryEvalCaseReport:
    failure_reasons: list[str] = []

    prompt_match, prompt_failures = compare_prompt_expectations(case.script, provider.calls)
    failure_reasons.extend(prompt_failures)

    summary_match, summary_failures = compare_summary_expectations(case, summary)
    failure_reasons.extend(summary_failures)

    memory_match, memory_failures = compare_memory_expectations(case, active_memories)
    failure_reasons.extend(memory_failures)

    answer_match, answer_failures = compare_answer_expectations(case, completion.content)
    failure_reasons.extend(answer_failures)

    return MemoryEvalCaseReport(
        case_id=case.id,
        passed=(
            prompt_match
            and summary_match
            and memory_match
            and answer_match
            and not failure_reasons
        ),
        failure_reasons=failure_reasons,
        assistant_message=completion.content,
        provider_calls=[serialize_provider_call(call) for call in provider.calls],
        summary=None if summary is None else serialize_summary(summary),
        active_memories=[serialize_memory(memory) for memory in active_memories],
        prompt_match=prompt_match,
        summary_match=summary_match,
        memory_match=memory_match,
        answer_match=answer_match,
    )


def compare_prompt_expectations(
    expected_steps: Sequence[MemoryEvalStep],
    actual_calls: Sequence[ProviderCallRecord],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if len(expected_steps) != len(actual_calls):
        failures.append(
            "provider call count mismatch: "
            f"expected {len(expected_steps)}, got {len(actual_calls)}"
        )
        return False, failures

    for index, (step, call) in enumerate(zip(expected_steps, actual_calls, strict=True)):
        rendered_prompt = render_messages(call.messages)
        for substring in step.expected_prompt_substrings:
            if substring not in rendered_prompt:
                failures.append(
                    f"provider call {index} ({step.kind}) missing expected prompt substring: "
                    f"{substring}"
                )
        for substring in step.forbidden_prompt_substrings:
            if substring in rendered_prompt:
                failures.append(
                    f"provider call {index} ({step.kind}) contained forbidden prompt substring: "
                    f"{substring}"
                )

    return not failures, failures


def compare_summary_expectations(
    case: MemoryEvalCase,
    summary: ConversationSummaryRecord | None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if case.expect_summary_present is not None:
        if case.expect_summary_present and summary is None:
            failures.append("expected conversation summary to be present")
        if not case.expect_summary_present and summary is not None:
            failures.append("expected conversation summary to be absent")

    if case.expected_summary is not None:
        if summary is None:
            failures.append("expected summary details but no summary was stored")
            return False, failures
        if (
            case.expected_summary.exact_text is not None
            and summary.summary_text != case.expected_summary.exact_text
        ):
            failures.append("persisted summary text mismatch")
        for substring in case.expected_summary.text_substrings:
            if substring not in summary.summary_text:
                failures.append(f"persisted summary missing substring: {substring}")
        if (
            case.expected_summary.last_summarized_message_id is not None
            and summary.last_summarized_message_id
            != case.expected_summary.last_summarized_message_id
        ):
            failures.append(
                "persisted summary last_summarized_message_id mismatch: "
                f"expected {case.expected_summary.last_summarized_message_id}, "
                f"got {summary.last_summarized_message_id}"
            )

    return not failures, failures


def compare_memory_expectations(
    case: MemoryEvalCase,
    active_memories: Sequence[MemoryRecord],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if case.expected_active_memory_count is not None:
        actual_count = len(active_memories)
        if actual_count != case.expected_active_memory_count:
            failures.append(
                "active memory count mismatch: "
                f"expected {case.expected_active_memory_count}, got {actual_count}"
            )

    memories_by_key = {memory.key: memory for memory in active_memories}
    for expected in case.expected_active_memories:
        actual = memories_by_key.get(expected.key)
        if actual is None:
            failures.append(f"missing expected active memory: {expected.key}")
            continue
        if expected.kind is not None and actual.kind != expected.kind:
            failures.append(
                f"active memory {expected.key} kind mismatch: expected {expected.kind}, "
                f"got {actual.kind}"
            )
        if (
            expected.extraction_method is not None
            and actual.extraction_method != expected.extraction_method
        ):
            failures.append(
                "active memory "
                f"{expected.key} extraction_method mismatch: "
                f"expected {expected.extraction_method}, got {actual.extraction_method}"
            )
        if expected.value_subset is not None and not is_deep_subset(
            expected.value_subset,
            actual.value_json,
        ):
            failures.append(f"active memory {expected.key} value subset mismatch")

    return not failures, failures


def compare_answer_expectations(
    case: MemoryEvalCase,
    assistant_message: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for substring in case.expected_answer_substrings:
        if substring not in assistant_message:
            failures.append(f"missing expected answer substring: {substring}")
    for substring in case.forbidden_answer_substrings:
        if substring in assistant_message:
            failures.append(f"found forbidden answer substring: {substring}")
    return not failures, failures


def case_has_prompt_expectations(case: MemoryEvalCase) -> bool:
    return any(
        step.expected_prompt_substrings or step.forbidden_prompt_substrings
        for step in case.script
    )


def case_has_summary_expectations(case: MemoryEvalCase) -> bool:
    return case.expect_summary_present is not None or case.expected_summary is not None


def case_has_memory_expectations(case: MemoryEvalCase) -> bool:
    return (
        case.expected_active_memory_count is not None
        or bool(case.expected_active_memories)
    )


def has_answer_expectations(case: MemoryEvalCase) -> bool:
    return bool(case.expected_answer_substrings or case.forbidden_answer_substrings)


def build_report(
    *,
    dataset_path: str | Path,
    dataset: MemoryEvalDataset,
    case_reports: Sequence[MemoryEvalCaseReport],
) -> MemoryEvalReport:
    total_cases = len(case_reports)
    passed_cases = sum(report.passed for report in case_reports)
    failed_case_ids = [report.case_id for report in case_reports if not report.passed]
    prompt_expectation_case_count = sum(
        case_has_prompt_expectations(case) for case in dataset.cases
    )
    summary_expectation_case_count = sum(
        case_has_summary_expectations(case) for case in dataset.cases
    )
    memory_expectation_case_count = sum(
        case_has_memory_expectations(case) for case in dataset.cases
    )
    answer_expectation_case_count = sum(has_answer_expectations(case) for case in dataset.cases)

    prompt_match_count = sum(
        report.prompt_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if case_has_prompt_expectations(case)
    )
    summary_match_count = sum(
        report.summary_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if case_has_summary_expectations(case)
    )
    memory_match_count = sum(
        report.memory_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if case_has_memory_expectations(case)
    )
    answer_match_count = sum(
        report.answer_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if has_answer_expectations(case)
    )

    return MemoryEvalReport(
        generated_at=datetime.now(UTC),
        dataset_path=str(Path(dataset_path)),
        summary=MemoryEvalSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            pass_rate=safe_ratio(passed_cases, total_cases),
            failed_case_ids=failed_case_ids,
            prompt_expectation_case_count=prompt_expectation_case_count,
            prompt_match_rate=(
                safe_ratio(prompt_match_count, prompt_expectation_case_count)
                if prompt_expectation_case_count
                else 0.0
            ),
            summary_expectation_case_count=summary_expectation_case_count,
            summary_match_rate=(
                safe_ratio(summary_match_count, summary_expectation_case_count)
                if summary_expectation_case_count
                else 0.0
            ),
            memory_expectation_case_count=memory_expectation_case_count,
            memory_match_rate=(
                safe_ratio(memory_match_count, memory_expectation_case_count)
                if memory_expectation_case_count
                else 0.0
            ),
            answer_expectation_case_count=answer_expectation_case_count,
            answer_match_rate=(
                safe_ratio(answer_match_count, answer_expectation_case_count)
                if answer_expectation_case_count
                else 0.0
            ),
        ),
        cases=list(case_reports),
    )


async def run_memory_eval(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path | None = None,
    settings: Settings | None = None,
) -> MemoryEvalReport:
    resolved_settings = settings or get_settings()
    dataset = load_memory_eval_dataset(dataset_path)
    case_reports = await evaluate_memory_dataset(dataset.cases, settings=resolved_settings)
    report = build_report(
        dataset_path=dataset_path,
        dataset=dataset,
        case_reports=case_reports,
    )
    if output_path is not None:
        write_report(output_path, report)
    return report


def render_messages(messages: Sequence[ChatTurn]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def serialize_provider_call(call: ProviderCallRecord) -> MemoryEvalProviderCallReport:
    return MemoryEvalProviderCallReport(
        kind=call.kind,
        messages=[
            MemoryEvalProviderCallMessage(role=message.role, content=message.content)
            for message in call.messages
        ],
    )


def serialize_summary(summary: ConversationSummaryRecord) -> MemoryEvalSummaryReport:
    return MemoryEvalSummaryReport(
        conversation_id=summary.conversation_id,
        summary_text=summary.summary_text,
        last_summarized_message_id=summary.last_summarized_message_id,
        updated_at=summary.updated_at,
    )


def serialize_memory(memory: MemoryRecord) -> MemoryEvalRecordReport:
    return MemoryEvalRecordReport(
        id=memory.id,
        user_id=memory.user_id,
        kind=memory.kind,
        key=memory.key,
        value_json=memory.value_json,
        confidence=memory.confidence,
        source_message_id=memory.source_message_id,
        extraction_method=memory.extraction_method,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def format_summary(report: MemoryEvalReport) -> str:
    summary = report.summary
    lines = [
        f"Dataset: {report.dataset_path}",
        f"Cases: {summary.total_cases}",
        f"Passed cases: {summary.passed_cases}",
        f"Pass rate: {summary.pass_rate:.3f}",
        (
            "Prompt match rate: "
            f"{summary.prompt_match_rate:.3f} ({summary.prompt_expectation_case_count} cases)"
        ),
        (
            "Summary match rate: "
            f"{summary.summary_match_rate:.3f} ({summary.summary_expectation_case_count} cases)"
        ),
        (
            "Memory match rate: "
            f"{summary.memory_match_rate:.3f} ({summary.memory_expectation_case_count} cases)"
        ),
        (
            "Answer match rate: "
            f"{summary.answer_match_rate:.3f} ({summary.answer_expectation_case_count} cases)"
        ),
    ]
    if summary.failed_case_ids:
        lines.append(f"Failed cases: {', '.join(summary.failed_case_ids)}")
    else:
        lines.append("Failed cases: none")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic memory regression eval.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the memory eval dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(
        run_memory_eval(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )
    print(format_summary(report))


if __name__ == "__main__":
    main()
