from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from chatbot_api.database import create_database_engine, create_session_factory
from chatbot_api.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from chatbot_api.eval_common import (
    ExpectedSource,
    is_deep_subset,
    safe_ratio,
    seed_history,
    source_matches_reference,
    write_report,
)
from chatbot_api.providers import (
    ChatCitation,
    ChatCompletion,
    ChatCompletionMetadata,
    ChatProvider,
    ChatTurn,
    TokenUsage,
    ToolCallBatch,
    ToolCallRequest,
    ToolResultMessage,
    ToolRun,
    UsageCost,
)
from chatbot_api.repositories import SqlAlchemyChatRepository, SqlAlchemyDocumentRepository
from chatbot_api.retrieval import DocumentRetriever
from chatbot_api.services import ChatService
from chatbot_api.settings import Settings, get_settings
from chatbot_api.tools import build_tool_registry

DEFAULT_DATASET_PATH = Path("evals/chat_tool_regression.json")


class EvalTokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total_tokens(self) -> EvalTokenUsage:
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class ChatEvalHistoryTurn(BaseModel):
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


class ScriptedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("call_id", "name")
    @classmethod
    def validate_non_blank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool call identifiers must not be blank")
        return normalized


class ToolCallBatchStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["tool_call_batch"]
    response_id: str
    tool_calls: list[ScriptedToolCall] = Field(min_length=1)
    usage: EvalTokenUsage | None = None
    provider: str = "openai"
    model: str = "gpt-4.1-mini"

    @field_validator("response_id", "provider", "model")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("step fields must not be blank")
        return normalized


class FinalAnswerStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["final_answer"]
    response_id: str
    content: str
    usage: EvalTokenUsage | None = None
    provider: str = "openai"
    model: str = "gpt-4.1-mini"

    @field_validator("response_id", "provider", "model", "content")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("step fields must not be blank")
        return normalized


ScriptedStep = Annotated[ToolCallBatchStep | FinalAnswerStep, Field(discriminator="type")]


class ExpectedChatEvalSource(ExpectedSource):
    pass


class ExpectedToolRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    status: Literal["completed", "failed", "rejected", "timed_out"]
    input_subset: dict[str, Any] | None = None
    output_subset: dict[str, Any] | None = None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name must not be blank")
        return normalized


class ChatEvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    message: str
    request_metadata: dict[str, Any] | None = None
    history: list[ChatEvalHistoryTurn] = Field(default_factory=list)
    script: list[ScriptedStep] = Field(min_length=1)
    expected_tool_runs: list[ExpectedToolRun] | None = None
    expected_sources: list[ExpectedChatEvalSource] = Field(default_factory=list)
    minimum_expected_source_hits: int = 0
    expected_answer_substrings: list[str] = Field(default_factory=list)
    forbidden_answer_substrings: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("id", "message")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case fields must not be blank")
        return normalized

    @field_validator("notes")
    @classmethod
    def validate_optional_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @field_validator("expected_answer_substrings", "forbidden_answer_substrings")
    @classmethod
    def validate_string_list(cls, values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("answer substring expectations must not be blank")
            normalized_values.append(normalized)
        return normalized_values

    @model_validator(mode="after")
    def validate_case(self) -> ChatEvalCase:
        if self.minimum_expected_source_hits < 0:
            raise ValueError("minimum_expected_source_hits must be non-negative")
        if self.minimum_expected_source_hits > len(self.expected_sources):
            raise ValueError(
                "minimum_expected_source_hits cannot exceed expected_sources length"
            )

        response_ids: set[str] = set()
        for index, step in enumerate(self.script):
            if step.response_id in response_ids:
                raise ValueError(f"duplicate script response_id: {step.response_id}")
            response_ids.add(step.response_id)

            is_last_step = index == len(self.script) - 1
            if is_last_step and not isinstance(step, FinalAnswerStep):
                raise ValueError("the final script step must be final_answer")
            if not is_last_step and not isinstance(step, ToolCallBatchStep):
                raise ValueError("only the final script step may be final_answer")

        return self


class ChatEvalDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[ChatEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> ChatEvalDataset:
        seen_case_ids: set[str] = set()
        for case in self.cases:
            if case.id in seen_case_ids:
                raise ValueError(f"duplicate chat eval case id: {case.id}")
            seen_case_ids.add(case.id)
        return self


class ChatEvalUsageReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatEvalCostReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    currency: Literal["USD"]


class ChatEvalCitationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    snippet: str


class ChatEvalToolRunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    status: Literal["completed", "failed", "rejected", "timed_out"]
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None


class ChatEvalCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    passed: bool
    failure_reasons: list[str]
    assistant_message: str | None
    tool_runs: list[ChatEvalToolRunReport]
    citations: list[ChatEvalCitationReport]
    usage: ChatEvalUsageReport | None = None
    cost: ChatEvalCostReport | None = None
    tool_runs_match: bool
    source_match: bool
    answer_match: bool
    matched_source_count: int


class ChatEvalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cases: int
    passed_cases: int
    pass_rate: float
    failed_case_ids: list[str]
    tool_expectation_case_count: int
    tool_match_rate: float
    source_expectation_case_count: int
    source_match_rate: float
    answer_expectation_case_count: int
    answer_match_rate: float


class ChatEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    dataset_path: str
    summary: ChatEvalSummary
    cases: list[ChatEvalCaseReport]


class ScriptedEvalProviderError(RuntimeError):
    """Raised when the chat eval provider is called with unexpected sequencing."""


class ScriptedEvalProvider(ChatProvider):
    provider_name = "scripted-eval"

    def __init__(self, steps: Sequence[ScriptedStep]) -> None:
        self._steps = list(steps)
        self._expected_previous_response_id: str | None = None
        self._expected_tool_output_count = 0
        self.calls: list[dict[str, Any]] = []

    async def generate_response(
        self,
        messages: Sequence[ChatTurn],
        *,
        tools: Sequence[Any] = (),
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolResultMessage] = (),
    ) -> ChatCompletion | ToolCallBatch:
        self.calls.append(
            {
                "messages": list(messages),
                "tool_names": [tool.name for tool in tools],
                "previous_response_id": previous_response_id,
                "tool_outputs": list(tool_outputs),
            }
        )
        self._validate_call(
            previous_response_id=previous_response_id,
            tool_outputs=tool_outputs,
        )
        if not self._steps:
            raise ScriptedEvalProviderError("provider called more times than the scripted steps")

        step = self._steps.pop(0)
        if isinstance(step, ToolCallBatchStep):
            self._expected_previous_response_id = step.response_id
            self._expected_tool_output_count = len(step.tool_calls)
            return ToolCallBatch(
                tool_calls=[
                    ToolCallRequest(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=dict(tool_call.arguments),
                    )
                    for tool_call in step.tool_calls
                ],
                provider=step.provider,
                model=step.model,
                response_id=step.response_id,
                usage=token_usage_from_eval(step.usage),
            )

        self._expected_previous_response_id = None
        self._expected_tool_output_count = 0
        return ChatCompletion(
            content=step.content,
            provider=step.provider,
            model=step.model,
            metadata=ChatCompletionMetadata(usage=token_usage_from_eval(step.usage)),
            response_id=step.response_id,
        )

    def assert_exhausted(self) -> None:
        if self._steps:
            raise ScriptedEvalProviderError(
                f"{len(self._steps)} scripted step(s) were not consumed by the workflow"
            )

    def _validate_call(
        self,
        *,
        previous_response_id: str | None,
        tool_outputs: Sequence[ToolResultMessage],
    ) -> None:
        if previous_response_id != self._expected_previous_response_id:
            raise ScriptedEvalProviderError(
                "unexpected previous_response_id: "
                f"expected {self._expected_previous_response_id!r}, got {previous_response_id!r}"
            )
        if len(tool_outputs) != self._expected_tool_output_count:
            raise ScriptedEvalProviderError(
                "unexpected tool output count: "
                f"expected {self._expected_tool_output_count}, got {len(tool_outputs)}"
            )


def load_chat_eval_dataset(path: str | Path) -> ChatEvalDataset:
    dataset_path = Path(path)
    payload = dataset_path.read_text(encoding="utf-8")
    try:
        parsed = ChatEvalDataset.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid chat eval dataset: {exc}") from exc
    return parsed


async def evaluate_chat_dataset(
    cases: Sequence[ChatEvalCase],
    *,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
) -> list[ChatEvalCaseReport]:
    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    reports: list[ChatEvalCaseReport] = []

    try:
        for case in cases:
            reports.append(
                await evaluate_chat_case(
                    case,
                    session_factory=session_factory,
                    settings=settings,
                    embedding_provider=embedding_provider,
                )
            )
    finally:
        await engine.dispose()

    return reports


async def evaluate_chat_case(
    case: ChatEvalCase,
    *,
    session_factory,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
) -> ChatEvalCaseReport:
    conversation_id = f"chat-eval-{case.id}-{uuid4()}"

    async with session_factory() as session:
        chat_repository = SqlAlchemyChatRepository(session)
        document_repository = SqlAlchemyDocumentRepository(session)
        retriever = DocumentRetriever(
            document_repository,
            embedding_provider,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
            max_chunks_per_document=settings.retrieval_max_chunks_per_document,
            candidate_limit=settings.retrieval_candidate_limit,
        )
        tool_registry = build_tool_registry(
            retriever=retriever,
            search_top_k=settings.tool_search_top_k,
            timeout_seconds=settings.tool_execution_timeout_seconds,
        )
        provider = ScriptedEvalProvider(case.script)
        service = ChatService(
            provider,
            chat_repository,
            tool_registry=tool_registry,
            tool_max_rounds=settings.tool_max_rounds,
            pricing_model=settings.openai_model,
            input_price_per_1m_tokens=settings.openai_model_input_price_per_1m_tokens,
            output_price_per_1m_tokens=settings.openai_model_output_price_per_1m_tokens,
        )

        if case.history:
            await seed_history(session, conversation_id=conversation_id, history=case.history)

        try:
            request_metadata = (
                {} if case.request_metadata is None else dict(case.request_metadata)
            )
            request_metadata["eval_case_id"] = case.id
            _, completion = await service.chat(
                conversation_id=conversation_id,
                message=case.message,
                metadata=request_metadata,
            )
            provider.assert_exhausted()
        except Exception as exc:
            return await build_execution_failure_case_report(
                case,
                conversation_id=conversation_id,
                chat_repository=chat_repository,
                error=exc,
            )

    return build_chat_case_report(case, completion)


async def build_execution_failure_case_report(
    case: ChatEvalCase,
    *,
    conversation_id: str,
    chat_repository: SqlAlchemyChatRepository,
    error: Exception,
) -> ChatEvalCaseReport:
    stored_tool_runs = list(
        reversed(await chat_repository.list_tool_runs(conversation_id, limit=100))
    )
    actual_tool_runs = [
        ToolRun(
            tool_call_id=tool_run.tool_call_id,
            tool_name=tool_run.tool_name,
            status=tool_run.status,  # type: ignore[arg-type]
            input=tool_run.input_payload,
            output=tool_run.output_payload,
            error=tool_run.error_message,
        )
        for tool_run in stored_tool_runs
    ]
    failure_reasons = [f"case execution failed: {error}"]
    return ChatEvalCaseReport(
        case_id=case.id,
        passed=False,
        failure_reasons=failure_reasons,
        assistant_message=None,
        tool_runs=[serialize_tool_run(tool_run) for tool_run in actual_tool_runs],
        citations=[],
        usage=None,
        cost=None,
        tool_runs_match=False if case.expected_tool_runs is not None else True,
        source_match=False if case.expected_sources else True,
        answer_match=False if has_answer_expectations(case) else True,
        matched_source_count=0,
    )


def build_chat_case_report(
    case: ChatEvalCase,
    completion: ChatCompletion,
) -> ChatEvalCaseReport:
    metadata = completion.metadata or ChatCompletionMetadata()
    citations = metadata.citations
    tool_runs = metadata.tool_runs

    failure_reasons: list[str] = []
    tool_runs_match, tool_run_failures = compare_tool_runs(case.expected_tool_runs, tool_runs)
    failure_reasons.extend(tool_run_failures)

    matched_source_count = calculate_matched_source_count(case.expected_sources, citations)
    source_match = matched_source_count >= case.minimum_expected_source_hits
    if not source_match:
        failure_reasons.append(
            "source expectations not met: "
            f"matched {matched_source_count} / {case.minimum_expected_source_hits} required"
        )

    answer_match, answer_failures = compare_answer_expectations(case, completion.content)
    failure_reasons.extend(answer_failures)

    return ChatEvalCaseReport(
        case_id=case.id,
        passed=tool_runs_match and source_match and answer_match and not failure_reasons,
        failure_reasons=failure_reasons,
        assistant_message=completion.content,
        tool_runs=[serialize_tool_run(tool_run) for tool_run in tool_runs],
        citations=[serialize_citation(citation) for citation in citations],
        usage=serialize_usage(metadata.usage),
        cost=serialize_cost(metadata.cost),
        tool_runs_match=tool_runs_match,
        source_match=source_match,
        answer_match=answer_match,
        matched_source_count=matched_source_count,
    )


def compare_tool_runs(
    expected_tool_runs: list[ExpectedToolRun] | None,
    actual_tool_runs: Sequence[ToolRun],
) -> tuple[bool, list[str]]:
    if expected_tool_runs is None:
        return True, []

    failures: list[str] = []
    if len(expected_tool_runs) != len(actual_tool_runs):
        failures.append(
            "tool run count mismatch: "
            f"expected {len(expected_tool_runs)}, got {len(actual_tool_runs)}"
        )
        return False, failures

    for index, (expected, actual) in enumerate(
        zip(expected_tool_runs, actual_tool_runs, strict=True)
    ):
        if expected.tool_name != actual.tool_name:
            failures.append(
                f"tool run {index} name mismatch: expected {expected.tool_name}, "
                f"got {actual.tool_name}"
            )
        if expected.status != actual.status:
            failures.append(
                f"tool run {index} status mismatch: expected {expected.status}, got {actual.status}"
            )
        if expected.input_subset is not None and not is_deep_subset(
            expected.input_subset,
            actual.input,
        ):
            failures.append(f"tool run {index} input subset mismatch")
        if expected.output_subset is not None and not is_deep_subset(
            expected.output_subset,
            actual.output,
        ):
            failures.append(f"tool run {index} output subset mismatch")

    return not failures, failures


def compare_answer_expectations(
    case: ChatEvalCase,
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


def calculate_matched_source_count(
    expected_sources: Sequence[ExpectedChatEvalSource],
    citations: Sequence[ChatCitation],
) -> int:
    return sum(
        1
        for source in expected_sources
        if any(source_matches_citation(source, citation) for citation in citations)
    )


def source_matches_citation(
    expected_source: ExpectedChatEvalSource,
    citation: ChatCitation,
) -> bool:
    if not source_matches_reference(
        expected_source,
        filename=citation.filename,
        document_id=citation.document_id,
    ):
        return False
    if expected_source.chunk_indexes and citation.chunk_index not in expected_source.chunk_indexes:
        return False
    return True


def has_answer_expectations(case: ChatEvalCase) -> bool:
    return bool(case.expected_answer_substrings or case.forbidden_answer_substrings)


def build_report(
    *,
    dataset_path: str | Path,
    dataset: ChatEvalDataset,
    case_reports: Sequence[ChatEvalCaseReport],
) -> ChatEvalReport:
    total_cases = len(case_reports)
    passed_cases = sum(report.passed for report in case_reports)
    failed_case_ids = [report.case_id for report in case_reports if not report.passed]
    tool_expectation_case_count = sum(case.expected_tool_runs is not None for case in dataset.cases)
    source_expectation_case_count = sum(bool(case.expected_sources) for case in dataset.cases)
    answer_expectation_case_count = sum(has_answer_expectations(case) for case in dataset.cases)
    tool_match_count = sum(
        report.tool_runs_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if case.expected_tool_runs is not None
    )
    source_match_count = sum(
        report.source_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if case.expected_sources
    )
    answer_match_count = sum(
        report.answer_match
        for case, report in zip(dataset.cases, case_reports, strict=True)
        if has_answer_expectations(case)
    )

    return ChatEvalReport(
        generated_at=datetime.now(UTC),
        dataset_path=str(Path(dataset_path)),
        summary=ChatEvalSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            pass_rate=safe_ratio(passed_cases, total_cases),
            failed_case_ids=failed_case_ids,
            tool_expectation_case_count=tool_expectation_case_count,
            tool_match_rate=safe_ratio(tool_match_count, tool_expectation_case_count)
            if tool_expectation_case_count
            else 0.0,
            source_expectation_case_count=source_expectation_case_count,
            source_match_rate=safe_ratio(source_match_count, source_expectation_case_count)
            if source_expectation_case_count
            else 0.0,
            answer_expectation_case_count=answer_expectation_case_count,
            answer_match_rate=safe_ratio(answer_match_count, answer_expectation_case_count)
            if answer_expectation_case_count
            else 0.0,
        ),
        cases=list(case_reports),
    )


async def run_chat_eval(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path | None = None,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> ChatEvalReport:
    resolved_settings = settings or get_settings()
    dataset = load_chat_eval_dataset(dataset_path)
    resolved_embedding_provider = embedding_provider or OpenAIEmbeddingProvider(resolved_settings)
    case_reports = await evaluate_chat_dataset(
        dataset.cases,
        settings=resolved_settings,
        embedding_provider=resolved_embedding_provider,
    )
    report = build_report(
        dataset_path=dataset_path,
        dataset=dataset,
        case_reports=case_reports,
    )
    if output_path is not None:
        write_report(output_path, report)
    return report


def serialize_citation(citation: ChatCitation) -> ChatEvalCitationReport:
    return ChatEvalCitationReport(
        document_id=citation.document_id,
        filename=citation.filename,
        chunk_index=citation.chunk_index,
        start_offset=citation.start_offset,
        end_offset=citation.end_offset,
        snippet=citation.snippet,
    )


def serialize_tool_run(tool_run: ToolRun) -> ChatEvalToolRunReport:
    return ChatEvalToolRunReport(
        tool_call_id=tool_run.tool_call_id,
        tool_name=tool_run.tool_name,
        status=tool_run.status,
        input=tool_run.input,
        output=tool_run.output,
        error=tool_run.error,
    )


def serialize_usage(usage: TokenUsage | None) -> ChatEvalUsageReport | None:
    if usage is None:
        return None
    return ChatEvalUsageReport(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def serialize_cost(cost: UsageCost | None) -> ChatEvalCostReport | None:
    if cost is None:
        return None
    return ChatEvalCostReport(
        input_cost_usd=cost.input_cost_usd,
        output_cost_usd=cost.output_cost_usd,
        total_cost_usd=cost.total_cost_usd,
        currency=cost.currency,
    )


def token_usage_from_eval(usage: EvalTokenUsage | None) -> TokenUsage | None:
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def format_summary(report: ChatEvalReport) -> str:
    summary = report.summary
    lines = [
        f"Dataset: {report.dataset_path}",
        f"Cases: {summary.total_cases}",
        f"Passed cases: {summary.passed_cases}",
        f"Pass rate: {summary.pass_rate:.3f}",
        (
            "Tool match rate: "
            f"{summary.tool_match_rate:.3f} ({summary.tool_expectation_case_count} cases)"
        ),
        (
            "Source match rate: "
            f"{summary.source_match_rate:.3f} ({summary.source_expectation_case_count} cases)"
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
    parser = argparse.ArgumentParser(description="Run end-to-end chat/tool regression eval.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the chat/tool eval dataset JSON file.",
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
        run_chat_eval(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )
    print(format_summary(report))


if __name__ == "__main__":
    main()
