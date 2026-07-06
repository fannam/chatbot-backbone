from __future__ import annotations

import re
from collections.abc import Callable

# Guardrails AI's `Guard.configure()` -- called from every `Guard`/`AsyncGuard`
# construction, not just once -- unconditionally reloads `settings.rc` from
# `~/.guardrailsrc` (or an all-defaults fallback when that file is absent,
# which defaults `enable_metrics`/`use_remote_inferencing` to True) via
# `RC.load()`. A one-time `settings.rc = ...` override at import time is
# therefore NOT sufficient: the very next Guard construction silently resets
# it back to the unsafe default. `RC.load` itself must be monkeypatched so
# every reload -- no matter how many Guards get built over the process
# lifetime -- keeps telemetry/remote inference off. This module must be the
# only place in the codebase that imports `guardrails`, so this patch is
# guaranteed to run before any Guard/Validator is constructed anywhere.
from guardrails.classes.rc import RC as _GuardrailsRC
from guardrails.settings import settings as _guardrails_settings

_SAFE_GUARDRAILS_RC = _GuardrailsRC(enable_metrics=False, use_remote_inferencing=False, token="")
_GuardrailsRC.load = classmethod(lambda cls, logger=None: _SAFE_GUARDRAILS_RC)  # type: ignore[method-assign]
_guardrails_settings.rc = _SAFE_GUARDRAILS_RC

from guardrails import AsyncGuard  # noqa: E402
from guardrails.errors import ValidationError as GuardrailsValidationError  # noqa: E402, F401
from guardrails.validator_base import (  # noqa: E402
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from openai import AsyncOpenAI  # noqa: E402

from chatbot_api.observability import ObservabilityService  # noqa: E402
from chatbot_api.providers import check_message_moderation  # noqa: E402

JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all|any|the) (previous|prior|above) instructions?",
        r"you are now (dan|in developer mode|unrestricted|jailbroken)",
        r"pretend (that )?you (have no|don't have any) (restrictions|rules|guidelines)",
        r"act as (if )?(you (have|had) no|an? (ai|assistant) (with no|without))",
        r"disregard (your|the) (system prompt|instructions|guidelines)",
        r"repeat (your|the) (system prompt|instructions) (verbatim|exactly|word for word)",
        r"reveal (your|the) (system prompt|hidden instructions)",
        r"(bypass|override|circumvent) (your|the) (safety|content) (filters?|policy|guidelines)",
        r"roleplay.*(with no|without) (restrictions|limits|filters)",
        r"\bdan\b.*do anything now",
        r"this is (a|for) .*(test|research|academic).*purposes?.*ignore.*"
        r"(rules|guidelines|restrictions)",
    )
]


def find_jailbreak_pattern(text: str) -> str | None:
    for pattern in JAILBREAK_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


@register_validator(name="chatbot-api/jailbreak-heuristic", data_type="string")
class JailbreakHeuristic(Validator):
    """Heuristic signal for common jailbreak/prompt-injection phrasing.

    This is a pattern-based layer, not an adversarially robust defense -- a
    paraphrased or novel attack can still slip through. Treat it as one signal
    among several (system prompt, moderation, output guardrails), not a
    silver bullet.
    """

    def __init__(self, on_fail: str | Callable | None = None) -> None:
        super().__init__(on_fail=on_fail)

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        matched = find_jailbreak_pattern(value)
        if matched is not None:
            return FailResult(error_message=f"jailbreak heuristic matched pattern: {matched}")
        return PassResult()


EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CREDIT_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
PHONE_PATTERN = re.compile(
    r"(?<!\d)\+?\d{0,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?!\d)"
)
NATIONAL_ID_PATTERN = re.compile(r"(?<!\d)\d{8,}(?!\d)")


def luhn_is_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, set[str]]:
    """Redact common PII patterns via regex/heuristics (email, phone, credit
    card, generic long-digit IDs) -- not an NER model. Will miss unusual
    formats and can false-positive on PII-shaped-but-not-PII numbers; only the
    credit-card pattern is precision-checked (via Luhn) since a false-positive
    redaction there is the most user-visible.
    """
    hits: set[str] = set()

    def _replace_email(match: re.Match[str]) -> str:
        hits.add("email")
        return "[REDACTED_EMAIL]"

    def _replace_card(match: re.Match[str]) -> str:
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_is_valid(digits):
            hits.add("credit_card")
            return "[REDACTED_CARD]"
        return match.group(0)

    def _replace_phone(match: re.Match[str]) -> str:
        hits.add("phone")
        return "[REDACTED_PHONE]"

    def _replace_id(match: re.Match[str]) -> str:
        hits.add("national_id")
        return "[REDACTED_ID]"

    redacted = EMAIL_PATTERN.sub(_replace_email, text)
    redacted = CREDIT_CARD_PATTERN.sub(_replace_card, redacted)
    redacted = PHONE_PATTERN.sub(_replace_phone, redacted)
    redacted = NATIONAL_ID_PATTERN.sub(_replace_id, redacted)
    return redacted, hits


@register_validator(name="chatbot-api/pii-detector", data_type="string")
class PiiDetector(Validator):
    """Heuristic PII detector/redactor (see `redact_pii`).

    Behavior depends entirely on the `on_fail` mode passed at construction:
    `on_fail="fix"` auto-redacts (Guardrails uses `FailResult.fix_value`),
    `on_fail="exception"` hard-blocks, and `on_fail=<callable>` (e.g. a
    log-and-pass-through callback) runs for awareness-only checks that must
    never alter or block the original text.
    """

    def __init__(self, on_fail: str | Callable | None = None) -> None:
        super().__init__(on_fail=on_fail)

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        redacted, hits = redact_pii(value)
        if not hits:
            return PassResult()
        return FailResult(
            error_message=f"pii detected: {', '.join(sorted(hits))}",
            fix_value=redacted,
        )


@register_validator(name="chatbot-api/moderation-check", data_type="string")
class ModerationCheckValidator(Validator):
    """Wraps the existing OpenAI Moderation API call as a Guardrails
    validator, reusing `check_message_moderation` rather than reinventing a
    toxicity classifier. Only meaningful with `on_fail="exception"`.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        on_fail: str | Callable | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail)
        self._client = client
        self._model = model

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        raise NotImplementedError("ModerationCheckValidator only supports async validation")

    async def async_validate(self, value: str, metadata: dict) -> ValidationResult:
        flagged = await check_message_moderation(self._client, value, model=self._model)
        if flagged:
            return FailResult(error_message="message flagged by moderation model")
        return PassResult()


def build_pii_logging_callback(
    observability: ObservabilityService | None,
) -> Callable[[str, FailResult], str]:
    def log_pii_and_pass(value: str, fail_result: FailResult) -> str:
        if observability is not None:
            observability.record_guardrail_check(direction="input", check="pii", outcome="detected")
            observability.log_event(
                "guardrail.input.pii_detected",
                level="info",
                check="pii",
                detail=fail_result.error_message,
            )
        return value

    return log_pii_and_pass


def build_input_guard(
    *,
    jailbreak_detection_enabled: bool,
    pii_detection_enabled: bool,
    observability: ObservabilityService | None = None,
) -> AsyncGuard | None:
    validators: list[Validator] = []
    if jailbreak_detection_enabled:
        validators.append(JailbreakHeuristic(on_fail="exception"))
    if pii_detection_enabled:
        validators.append(PiiDetector(on_fail=build_pii_logging_callback(observability)))
    if not validators:
        return None
    return AsyncGuard.for_string(validators)


def build_output_guard(
    *,
    output_guardrails_enabled: bool,
    moderation_client: AsyncOpenAI | None,
    moderation_model: str,
) -> AsyncGuard | None:
    if not output_guardrails_enabled:
        return None
    validators: list[Validator] = [PiiDetector(on_fail="fix")]
    if moderation_client is not None:
        validators.append(
            ModerationCheckValidator(moderation_client, moderation_model, on_fail="exception")
        )
    return AsyncGuard.for_string(validators)
