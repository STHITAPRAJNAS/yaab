"""Protect AI LLM-Guard adapter — input/output scanner suite.

LLM-Guard ships a battery of scanners (PromptInjection, Toxicity, Anonymize,
BanTopics, Secrets, …). Each exposes ``scan(text) -> (sanitized, is_valid,
risk_score)``. This adapter runs a configured list of them behind YAAB's
:class:`~yaab.governance.policy.GuardrailScanner` Protocol: an invalid result
``BLOCK``s, a changed (sanitized) text ``REDACT``s, otherwise ``ALLOW``.

``llm-guard`` is an optional extra (``pip install 'yaab-sdk[llm-guard]'``). Pass your
own scanner lists; if none are given, a small sensible default set is built
lazily (and only then is the dependency imported).
"""

from __future__ import annotations

from typing import Any

from ..policy import Action, GuardrailResult, Stage


class LLMGuardScanner:
    """Run Protect AI LLM-Guard scanners as a YAAB guardrail."""

    name = "llm_guard"
    stages: tuple[Stage, ...] = (Stage.INPUT, Stage.OUTPUT)

    def __init__(
        self,
        *,
        input_scanners: list[Any] | None = None,
        output_scanners: list[Any] | None = None,
        fail_fast: bool = True,
    ) -> None:
        self._input = input_scanners
        self._output = output_scanners
        self.fail_fast = fail_fast

    def _ensure_defaults(self) -> None:
        if self._input is not None or self._output is not None:
            return
        try:
            from llm_guard.input_scanners import PromptInjection
            from llm_guard.output_scanners import NoRefusal
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "LLMGuardScanner needs Protect AI LLM-Guard. "
                "Install it with `pip install 'yaab-sdk[llm-guard]'`, or pass explicit "
                "input_scanners/output_scanners."
            ) from exc
        self._input = [PromptInjection()]
        self._output = [NoRefusal()]

    def _scanners_for(self, stage: Stage) -> list[Any]:
        self._ensure_defaults()
        return (self._input or []) if stage is Stage.INPUT else (self._output or [])

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        current = text
        redacted = False
        for scanner in self._scanners_for(stage):
            sanitized, is_valid, _score = scanner.scan(current)
            if not is_valid:
                return GuardrailResult(
                    action=Action.BLOCK,
                    scanner=self.name,
                    reason=f"{type(scanner).__name__} flagged the text",
                    text=current,
                )
            if sanitized != current:
                current = sanitized
                redacted = True
        if redacted:
            return GuardrailResult(
                action=Action.REDACT,
                scanner=self.name,
                reason="LLM-Guard sanitized the text",
                text=current,
            )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


__all__ = ["LLMGuardScanner"]
