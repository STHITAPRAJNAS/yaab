"""NVIDIA NeMo Guardrails adapter — programmable conversational rails.

NeMo Guardrails enforces Colang-defined rails (topic boundaries, jailbreak
checks, fact-checking, etc.). Its native surface is conversational rather than a
stateless text scan, so this adapter bridges it through a ``check`` callable
``(text, stage) -> (allowed: bool, reason: str)`` that returns ``BLOCK`` when a
rail denies the text and ``ALLOW`` otherwise.

Provide either:

* ``check`` — any callable implementing the contract (easiest to test), or
* ``rails`` — a configured ``nemoguardrails.LLMRails``; the adapter derives a
  ``check`` from it (input rails on the prompt).

``nemoguardrails`` is an optional extra (``pip install 'yaab-sdk[nemo]'``), imported
lazily only when ``rails`` is used without a ``check``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..policy import Action, GuardrailResult, Stage

Check = Callable[[str, Stage], "tuple[bool, str]"]


class NeMoGuardrailsScanner:
    """Enforce NeMo Guardrails rails as a YAAB guardrail."""

    name = "nemo"
    stages: tuple[Stage, ...] = (Stage.INPUT, Stage.OUTPUT)

    def __init__(
        self,
        *,
        check: Check | None = None,
        rails: Any | None = None,
    ) -> None:
        if check is None and rails is None:
            raise ValueError("NeMoGuardrailsScanner requires either `check` or `rails`")
        self._check = check
        self._rails: Any = rails

    def _resolve_check(self) -> Check:
        if self._check is not None:
            return self._check
        rails = self._rails

        def _check(text: str, stage: Stage) -> tuple[bool, str]:
            # Run the prompt through NeMo's rails; a refusal/blocked response
            # means the rail denied it.
            result = rails.generate(messages=[{"role": "user", "content": text}])
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            blocked = "i'm not able to respond" in content.lower() or "blocked" in content.lower()
            return (not blocked, content if blocked else "")

        self._check = _check
        return _check

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        allowed, reason = self._resolve_check()(text, stage)
        if not allowed:
            return GuardrailResult(
                action=Action.BLOCK,
                scanner=self.name,
                reason=reason or "blocked by a NeMo rail",
                text=text,
            )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


__all__ = ["NeMoGuardrailsScanner"]
