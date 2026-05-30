"""Policy / Guardrail engine — defense-in-depth, runs as a runner plugin.

Input scanners (prompt-injection, PII, secrets, topic bans) run before the
model sees a prompt; output scanners (secret/PII leakage, system-prompt leak)
run on responses. Each scanner returns an :class:`Action` — ``allow``,
``redact``, ``flag``, or ``block`` — and every decision is audited.

The built-in scanners are dependency-free (regex + keyword). Adapters for
LLM Guard / NeMo Guardrails / custom NER can be dropped in by implementing the
:class:`GuardrailScanner` protocol.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel


class Stage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class Action(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    FLAG = "flag"
    BLOCK = "block"


class GuardrailResult(BaseModel):
    action: Action = Action.ALLOW
    scanner: str = ""
    reason: str = ""
    text: str = ""  # possibly redacted


@runtime_checkable
class GuardrailScanner(Protocol):
    name: str
    stages: tuple[Stage, ...]

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        ...


class PromptInjectionScanner:
    """Heuristic prompt-injection / jailbreak detector."""

    name = "prompt_injection"
    stages = (Stage.INPUT,)
    _patterns = [
        r"ignore (all |the )?(previous|prior|above) instructions",
        r"disregard (your|the) (system )?(prompt|instructions)",
        r"you are now (in )?(developer|dan|jailbreak) mode",
        r"reveal (your|the) system prompt",
        r"pretend (you are|to be) .* without restrictions",
    ]

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        for pat in self._patterns:
            if re.search(pat, text, re.IGNORECASE):
                return GuardrailResult(
                    action=Action.BLOCK,
                    scanner=self.name,
                    reason="possible prompt-injection / jailbreak attempt",
                    text=text,
                )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


class PIIScanner:
    """Detect and redact common PII (email, phone, SSN, credit card)."""

    name = "pii"
    stages = (Stage.INPUT, Stage.OUTPUT)
    _patterns = {
        "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "phone": r"\b\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b",
    }

    def __init__(self, action: Action = Action.REDACT) -> None:
        self.action = action

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        redacted = text
        found: list[str] = []
        for label, pat in self._patterns.items():
            if re.search(pat, redacted):
                found.append(label)
                redacted = re.sub(pat, f"[REDACTED_{label.upper()}]", redacted)
        if found:
            return GuardrailResult(
                action=self.action,
                scanner=self.name,
                reason=f"PII detected: {', '.join(found)}",
                text=redacted,
            )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


class SecretScanner:
    """Detect leaked credentials/API keys (blocks on output)."""

    name = "secrets"
    stages = (Stage.INPUT, Stage.OUTPUT)
    _patterns = [
        r"sk-[A-Za-z0-9]{16,}",
        r"AKIA[0-9A-Z]{16}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    ]

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        for pat in self._patterns:
            if re.search(pat, text):
                action = Action.BLOCK if stage is Stage.OUTPUT else Action.REDACT
                redacted = re.sub(pat, "[REDACTED_SECRET]", text)
                return GuardrailResult(
                    action=action,
                    scanner=self.name,
                    reason="secret/credential detected",
                    text=redacted,
                )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


class TopicScanner:
    """Allow/deny list of banned topics (keyword based)."""

    name = "topics"
    stages = (Stage.INPUT, Stage.OUTPUT)

    def __init__(self, banned: list[str]) -> None:
        self.banned = [b.lower() for b in banned]

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        low = text.lower()
        for term in self.banned:
            if term in low:
                return GuardrailResult(
                    action=Action.BLOCK,
                    scanner=self.name,
                    reason=f"banned topic: {term}",
                    text=text,
                )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


class SystemPromptLeakScanner:
    """Prevent the model from echoing its own system prompt."""

    name = "system_prompt_leak"
    stages = (Stage.OUTPUT,)

    def __init__(self, system_prompt: str = "") -> None:
        self.fingerprint = system_prompt.strip()[:80]

    def scan(self, text: str, stage: Stage) -> GuardrailResult:
        if self.fingerprint and self.fingerprint in text:
            return GuardrailResult(
                action=Action.BLOCK,
                scanner=self.name,
                reason="response leaks the system prompt",
                text=text,
            )
        return GuardrailResult(action=Action.ALLOW, scanner=self.name, text=text)


class PolicyEngine:
    """Runs a set of scanners over text at a given stage."""

    def __init__(self, scanners: Optional[list[GuardrailScanner]] = None) -> None:
        self.scanners: list[GuardrailScanner] = scanners or [
            PromptInjectionScanner(),
            PIIScanner(),
            SecretScanner(),
        ]

    def add(self, scanner: GuardrailScanner) -> None:
        self.scanners.append(scanner)

    def evaluate(self, text: str, stage: Stage) -> list[GuardrailResult]:
        """Run all stage-relevant scanners; redactions chain through the text."""
        results: list[GuardrailResult] = []
        current = text
        for scanner in self.scanners:
            if stage not in scanner.stages:
                continue
            result = scanner.scan(current, stage)
            if result.action is Action.REDACT:
                current = result.text
            results.append(result)
        return results

    @staticmethod
    def decide(results: list[GuardrailResult]) -> tuple[Action, str]:
        """Collapse scanner results into a single effective action + text."""
        text = ""
        effective = Action.ALLOW
        order = {Action.ALLOW: 0, Action.FLAG: 1, Action.REDACT: 2, Action.BLOCK: 3}
        for r in results:
            if r.text:
                text = r.text
            if order[r.action] > order[effective]:
                effective = r.action
        return effective, text
