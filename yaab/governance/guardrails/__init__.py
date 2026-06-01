"""Industry-standard guardrail adapters.

Each adapter wraps a third-party guardrail engine behind YAAB's
:class:`~yaab.governance.policy.GuardrailScanner` Protocol, so it drops straight
into a :class:`~yaab.governance.policy.PolicyEngine` (or the component registry)
alongside the built-in scanners:

* :class:`PresidioPIIScanner` — Microsoft Presidio NER-based PII detect/redact.
* :class:`LLMGuardScanner`    — Protect AI's LLM-Guard input/output scanners.
* :class:`NeMoGuardrailsScanner` — NVIDIA NeMo Guardrails programmable rails.

The heavy dependencies are optional extras, imported lazily; each adapter also
accepts an injected engine so it is testable offline. All three register
themselves in the component registry under the ``guardrail`` kind, next to the
built-in scanners, so they are selectable by name::

    from yaab import get_component
    pii = get_component("guardrail", "presidio")
"""

from __future__ import annotations

from ...extensions import register  # noqa: E402

# Register the adapters + the built-in scanners in the component registry so the
# whole guardrail catalog is discoverable/selectable by name.
from ..policy import (  # noqa: E402
    PIIScanner,
    PromptInjectionScanner,
    SecretScanner,
    SystemPromptLeakScanner,
    TopicScanner,
)
from .llm_guard import LLMGuardScanner
from .nemo import NeMoGuardrailsScanner
from .presidio import PresidioPIIScanner

register("guardrail", "prompt_injection", lambda **kw: PromptInjectionScanner())
register("guardrail", "pii", lambda **kw: PIIScanner(**kw))
register("guardrail", "secrets", lambda **kw: SecretScanner())
register("guardrail", "topics", lambda **kw: TopicScanner(**kw))
register("guardrail", "system_prompt_leak", lambda **kw: SystemPromptLeakScanner(**kw))
register("guardrail", "presidio", lambda **kw: PresidioPIIScanner(**kw))
register("guardrail", "llm_guard", lambda **kw: LLMGuardScanner(**kw))
register("guardrail", "nemo", lambda **kw: NeMoGuardrailsScanner(**kw))


__all__ = ["PresidioPIIScanner", "LLMGuardScanner", "NeMoGuardrailsScanner"]
