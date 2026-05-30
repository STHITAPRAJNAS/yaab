"""Built-in plugins: audit, cost budget, and response caching.

These are the cross-cutting concerns most teams want from day one. Governance
enforcement (registry gate + guardrails) is wired directly into the Runner via
:class:`~yaab.governance.service.GovernanceService`; these plugins complement it.
"""

from __future__ import annotations

from typing import Any, Optional

from ..exceptions import YaabError
from ..governance.audit import AuditKind, AuditLog
from ..models.base import ModelResponse
from ..types import Message, RunContext
from . import Plugin


class BudgetExceeded(YaabError):
    """Raised when a run exceeds its configured cost budget."""


class AuditPlugin(Plugin):
    """Record fine-grained model and tool calls to an :class:`AuditLog`."""

    name = "audit"

    def __init__(self, audit: AuditLog) -> None:
        self.audit = audit

    async def after_model(
        self, ctx: RunContext, agent: str, response: ModelResponse
    ) -> Optional[ModelResponse]:
        self.audit.record(
            AuditKind.MODEL_CALL,
            identity=ctx.identity,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            finish_reason=response.finish_reason,
        )
        return None

    async def after_tool(self, ctx: RunContext, agent: str, tool: str, result: Any) -> Any:
        self.audit.record(AuditKind.TOOL_CALL, identity=ctx.identity, tool=tool)
        return None


class CostBudgetPlugin(Plugin):
    """Abort a run once its accumulated cost exceeds ``max_usd``."""

    name = "cost_budget"

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd

    async def after_model(
        self, ctx: RunContext, agent: str, response: ModelResponse
    ) -> Optional[ModelResponse]:
        if ctx.usage.cost_usd > self.max_usd:
            raise BudgetExceeded(
                f"run exceeded cost budget: ${ctx.usage.cost_usd:.4f} > ${self.max_usd:.4f}"
            )
        return None


class CachingPlugin(Plugin):
    """Cache model responses keyed by the conversation, short-circuiting repeats."""

    name = "caching"

    def __init__(self) -> None:
        self._cache: dict[str, ModelResponse] = {}
        self._last_key = ""

    @staticmethod
    def _key(messages: list[Message]) -> str:
        return "|".join(f"{m.role.value}:{m.content}" for m in messages)

    async def before_model(
        self, ctx: RunContext, agent: str, messages: list[Message]
    ) -> Optional[ModelResponse]:
        self._last_key = self._key(messages)
        cached = self._cache.get(self._last_key)
        if cached is not None:
            hit = cached.model_copy(deep=True)
            hit.usage.requests = 0  # a cache hit costs nothing
            hit.usage.cost_usd = 0.0
            return hit
        return None

    async def after_model(
        self, ctx: RunContext, agent: str, response: ModelResponse
    ) -> Optional[ModelResponse]:
        # Cache only terminal (non-tool) responses to keep keys stable.
        if self._last_key and not response.has_tool_calls:
            self._cache[self._last_key] = response
        return None


__all__ = ["AuditPlugin", "CostBudgetPlugin", "CachingPlugin", "BudgetExceeded"]
