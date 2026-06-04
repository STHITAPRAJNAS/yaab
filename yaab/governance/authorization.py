"""Pre-tool-call authorization & idempotency — a core governance seam.

Two common needs go together here: *authorize a tool call before it runs* and
*don't double-execute side-effecting tools on retry*. YAAB answers both as
Runner plugins that slot into the existing `before_tool` / `after_tool` hooks,
so they compose with guardrails and audit and require no changes to agents or
tools.

* :class:`ToolAuthorizer` — a protocol returning an allow/deny :class:`Decision`
  for a `(tool, args, ctx)` triple. Ship `RBACAuthorizer` (allow/deny lists +
  capability checks) and `CallableAuthorizer` (wrap any function).
* :class:`ToolAuthorizationPlugin` — enforces a list of authorizers before a
  tool runs; a denial is audited and either blocks the call (returns an error
  result to the model) or raises, by mode.
* :class:`IdempotencyPlugin` — dedupes side-effecting tool calls by an
  idempotency key (default: hash of tool name + args); a repeat returns the
  cached result instead of re-executing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..plugins import Plugin
from ..types import RunContext
from .audit import AuditKind, AuditLog


class Decision:
    """An authorization decision with an optional human-readable reason."""

    __slots__ = ("allowed", "reason")

    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason

    @classmethod
    def allow(cls, reason: str = "") -> Decision:
        return cls(True, reason)

    @classmethod
    def deny(cls, reason: str = "denied") -> Decision:
        return cls(False, reason)


@runtime_checkable
class ToolAuthorizer(Protocol):
    """Decides whether a tool call may proceed."""

    def authorize(self, tool: str, args: dict[str, Any], ctx: RunContext) -> Decision: ...


class RBACAuthorizer:
    """Allow/deny tools by name and by required capability.

    * ``allow`` — if set, only these tools may run (allow-list);
    * ``deny`` — these tools never run (takes precedence);
    * ``require_capability`` — map tool name -> capability string that the
      caller's ``ctx.state['capabilities']`` (a set/list) must contain.
    """

    def __init__(
        self,
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        require_capability: dict[str, str] | None = None,
    ) -> None:
        self.allow = set(allow) if allow is not None else None
        self.deny = set(deny or [])
        self.require_capability = dict(require_capability or {})

    def authorize(self, tool: str, args: dict[str, Any], ctx: RunContext) -> Decision:
        if tool in self.deny:
            return Decision.deny(f"tool '{tool}' is on the deny list")
        if self.allow is not None and tool not in self.allow:
            return Decision.deny(f"tool '{tool}' is not on the allow list")
        needed = self.require_capability.get(tool)
        if needed is not None:
            # The caller declares held capabilities on the run's shared state.
            # Accept a run-local (``temp:``) declaration as well as a plain one so
            # the scope is the caller's choice.
            held = set(ctx.state.get("temp:capabilities") or ctx.state.get("capabilities") or [])
            if needed not in held:
                return Decision.deny(f"missing capability '{needed}' for tool '{tool}'")
        return Decision.allow()


class CallableAuthorizer:
    """Wrap a plain function ``(tool, args, ctx) -> bool | Decision``."""

    def __init__(self, fn: Callable[[str, dict[str, Any], RunContext], Any]) -> None:
        self.fn = fn

    def authorize(self, tool: str, args: dict[str, Any], ctx: RunContext) -> Decision:
        result = self.fn(tool, args, ctx)
        if isinstance(result, Decision):
            return result
        return Decision.allow() if result else Decision.deny()


class ToolAuthorizationPlugin(Plugin):
    """Enforce a chain of authorizers before each tool call.

    All authorizers must allow; the first denial wins. With ``hard=True`` a
    denial raises :class:`PolicyViolation`; otherwise it short-circuits the tool
    with an error string fed back to the model (so the agent can adapt). Every
    decision that isn't a plain allow is audited.
    """

    name = "tool_authorization"

    def __init__(
        self,
        authorizers: list[ToolAuthorizer],
        *,
        audit: AuditLog | None = None,
        hard: bool = False,
    ) -> None:
        self.authorizers = authorizers
        self.audit = audit
        self.hard = hard

    async def before_tool(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> Any:
        for authorizer in self.authorizers:
            decision = authorizer.authorize(tool, args, ctx)
            if not decision.allowed:
                if self.audit is not None:
                    self.audit.record(
                        AuditKind.GUARDRAIL,
                        identity=ctx.identity,
                        stage="tool_authorization",
                        tool=tool,
                        action="deny",
                        reason=decision.reason,
                    )
                if self.hard:
                    from ..exceptions import PolicyViolation

                    raise PolicyViolation(
                        decision.reason, scanner="tool_authorization", stage="tool"
                    )
                return f"error: tool '{tool}' not authorized: {decision.reason}"
        return None


class IdempotencyPlugin(Plugin):
    """Dedupe side-effecting tool calls within a run (or across runs via a store).

    The idempotency key defaults to a hash of the tool name + sorted args; pass
    ``key_fn`` to derive it from domain fields (e.g. an order id). On a repeat
    key the cached result is returned and the tool is not executed again.

    By default the cache lives for the plugin's lifetime (shared across runs on
    the same Runner). Pass ``per_run=True`` to scope it to a single run via
    ``ctx.state``.
    """

    name = "idempotency"

    def __init__(
        self,
        *,
        tools: list[str] | None = None,
        key_fn: Callable[[str, dict[str, Any]], str] | None = None,
        per_run: bool = False,
    ) -> None:
        self.tools = set(tools) if tools is not None else None
        self.key_fn = key_fn
        self.per_run = per_run
        self._cache: dict[str, Any] = {}

    def _applies(self, tool: str) -> bool:
        return self.tools is None or tool in self.tools

    def _key(self, tool: str, args: dict[str, Any]) -> str:
        if self.key_fn is not None:
            return f"{tool}:{self.key_fn(tool, args)}"
        blob = json.dumps(args, sort_keys=True, default=str)
        return f"{tool}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"

    def _store(self, ctx: RunContext) -> dict[str, Any]:
        if self.per_run:
            # Run-local (temp:) so the dedupe cache never persists into the
            # durable session/checkpoint.
            return ctx.state.setdefault("temp:_idempotency", {})
        return self._cache

    async def before_tool(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> Any:
        if not self._applies(tool):
            return None
        key = self._key(tool, args)
        cached = self._store(ctx).get(key)
        if cached is not None:
            # Cache hit: short-circuit; the tool is NOT re-executed.
            return cached["value"]
        # Cache miss: leave a breadcrumb so after_tool can store the result.
        ctx.state.setdefault("temp:_idempotency_pending", []).append(key)
        return None

    async def after_tool(self, ctx: RunContext, agent: str, tool: str, result: Any) -> Any:
        if not self._applies(tool):
            return None
        pending = ctx.state.get("temp:_idempotency_pending")
        if pending:
            key = pending.pop()
            # Don't cache error strings — let the model retry a genuine failure.
            if not (isinstance(result, str) and result.startswith("error:")):
                self._store(ctx)[key] = {"value": result}
        return None


__all__ = [
    "Decision",
    "ToolAuthorizer",
    "RBACAuthorizer",
    "CallableAuthorizer",
    "ToolAuthorizationPlugin",
    "IdempotencyPlugin",
]
