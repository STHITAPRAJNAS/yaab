"""Human-in-the-loop tool approval for the model-driven fast path.

The graph engine pauses with ``interrupt()``; the fast-path agent loop couldn't —
so a sensitive tool call (``wire_transfer``, ``delete_account``) had no way to
require a human's sign-off mid-run. :class:`ToolApprovalPlugin` closes that gap
as a Runner plugin over the existing ``before_tool`` hook.

Two modes:

* **inline** — an async ``approver(tool, args, ctx) -> bool`` is awaited before
  the tool runs (e.g. prompt a CLI, call a Slack approval bot, check a queue);
  a rejection short-circuits the tool with a message the model can adapt to.
* **block** — no approver is given: a guarded tool raises
  :class:`~yaab.exceptions.ApprovalRequired`, surfacing the pending call so an
  out-of-band flow can approve and re-run.

Which tools need approval is decided by a name set or a predicate, so it pairs
naturally with the authorization layer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..exceptions import ApprovalRequired
from ..plugins import Plugin
from ..types import RunContext
from .audit import AuditKind, AuditLog

Approver = Callable[[str, dict, RunContext], Awaitable[bool]]
NeedsApproval = Callable[[str, dict, RunContext], bool]


class ToolApprovalPlugin(Plugin):
    """Require human approval before sensitive tool calls run."""

    name = "tool_approval"

    def __init__(
        self,
        *,
        tools: list[str] | None = None,
        needs_approval: NeedsApproval | None = None,
        approver: Approver | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        if tools is None and needs_approval is None:
            raise ValueError("specify `tools` and/or `needs_approval`")
        self._tools = set(tools or [])
        self._needs_approval = needs_approval
        self.approver = approver
        self.audit = audit

    def _guarded(self, tool: str, args: dict, ctx: RunContext) -> bool:
        if tool in self._tools:
            return True
        if self._needs_approval is not None:
            return bool(self._needs_approval(tool, args, ctx))
        return False

    async def before_tool(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> Any:
        if not self._guarded(tool, args, ctx):
            return None

        if self.approver is None:
            # No inline approver: surface the pending call for out-of-band review.
            if self.audit is not None:
                self.audit.record(
                    AuditKind.APPROVAL, identity=ctx.identity, tool=tool, decision="pending"
                )
            raise ApprovalRequired(tool, args)

        approved = await self.approver(tool, args, ctx)
        if self.audit is not None:
            self.audit.record(
                AuditKind.APPROVAL,
                identity=ctx.identity,
                tool=tool,
                decision="approved" if approved else "rejected",
            )
        if approved:
            return None  # let the tool run
        return f"error: tool '{tool}' was not approved by a human reviewer."


__all__ = ["ToolApprovalPlugin", "Approver", "NeedsApproval"]
