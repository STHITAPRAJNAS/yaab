"""Human-in-the-loop tool approval for the model-driven fast path.

The graph engine pauses with ``interrupt()``; the fast-path agent loop couldn't —
so a sensitive tool call (``wire_transfer``, ``delete_account``) had no way to
require a human's sign-off mid-run. :class:`ToolApprovalPlugin` closes that gap
as a Runner plugin over the existing ``before_tool`` hook.

Three modes:

* **inline** — an async ``approver(tool, args, ctx) -> bool`` is awaited before
  the tool runs (e.g. prompt a CLI, call a Slack approval bot, check a queue);
  a rejection short-circuits the tool with a message the model can adapt to.
* **block** — no approver is given: a guarded tool raises
  :class:`~yaab.exceptions.ApprovalRequired`, surfacing the pending call so an
  out-of-band flow can approve and re-run.
* **queue** — a durable :class:`~yaab.governance.approvals.ApprovalStore` is
  given: a guarded tool persists a pending approval record (so any replica can
  review it) and raises :class:`~yaab.exceptions.ApprovalPending`, which the
  runner turns into a durable pause + resume signal instead of a hard error.
  This is out-of-band human sign-off that survives a restart and consumes zero
  compute while it waits.

Which tools need approval is decided by a name set or a predicate, so it pairs
naturally with the authorization layer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..exceptions import ApprovalPending, ApprovalRequired
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
        mode: str = "inline",
        store: Any | None = None,
    ) -> None:
        if tools is None and needs_approval is None:
            raise ValueError("specify `tools` and/or `needs_approval`")
        if mode not in ("inline", "block", "queue"):
            raise ValueError("mode must be 'inline', 'block', or 'queue'")
        if mode == "queue" and store is None:
            raise ValueError("mode='queue' requires a `store` (an ApprovalStore)")
        self._tools = set(tools or [])
        self._needs_approval = needs_approval
        self.approver = approver
        self.audit = audit
        #: Resolution strategy for a guarded call with no inline approver:
        #: ``"queue"`` persists a durable pending record and raises
        #: :class:`ApprovalPending`; otherwise a bare :class:`ApprovalRequired`
        #: is raised (``"inline"``/``"block"`` keep the original behavior).
        self.mode = mode
        #: Optional durable :class:`ApprovalStore`. When set, queued requests are
        #: persisted so a reviewer on any replica can decide them out of band.
        self.store = store

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
            if self.mode == "queue" and self.store is not None:
                await self._queue_and_pause(ctx, agent, tool, args)
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

    async def _queue_and_pause(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> None:
        """Persist a durable pending approval and raise :class:`ApprovalPending`.

        The run's checkpoint key (its ``resume_id``) is read from the run context
        — the runner threads it into ``ctx.state['temp:__resume_id__']`` — so the
        record correlates to the exact checkpoint the loop will resume from once a
        reviewer decides. Falls back to the run id when no resume key is set.

        The ``approval_id`` is *deterministic* — derived from the run, resume key,
        and tool — and :meth:`ApprovalStore.create` is idempotent. So if the
        process dies after this write but before the loop checkpoints the pending
        marker, the resume re-runs to this same gate and re-creates the identical
        record (never a duplicate, never clobbering a decision a reviewer already
        made), which makes the run self-heal across that crash window.
        """
        import hashlib

        from .approvals import ApprovalRequest

        assert self.store is not None  # guaranteed by the queue-mode caller
        resume_id = ctx.state.get("temp:__resume_id__") or ctx.run_id
        digest = hashlib.sha256(f"{ctx.run_id}|{resume_id}|{tool}".encode()).hexdigest()[:12]
        approval_id = f"ap_{digest}"
        req = ApprovalRequest(
            approval_id=approval_id,
            run_id=ctx.run_id,
            resume_id=resume_id,
            agent=agent,
            identity=ctx.identity,
            tool=tool,
            arguments=dict(args),
        )
        await self.store.create(req)
        raise ApprovalPending(
            tool,
            args,
            approval_id=req.approval_id,
            run_id=ctx.run_id,
            resume_id=resume_id,
        )


__all__ = ["ToolApprovalPlugin", "Approver", "NeedsApproval"]
