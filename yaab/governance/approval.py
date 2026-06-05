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

#: Multipliers for the suffixes accepted by :func:`_parse_duration`.
_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _parse_duration(value: str | float | None) -> float | None:
    """Parse a timeout into seconds: ``None``/number pass through; ``"2h"`` → 7200.

    Accepts a bare number (seconds) or a string with a unit suffix
    (``s``/``m``/``h``/``d``). A plain numeric string is read as seconds.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().lower()
    unit = text[-1]
    if unit in _DURATION_UNITS:
        return float(text[:-1]) * _DURATION_UNITS[unit]
    return float(text)


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
        correlation_key: Callable[[str, dict, RunContext], str] | None = None,
        timeout: str | float | None = None,
        on_timeout: str = "deny",
        escalate_to: str | None = None,
    ) -> None:
        if tools is None and needs_approval is None:
            raise ValueError("specify `tools` and/or `needs_approval`")
        if mode not in ("inline", "block", "queue"):
            raise ValueError("mode must be 'inline', 'block', or 'queue'")
        if mode == "queue" and store is None:
            raise ValueError("mode='queue' requires a `store` (an ApprovalStore)")
        if on_timeout not in ("deny", "approve", "escalate"):
            raise ValueError("on_timeout must be 'deny', 'approve', or 'escalate'")
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
        #: Optional ``(tool, args, ctx) -> str`` deriving a business key (e.g.
        #: ``f"customer:{args['to']}"``) so a reviewer who knows only the business
        #: identity can find the pending record via ``store.list_by_key``.
        self.correlation_key = correlation_key
        #: How long a queued request waits before the worker's reaper applies
        #: ``on_timeout``. Accepts a duration string (``"2h"``, ``"30m"``) or
        #: seconds; ``None`` means no deadline.
        self.timeout_seconds = _parse_duration(timeout)
        #: What the timeout reaper does when ``timeout`` elapses: ``"deny"`` |
        #: ``"approve"`` | ``"escalate"`` (route to ``escalate_to``).
        self.on_timeout = on_timeout
        #: The next reviewer/agent to route to when ``on_timeout == "escalate"``.
        self.escalate_to = escalate_to

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
        import json
        import time

        from .approvals import ApprovalRequest

        assert self.store is not None  # guaranteed by the queue-mode caller
        resume_id = ctx.state.get("temp:__resume_id__") or ctx.run_id
        # The id is deterministic so a crash-window re-pause re-derives the SAME id
        # (idempotent create self-heals). It folds in the arguments so two guarded
        # calls to the *same* tool in one parallel turn get distinct records — a
        # pure tool-name digest would collapse them into one (last-write-wins).
        arg_sig = json.dumps(args, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{ctx.run_id}|{resume_id}|{tool}|{arg_sig}".encode()).hexdigest()[
            :12
        ]
        approval_id = f"ap_{digest}"
        # The ``ask_user`` built-in is a *question*, not an approve/deny gate: its
        # prompt and answer schema travel from the call's arguments into the record
        # so the same pause machinery surfaces a typed question the human answers.
        kind = "question" if tool == "ask_user" else "approval"
        prompt = args.get("question") if kind == "question" else None
        answer_schema = args.get("answer_schema") if kind == "question" else None
        correlation_key = (
            self.correlation_key(tool, dict(args), ctx)
            if self.correlation_key is not None
            else None
        )
        expires_at = (
            time.time() + self.timeout_seconds if self.timeout_seconds is not None else None
        )
        req = ApprovalRequest(
            approval_id=approval_id,
            run_id=ctx.run_id,
            resume_id=resume_id,
            agent=agent,
            identity=ctx.identity,
            tool=tool,
            arguments=dict(args),
            kind=kind,
            prompt=prompt,
            answer_schema=answer_schema,
            correlation_key=correlation_key,
            expires_at=expires_at,
            on_timeout=self.on_timeout if expires_at is not None else None,
            escalate_to=self.escalate_to if expires_at is not None else None,
            timeout_seconds=self.timeout_seconds,
        )
        await self.store.create(req)
        raise ApprovalPending(
            tool,
            args,
            approval_id=req.approval_id,
            run_id=ctx.run_id,
            resume_id=resume_id,
            kind=kind,
            prompt=prompt,
            answer_schema=answer_schema,
            correlation_key=correlation_key,
            expires_at=expires_at,
        )


__all__ = ["ToolApprovalPlugin", "Approver", "NeedsApproval"]
