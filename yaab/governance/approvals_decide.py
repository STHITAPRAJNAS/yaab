"""The unified human decision surface — one verb set, one resume value.

A paused run waits on a person. This module is *how a person decides*, in one
vocabulary that works the same whether the pause came from a guarded tool, an
``ask_user`` question, or a flow/graph step:

* :func:`approve` — let the held tool run;
* :func:`deny` — refuse it, with a ``reason`` the model reads and can revise from;
* :func:`edit` — approve with corrected ``arguments`` (the tool runs with those);
* :func:`respond` — answer an ``ask_user`` with a typed ``answer``.

Each returns a :class:`Decision` — the *single* value ``agent.run(resume=...)``
consumes. The decision is self-correlating: it carries the ``approval_id`` and the
``resume_id`` (the checkpoint key), so resume needs no ``session_id`` and works
from a fresh process given only the ``approval_id`` and the same store config.

Every verb accepts a ``RunResult``, a single :class:`~yaab.types.Pending`, or a
bare ``approval_id`` string as its target. When the target has several pendings,
pass ``approval_id=`` to choose one (or :func:`multiplex` to decide them all and
resume once). The typed payload is validated **before** the store is mutated, and
:meth:`ApprovalStore.decide` is first-write-wins — so a double-approve resumes
the run exactly once.

This module is exported as the ``approvals`` namespace::

    from yaab.governance import approvals
    decision = await approvals.approve(result, by="alice", store=store)
    final = await agent.run(resume=decision)
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from ..exceptions import GovernanceError
from .approvals import ApprovalDecision, ApprovalRequest, ApprovalStore
from .audit import AuditKind


class DecisionValidationError(GovernanceError):
    """Raised when a human's payload fails validation *before* anything is stored.

    The pending record stays pending and the run stays paused — a mistyped answer
    or malformed edit never half-commits a decision.
    """


class Decision(BaseModel):
    """A human's decision on one :class:`~yaab.types.Pending`.

    The only thing ``agent.run(resume=...)`` consumes. It is self-correlating: the
    ``approval_id`` (and the ``resume_id`` copied from the store row) locate the
    parked run's checkpoint, so resume never needs the original session or any
    in-memory object from the process that paused.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    approval_id: str
    verdict: Literal["approved", "denied"]
    by: str
    #: Denial feedback fed back to the model (``deny``), so it can revise.
    reason: str | None = None
    #: Reviewer-edited tool arguments (``edit``); the held tool runs with these.
    arguments: dict[str, Any] | None = None
    #: A typed answer to an ``ask_user`` (``respond``); becomes the tool's return.
    answer: Any = None
    #: The checkpoint key, copied from the store row — the resume correlation key.
    resume_id: str = ""
    run_id: str = ""


class ResumeBundle(BaseModel):
    """Several :class:`Decision` values keyed by ``approval_id``, resumed at once.

    Built by :func:`multiplex` when one model turn guarded multiple tools: decide
    each, then ``agent.run(resume=bundle)`` resolves every held tool with its
    matching decision in a single resume.
    """

    decisions: dict[str, Decision]


def _resolve_id(target: Any, approval_id: str | None) -> str:
    """Resolve the single ``approval_id`` a verb operates on.

    Accepts a ``RunResult`` (reads ``.pending``), a single ``Pending`` (reads its
    ``.approval_id``), or a bare ``approval_id`` string. An explicit ``approval_id=``
    always wins (and is required when the target carries several pendings).
    """
    if isinstance(target, str):
        return target
    if approval_id is not None:
        return approval_id
    # A single Pending (has approval_id but no pending list).
    pendings = getattr(target, "pending", None)
    if pendings is None and hasattr(target, "approval_id"):
        return target.approval_id  # type: ignore[no-any-return]
    pendings = pendings or []
    if len(pendings) == 1:
        return pendings[0].approval_id  # type: ignore[no-any-return]
    if not pendings:
        raise ValueError("target has no pending decision to act on (is the run actually paused?)")
    raise ValueError(
        "target has multiple pending decisions; pass approval_id= to choose one "
        "(or use approvals.multiplex to decide them all and resume once)"
    )


def _validate_payload(
    req: ApprovalRequest, *, arguments: dict[str, Any] | None = None, answer: Any = None
) -> None:
    """Validate a human's structured payload BEFORE the store mutates.

    An ``ask_user`` answer is checked against the request's declared JSON Schema
    (``answer_schema``) with ``jsonschema`` (the ``[hitl]`` extra, imported
    lazily). A mismatch raises :class:`DecisionValidationError` and nothing is
    written, so the pending row stays pending and the run stays paused. Edited
    arguments are validated against the held tool's own argument model in the
    runner before execution; a malformed mapping is rejected here defensively.
    """
    if arguments is not None and not isinstance(arguments, dict):
        raise DecisionValidationError("edited arguments must be a mapping of name -> value")
    if answer is not None and req.answer_schema is not None:
        try:
            import jsonschema
        except ImportError as exc:  # pragma: no cover - optional extra
            raise DecisionValidationError(
                "validating a typed answer against an answer_schema requires jsonschema; "
                "install with `pip install 'yaab-sdk[hitl]'`"
            ) from exc
        try:
            jsonschema.validate(answer, req.answer_schema)
        except jsonschema.ValidationError as exc:
            raise DecisionValidationError(
                f"answer does not match the declared answer_schema: {exc.message}"
            ) from exc


async def _decide(
    target: Any,
    *,
    store: ApprovalStore,
    verdict: Literal["approved", "denied"],
    by: str,
    approval_id: str | None = None,
    reason: str | None = None,
    arguments: dict[str, Any] | None = None,
    answer: Any = None,
    audit: Any = None,
) -> Decision:
    """One decision body shared by all four verbs.

    Resolves the target to an ``approval_id``, validates the payload (raising
    before any write), records the verdict idempotently (first-write-wins), audits
    it, and returns the self-correlating :class:`Decision` resume consumes.
    """
    resolved = _resolve_id(target, approval_id)
    req = await store.get(resolved)
    if req is None:
        raise KeyError(f"unknown approval {resolved!r}")
    # Validate-before-mutate: a bad payload leaves the row pending, the run paused.
    _validate_payload(req, arguments=arguments, answer=answer)
    decision_enum = ApprovalDecision.APPROVED if verdict == "approved" else ApprovalDecision.DENIED
    updated = await store.decide(
        resolved,
        decision=decision_enum,
        reviewer=by,
        reason=reason,
        override_arguments=arguments,
        answer=answer,
    )
    if updated is None:  # pragma: no cover - get() already proved it exists
        raise KeyError(f"unknown approval {resolved!r}")
    if audit is not None:
        audit.record(
            AuditKind.APPROVAL,
            identity=by,
            tool=req.tool,
            decision=verdict,
            reason=reason,
        )
    return Decision(
        approval_id=resolved,
        # Report the *stored* verdict, not the requested one: decide() is
        # first-write-wins, so a deny that loses to an earlier approve must
        # report "approved". The stored decision is always approved/denied here.
        verdict=cast(Literal["approved", "denied"], updated.decision.value),
        by=by,
        reason=reason,
        arguments=arguments,
        answer=answer,
        resume_id=updated.resume_id,
        run_id=updated.run_id,
    )


async def approve(
    target: Any, *, by: str, store: ApprovalStore, approval_id: str | None = None, audit: Any = None
) -> Decision:
    """Approve a parked tool call; on resume the held tool runs."""
    return await _decide(
        target, store=store, verdict="approved", by=by, approval_id=approval_id, audit=audit
    )


async def deny(
    target: Any,
    *,
    by: str,
    reason: str,
    store: ApprovalStore,
    approval_id: str | None = None,
    audit: Any = None,
) -> Decision:
    """Deny a parked tool call; ``reason`` is fed back to the model on resume."""
    return await _decide(
        target,
        store=store,
        verdict="denied",
        by=by,
        reason=reason,
        approval_id=approval_id,
        audit=audit,
    )


async def edit(
    target: Any,
    *,
    by: str,
    arguments: dict[str, Any],
    store: ApprovalStore,
    approval_id: str | None = None,
    audit: Any = None,
) -> Decision:
    """Approve with corrected ``arguments``; the held tool runs with those args."""
    return await _decide(
        target,
        store=store,
        verdict="approved",
        by=by,
        arguments=arguments,
        approval_id=approval_id,
        audit=audit,
    )


async def respond(
    target: Any,
    *,
    by: str,
    answer: Any,
    store: ApprovalStore,
    approval_id: str | None = None,
    audit: Any = None,
) -> Decision:
    """Answer an ``ask_user`` question; ``answer`` becomes the tool's return value.

    The typed ``answer`` is validated against the request's declared
    ``answer_schema`` (if any) before anything is stored.
    """
    return await _decide(
        target,
        store=store,
        verdict="approved",
        by=by,
        answer=answer,
        approval_id=approval_id,
        audit=audit,
    )


async def multiplex(target: Any, decisions: dict[str, Decision]) -> ResumeBundle:
    """Bundle a ``{approval_id: Decision}`` map for a multi-pending result.

    Decide several pendings (each with its own verb), then resume them all at
    once: ``agent.run(resume=await approvals.multiplex(result, {...}))``.
    """
    return ResumeBundle(decisions=dict(decisions))


__all__ = [
    "Decision",
    "ResumeBundle",
    "DecisionValidationError",
    "approve",
    "deny",
    "edit",
    "respond",
    "multiplex",
]
