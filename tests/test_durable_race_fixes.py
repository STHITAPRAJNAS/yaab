"""Regression tests for durable-store race conditions and fencing tokens.

These cover the atomicity contracts the production-readiness audit flagged:

* :class:`InMemoryRunStore.claim_next` is atomic — two concurrent claimers never
  win the same QUEUED row (finding: non-atomic read-filter-modify-write).
* ``RunStore.update(expect_status=...)`` is a guarded compare-and-set used to
  win a resume race exactly once (finding: double-resume).
* The ``lease_generation`` fencing token advances on claim and on reap, so a
  worker whose lease was reaped cannot finalize over a newer claimant (finding:
  reaper + stale worker double-execution).
* :meth:`SQLiteApprovalStore.decide` / :meth:`InMemoryApprovalStore.decide` are
  atomic and idempotent — the first decision is permanent (finding: two
  reviewers both decide the same request).
* :meth:`ApprovalStore.create` is idempotent — a crash-window re-pause with the
  same deterministic id never duplicates or clobbers a decision (finding 7).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab.governance.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    InMemoryApprovalStore,
    SQLiteApprovalStore,
)
from yaab.runs.base import RunRecord, RunStatus
from yaab.runs.memory import InMemoryRunStore
from yaab.runs.sqlite import SQLiteRunStore


def _record(run_id: str) -> RunRecord:
    now = time.time()
    return RunRecord(
        run_id=run_id,
        agent="svc",
        status=RunStatus.QUEUED,
        prompt="hi",
        background=True,
        resume_id=run_id,
        created_at=now,
        updated_at=now,
    )


# --- finding 1: InMemoryRunStore.claim_next atomicity ----------------------
def test_in_memory_claim_next_is_atomic_under_concurrency():
    """Many concurrent claimers across one queued row: exactly one wins it."""

    async def go() -> None:
        store = InMemoryRunStore()
        # 20 queued runs, 50 concurrent claimers -> at most 20 successful claims,
        # and never the same run_id claimed twice.
        for i in range(20):
            await store.create(_record(f"r{i}"))

        async def claim() -> str | None:
            rec = await store.claim_next(pod_id="p", lease_seconds=30)
            return rec.run_id if rec else None

        results = await asyncio.gather(*[claim() for _ in range(50)])
        claimed = [r for r in results if r is not None]
        assert len(claimed) == 20  # exactly the queue depth
        assert len(set(claimed)) == 20  # no run claimed twice

    asyncio.run(go())


# --- finding 5: guarded compare-and-set transition -------------------------
def test_update_expect_status_wins_once():
    """Concurrent PAUSED->RUNNING transitions: exactly one update succeeds."""

    async def go() -> None:
        store = InMemoryRunStore()
        rec = _record("r1")
        await store.create(rec)
        await store.update("r1", status=RunStatus.PAUSED)

        async def resume() -> bool:
            got = await store.update("r1", expect_status=RunStatus.PAUSED, status=RunStatus.RUNNING)
            return got is not None

        wins = await asyncio.gather(*[resume() for _ in range(25)])
        assert sum(wins) == 1  # only one caller flips PAUSED -> RUNNING

    asyncio.run(go())


def test_update_expect_status_mismatch_returns_none():
    async def go() -> None:
        store = InMemoryRunStore()
        await store.create(_record("r1"))  # QUEUED
        got = await store.update("r1", expect_status=RunStatus.PAUSED, status=RunStatus.RUNNING)
        assert got is None
        # The record is unchanged (still QUEUED).
        cur = await store.get("r1")
        assert cur is not None and cur.status is RunStatus.QUEUED

    asyncio.run(go())


# --- finding 6: lease_generation fencing token -----------------------------
def test_lease_generation_advances_on_claim_and_reap():
    async def go() -> None:
        store = InMemoryRunStore()
        await store.create(_record("r1"))
        first = await store.claim_next(pod_id="A", lease_seconds=0.0)
        assert first is not None
        gen_after_claim = first.lease_generation
        assert gen_after_claim == 1
        # Lease already expired (0s); reaper re-queues and bumps the generation.
        reaped = await store.reap_expired_leases()
        assert reaped == ["r1"]
        requeued = await store.get("r1")
        assert requeued is not None
        assert requeued.lease_generation == gen_after_claim + 1
        assert requeued.status is RunStatus.QUEUED
        # The next claim bumps it again — a stale worker holding gen=1 is fenced.
        second = await store.claim_next(pod_id="B", lease_seconds=30)
        assert second is not None and second.lease_generation == 3

    asyncio.run(go())


def test_sqlite_lease_generation_advances(tmp_path):
    async def go() -> None:
        store = SQLiteRunStore(path=str(tmp_path / "runs.db"))
        await store.create(_record("r1"))
        claimed = await store.claim_next(pod_id="A", lease_seconds=0.0)
        assert claimed is not None and claimed.lease_generation == 1
        await store.reap_expired_leases()
        requeued = await store.get("r1")
        assert requeued is not None and requeued.lease_generation == 2

    asyncio.run(go())


# --- findings 3/4: approval decide atomic + idempotent ---------------------
def test_in_memory_decide_first_decision_is_permanent():
    async def go() -> None:
        store = InMemoryApprovalStore()
        await store.create(
            ApprovalRequest(
                approval_id="ap1", run_id="r1", resume_id="r1", agent="svc", tool="wire"
            )
        )

        async def approve() -> str:
            r = await store.decide("ap1", decision=ApprovalDecision.APPROVED, reviewer="alice")
            return r.decision.value if r else "none"

        async def deny() -> str:
            r = await store.decide("ap1", decision=ApprovalDecision.DENIED, reviewer="bob")
            return r.decision.value if r else "none"

        # Race an approve against a deny: whichever lands first is permanent, and
        # the loser observes that same decision (never overwrites it).
        results = await asyncio.gather(approve(), deny())
        assert results[0] == results[1]  # both see the same final decision
        final = await store.get("ap1")
        assert final is not None and final.decision in (
            ApprovalDecision.APPROVED,
            ApprovalDecision.DENIED,
        )

    asyncio.run(go())


def test_sqlite_decide_is_atomic_and_idempotent(tmp_path):
    async def go() -> None:
        store = SQLiteApprovalStore(path=str(tmp_path / "ap.db"))
        await store.create(
            ApprovalRequest(
                approval_id="ap1", run_id="r1", resume_id="r1", agent="svc", tool="wire"
            )
        )
        first = await store.decide("ap1", decision=ApprovalDecision.APPROVED, reviewer="alice")
        assert first is not None and first.decision is ApprovalDecision.APPROVED
        # A second decide does not overwrite the first — returns the existing one.
        second = await store.decide("ap1", decision=ApprovalDecision.DENIED, reviewer="bob")
        assert second is not None
        assert second.decision is ApprovalDecision.APPROVED
        assert second.reviewer == "alice"

    asyncio.run(go())


# --- finding 7: idempotent create + deterministic id self-heals ------------
def test_in_memory_create_is_idempotent_and_preserves_decision():
    async def go() -> None:
        store = InMemoryApprovalStore()
        req = ApprovalRequest(
            approval_id="ap_fixed", run_id="r1", resume_id="r1", agent="svc", tool="wire"
        )
        await store.create(req)
        await store.decide("ap_fixed", decision=ApprovalDecision.APPROVED, reviewer="alice")
        # A crash-window re-pause re-creates the SAME deterministic id; it must
        # not clobber the reviewer's decision.
        await store.create(req)
        got = await store.get("ap_fixed")
        assert got is not None and got.decision is ApprovalDecision.APPROVED
        assert got.reviewer == "alice"

    asyncio.run(go())


def test_sqlite_create_is_idempotent_and_preserves_decision(tmp_path):
    async def go() -> None:
        store = SQLiteApprovalStore(path=str(tmp_path / "ap.db"))
        req = ApprovalRequest(
            approval_id="ap_fixed", run_id="r1", resume_id="r1", agent="svc", tool="wire"
        )
        await store.create(req)
        await store.decide("ap_fixed", decision=ApprovalDecision.DENIED, reviewer="bob")
        await store.create(req)  # idempotent re-create
        got = await store.get("ap_fixed")
        assert got is not None and got.decision is ApprovalDecision.DENIED
        assert got.reviewer == "bob"

    asyncio.run(go())


def test_deterministic_approval_id_from_plugin():
    """The queue-mode plugin derives a stable id, so a re-pause reuses it."""
    import hashlib

    from yaab.governance.approval import ToolApprovalPlugin
    from yaab.types import RunContext

    async def go() -> None:
        import json

        store = InMemoryApprovalStore()
        plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=store)
        ctx = RunContext(deps=None, session_id=None, identity="alice")
        ctx.state["temp:__resume_id__"] = "resume-key"

        # The id includes an arg signature so concurrent calls to the SAME tool in
        # one parallel turn get distinct records (a pure tool-name digest would
        # collapse them, last-write-wins).
        args = {"amount": 1}
        arg_sig = json.dumps(args, sort_keys=True, default=str)
        expected_digest = hashlib.sha256(
            f"{ctx.run_id}|resume-key|wire|{arg_sig}".encode()
        ).hexdigest()[:12]
        expected_id = f"ap_{expected_digest}"

        with pytest.raises(Exception):  # noqa: B017 - ApprovalPending
            await plugin._queue_and_pause(ctx, "svc", "wire", args)
        got = await store.get(expected_id)
        assert got is not None and got.tool == "wire"

        # A re-pause with the same run + resume key + tool + args reuses the id
        # (idempotent create), so a crash-then-resume doesn't duplicate the record.
        with pytest.raises(Exception):  # noqa: B017
            await plugin._queue_and_pause(ctx, "svc", "wire", dict(args))
        assert len(await store.list_pending()) == 1

    asyncio.run(go())
