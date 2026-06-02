"""Regression tests for worker fencing and decided-but-paused reconciliation.

* A worker whose lease was reaped mid-run (so the record was re-queued and its
  generation bumped) must not finalize over the newer claimant — the stale
  ``_finalize`` is a no-op (finding 6).
* The worker reconciles a PAUSED run whose approval was decided but whose resume
  never fired, so a decision recorded just before a crash doesn't orphan its run
  forever (finding 8).
"""

from __future__ import annotations

import asyncio
import time

from yaab import Agent, tool
from yaab.governance.approval import ToolApprovalPlugin
from yaab.governance.approvals import ApprovalDecision, InMemoryApprovalStore
from yaab.graph.checkpoint import MemorySaver
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.runs.base import RunRecord, RunStatus
from yaab.runs.memory import InMemoryRunStore
from yaab.runs.worker import RunWorker


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


def test_stale_worker_does_not_finalize_over_newer_generation():
    """A reaped (fenced) worker's terminal write is skipped, not applied."""

    async def go() -> None:
        store = InMemoryRunStore()
        agent = Agent("svc", model=TestModel("done"))
        worker = RunWorker(agent, store)

        await store.create(_record("r1"))
        # The worker claimed under generation 1.
        claimed = await store.claim_next(pod_id="A", lease_seconds=30)
        assert claimed is not None and claimed.lease_generation == 1

        # Meanwhile the reaper re-queued the run (generation -> 2) and another
        # worker re-claimed it (generation -> 3) and completed it.
        await store.update("r1", status=RunStatus.QUEUED, lease_generation=2, owner_pod=None)
        await store.update(
            "r1", status=RunStatus.COMPLETED, lease_generation=3, output="newer-result"
        )

        # The stale worker (generation 1) now tries to finalize. The fence must
        # make this a no-op so the newer result survives.
        result_event = None
        await worker._finalize(claimed, result_event, error=RuntimeError("boom"), generation=1)

        final = await store.get("r1")
        assert final is not None
        assert final.status is RunStatus.COMPLETED
        assert final.output == "newer-result"  # not overwritten by the stale worker

    asyncio.run(go())


def test_current_generation_worker_finalizes_normally():
    async def go() -> None:
        store = InMemoryRunStore()
        agent = Agent("svc", model=TestModel("done"))
        worker = RunWorker(agent, store)
        await store.create(_record("r1"))
        claimed = await store.claim_next(pod_id="A", lease_seconds=30)
        assert claimed is not None

        await worker._finalize(
            claimed,
            result_event=None,
            error=RuntimeError("boom"),
            generation=claimed.lease_generation,
        )
        final = await store.get("r1")
        assert final is not None and final.status is RunStatus.FAILED

    asyncio.run(go())


def _approval_agent() -> Agent:
    @tool
    async def wire(ctx, amount: int = 100) -> str:
        """move money"""
        return f"wired {amount}"

    return Agent(
        "svc",
        model=TestModel(custom_output="all done", call_tools=["wire"]),
        tools=[wire],
        registry_id="svc",
    )


def test_worker_reconciles_decided_but_paused_run():
    """A decided approval whose resume never fired is re-driven by the worker."""

    async def go() -> None:
        approvals = InMemoryApprovalStore()
        runs = InMemoryRunStore()
        agent = _approval_agent()
        plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=approvals)
        runner = Runner(run_checkpointer=MemorySaver(), plugins=[plugin])

        resumed: list[tuple[str, str | None]] = []

        async def fake_resume(run_id: str, *, decision: str | None) -> bool:
            resumed.append((run_id, decision))
            await runs.update(run_id, status=RunStatus.COMPLETED)
            return True

        worker = RunWorker(
            agent,
            runs,
            runner=runner,
            approval_store=approvals,
            resume_paused=fake_resume,
        )

        # Stage a PAUSED run whose approval has already been decided but never
        # resumed (the crash window: decision recorded, resume never ran).
        rec = _record("r1")
        await runs.create(rec)
        await runs.update("r1", status=RunStatus.PAUSED)
        from yaab.governance.approvals import ApprovalRequest

        await approvals.create(
            ApprovalRequest(
                approval_id="ap1", run_id="r1", resume_id="r1", agent="svc", tool="wire"
            )
        )
        await approvals.decide("ap1", decision=ApprovalDecision.APPROVED, reviewer="alice")

        reconciled = await worker.reconcile_paused()
        assert reconciled == ["r1"]
        assert resumed == [("r1", "approved")]
        final = await runs.get("r1")
        assert final is not None and final.status is RunStatus.COMPLETED

    asyncio.run(go())


def test_reconcile_skips_undecided_paused_runs():
    async def go() -> None:
        approvals = InMemoryApprovalStore()
        runs = InMemoryRunStore()
        agent = _approval_agent()

        async def fake_resume(run_id: str, *, decision: str | None) -> bool:  # pragma: no cover
            raise AssertionError("should not resume an undecided run")

        worker = RunWorker(agent, runs, approval_store=approvals, resume_paused=fake_resume)
        await runs.create(_record("r1"))
        await runs.update("r1", status=RunStatus.PAUSED)
        from yaab.governance.approvals import ApprovalRequest

        await approvals.create(
            ApprovalRequest(
                approval_id="ap1", run_id="r1", resume_id="r1", agent="svc", tool="wire"
            )
        )  # still PENDING

        assert await worker.reconcile_paused() == []

    asyncio.run(go())
