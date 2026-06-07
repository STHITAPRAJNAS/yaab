"""D5: a Flow ``pause_for`` creates an ApprovalRequest(kind="flow_pause") row.

The binding decision (APPROVED-DECISIONS D5): every Flow pause is *also* an
approval record in the ApprovalStore, so it is visible in ``GET /approvals`` and
``approvals.respond()`` works on it identically to a tool approval. This test
pins that the row exists, is the right kind, is decidable, and that the decided
record drives the resume.
"""

from __future__ import annotations

import pytest

from yaab import Flow, Runner, State
from yaab.governance import approvals
from yaab.governance.approvals import ApprovalDecision, InMemoryApprovalStore
from yaab.graph.checkpoint import MemorySaver
from yaab.types import RunContext


def _pause_flow() -> Flow:
    def gate(state: State, ctx: RunContext) -> dict:
        decision = ctx.pause_for({"question": "approve refund?"})
        return {"answer": decision}

    return Flow[None, str]("gate_flow").step("gate", fn=gate).start_at("gate").returns("answer")


@pytest.mark.asyncio
async def test_pause_for_creates_flow_pause_approval_row():
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    flow = _pause_flow()

    result = await runner.run(flow, "go", session_id="rec-1")
    assert result.paused

    # D5: an ApprovalRequest row exists in the store, kind == "flow_pause".
    pending = await store.list_pending()
    assert len(pending) == 1
    req = pending[0]
    assert req.kind == "flow_pause"
    assert req.agent == "gate_flow"
    assert req.decision is ApprovalDecision.PENDING
    # The pause value travels into the record so a reviewer sees what to decide.
    assert req.arguments == {"question": "approve refund?"} or req.prompt is not None

    # The Pending surfaced to the caller carries the SAME approval_id as the row.
    assert result.pending[0].approval_id == req.approval_id


@pytest.mark.asyncio
async def test_flow_pause_visible_in_for_run_listing():
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    flow = _pause_flow()
    result = await runner.run(flow, "go", session_id="rec-2")

    # Visible via the same store query that GET /approvals uses.
    req = result.pending[0]
    for_run = await store.for_run(req.run_id)
    assert any(r.approval_id == req.approval_id for r in for_run)


@pytest.mark.asyncio
async def test_respond_decides_the_flow_pause_row_then_resume():
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    flow = _pause_flow()

    result = await runner.run(flow, "go", session_id="rec-3")
    req = result.pending[0]

    # approvals.respond works identically regardless of where the pause came from.
    decision = await approvals.respond(result, by="alice", answer="yes", store=store)

    # The store row is now decided (the audit trail of who/when).
    decided = await store.get(req.approval_id)
    assert decided is not None
    assert decided.decision is ApprovalDecision.APPROVED
    assert decided.reviewer == "alice"

    # And the same decision resumes the flow with the decided value.
    resumed = await runner.run(flow, resume=decision, session_id="rec-3")
    assert resumed.output == "yes"


@pytest.mark.asyncio
async def test_idempotent_pause_record_on_re_pause():
    # Re-running the pausing flow before a decision must not duplicate the record
    # (deterministic id + idempotent create), so a crash-window re-pause self-heals.
    store = InMemoryApprovalStore()
    saver = MemorySaver()
    flow = _pause_flow()

    runner = Runner(run_checkpointer=saver, approval_store=store)
    await runner.run(flow, "go", session_id="rec-4")
    await runner.run(flow, "go", session_id="rec-4")

    pending = await store.list_pending()
    assert len(pending) == 1
