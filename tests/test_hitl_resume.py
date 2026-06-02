"""Durable human-in-the-loop: park a sensitive call, then resume on a decision.

This pins the full pause/checkpoint/resume mechanic for out-of-band tool
sign-off on the fast path:

* a guarded tool in ``queue`` mode persists a pending approval record and the
  run parks durably (no thread blocked) — emitting an ``APPROVAL_REQUIRED``
  event instead of raising, as long as a checkpointer + ``resume_id`` are set;
* re-invoking with the same ``resume_id`` and an injected approval decision
  executes the approved tool *now* (or injects the denial) and finishes the
  loop WITHOUT re-requesting the captured model turns;
* without a checkpointer the queue-mode plugin still raises ``ApprovalPending``
  (backward-compatible block behavior).
"""

from __future__ import annotations

from typing import Any

import pytest

from yaab import Agent
from yaab.exceptions import ApprovalPending
from yaab.governance.approval import ToolApprovalPlugin
from yaab.governance.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    InMemoryApprovalStore,
)
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool
from yaab.types import EventType, ToolCall

_EXECUTED: list[int] = []


def wire_transfer_impl(amount: int = 0) -> str:
    """Send money."""
    _EXECUTED.append(amount)
    return f"sent {amount}"


wire_transfer = FunctionTool(wire_transfer_impl, name="wire_transfer")


def _calls_wire_then_answers() -> TestModel:
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="wire_transfer", arguments={"amount": 100})],
                finish_reason="tool_calls",
            ),
            "transfer complete",
        ]
    )


@pytest.fixture(autouse=True)
def _reset_executed():
    _EXECUTED.clear()
    yield
    _EXECUTED.clear()


# --------------------------------------------------------------------------
# queue mode + checkpointer => durable pause (no raise).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_queue_mode_parks_run_and_emits_approval_required():
    store = InMemoryApprovalStore()
    saver = MemorySaver()
    plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)
    runner = Runner(run_checkpointer=saver, plugins=[plugin])
    agent = Agent("banker", model=_calls_wire_then_answers(), tools=[wire_transfer], runner=runner)

    events = []
    async for ev in runner.run_stream(agent, "wire 100", resume_id="hitl-1"):
        events.append(ev)

    types = [e.type for e in events]
    # Parked, not errored.
    assert EventType.APPROVAL_REQUIRED in types
    assert EventType.ERROR not in types
    assert EventType.RUN_END not in types
    # The guarded tool did NOT run yet.
    assert _EXECUTED == []

    # A pending approval was persisted, correlated to this run's resume_id.
    pending = await store.list_pending()
    assert len(pending) == 1
    req = pending[0]
    assert req.tool == "wire_transfer"
    assert req.arguments == {"amount": 100}
    assert req.resume_id == "hitl-1"
    assert req.decision is ApprovalDecision.PENDING

    # The APPROVAL_REQUIRED event carries the correlation id + call details.
    ap_ev = next(e for e in events if e.type is EventType.APPROVAL_REQUIRED)
    assert ap_ev.payload["approval_id"] == req.approval_id
    assert ap_ev.payload["tool"] == "wire_transfer"
    assert ap_ev.payload["arguments"] == {"amount": 100}

    # A pending-approval checkpoint was written so the run can resume later.
    _, state = saver.get("hitl-1")
    assert "pending_approval" in state
    assert state["pending_approval"]["tool"] == "wire_transfer"


# --------------------------------------------------------------------------
# Resume after APPROVED => execute the tool now, finish, no model replay.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_after_approval_executes_tool_and_finishes():
    store = InMemoryApprovalStore()
    saver = MemorySaver()
    plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)

    pausing_model = _calls_wire_then_answers()
    runner = Runner(run_checkpointer=saver, plugins=[plugin])
    agent = Agent("banker", model=pausing_model, tools=[wire_transfer], runner=runner)

    async for _ in runner.run_stream(agent, "wire 100", resume_id="hitl-2"):
        pass
    assert _EXECUTED == []
    req = (await store.list_pending())[0]

    # Reviewer approves out of band.
    await store.decide(req.approval_id, decision=ApprovalDecision.APPROVED, reviewer="alice")

    # Resume: fresh model that, if asked, would answer — but on resume the loop
    # only needs to run the approved tool and then take the captured final turn.
    resume_model = TestModel("transfer complete")
    resume_agent = Agent("banker", model=resume_model, tools=[wire_transfer], runner=runner)
    output = None
    saw_approval_required = False
    async for ev in runner.run_stream(
        resume_agent,
        "wire 100",
        resume_id="hitl-2",
        approval_decision="approved",
    ):
        if ev.type is EventType.RUN_END:
            output = ev.payload["result"].output
        if ev.type is EventType.APPROVAL_REQUIRED:
            saw_approval_required = True

    # The approved tool ran exactly once; the run finished.
    assert _EXECUTED == [100]
    assert output == "transfer complete"
    assert not saw_approval_required
    # The captured first model turn (the one that requested the tool) was NOT
    # re-requested on resume — only the post-tool continuation.
    assert len(resume_model.calls) <= 1


# --------------------------------------------------------------------------
# Resume after DENY => inject denial, continue, tool never runs.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_after_deny_injects_denial_and_continues():
    store = InMemoryApprovalStore()
    saver = MemorySaver()
    plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)
    runner = Runner(run_checkpointer=saver, plugins=[plugin])
    agent = Agent("banker", model=_calls_wire_then_answers(), tools=[wire_transfer], runner=runner)

    async for _ in runner.run_stream(agent, "wire 100", resume_id="hitl-3"):
        pass
    req = (await store.list_pending())[0]
    await store.decide(req.approval_id, decision=ApprovalDecision.DENIED, reviewer="bob")

    resume_model = TestModel("could not complete transfer")
    resume_agent = Agent("banker", model=resume_model, tools=[wire_transfer], runner=runner)
    output = None
    tool_results: list[Any] = []
    async for ev in runner.run_stream(
        resume_agent,
        "wire 100",
        resume_id="hitl-3",
        approval_decision="denied",
    ):
        if ev.type is EventType.TOOL_RESULT:
            tool_results.append(ev.payload["result"])
        if ev.type is EventType.RUN_END:
            output = ev.payload["result"].output

    # The guarded tool never executed; a denial was surfaced to the model.
    assert _EXECUTED == []
    assert output == "could not complete transfer"
    assert any("denied" in str(r).lower() for r in tool_results)


# --------------------------------------------------------------------------
# Back-compat: queue mode WITHOUT a checkpointer raises ApprovalPending.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_queue_mode_without_checkpointer_raises():
    store = InMemoryApprovalStore()
    plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)
    runner = Runner(plugins=[plugin])  # no checkpointer
    agent = Agent("banker", model=_calls_wire_then_answers(), tools=[wire_transfer], runner=runner)

    with pytest.raises(ApprovalPending) as ei:
        await runner.run(agent, "wire 100", resume_id="hitl-4")
    assert ei.value.tool == "wire_transfer"
    # The pending record was still persisted (decidable out of band).
    assert ei.value.approval_id
    persisted = await store.get(ei.value.approval_id)
    assert persisted is not None


# --------------------------------------------------------------------------
# A pending record persisted on one store view is visible to another (the
# "resume on a different worker" floor) — proven with the in-memory store by
# sharing the same instance the way two pods share one durable backend.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pending_record_is_durable_for_out_of_band_decision():
    store = InMemoryApprovalStore()
    req = ApprovalRequest(
        run_id="r1", resume_id="r1", agent="banker", tool="wire_transfer", arguments={"amount": 5}
    )
    await store.create(req)
    fetched = await store.get(req.approval_id)
    assert fetched is not None
    decided = await store.decide(
        req.approval_id, decision=ApprovalDecision.APPROVED, reviewer="alice"
    )
    assert decided is not None
    assert decided.decision is ApprovalDecision.APPROVED
