"""Flow HITL: pause -> Pending(kind="flow_pause") -> respond -> durable resume.

This pins the one idiom from Part 3, expressed through a Flow step: a step calls
``ctx.pause_for(value)``; the run pauses durably with a typed
``Pending(kind="flow_pause")``; a reviewer decides via ``approvals.respond(...)``;
``runner.run(flow, resume=decision, ...)`` continues the flow with the decided
value threaded back as ``pause_for``'s return value. Same surface as Wave 3 tool
approval.
"""

from __future__ import annotations

import pytest

from yaab import Flow, Runner, State
from yaab.governance import approvals
from yaab.governance.approvals import InMemoryApprovalStore
from yaab.graph.checkpoint import MemorySaver
from yaab.types import RunContext


def _approval_flow() -> Flow:
    def parse(state: State, ctx: RunContext) -> dict:
        return {"amount": 5000}

    def await_approval(state: State, ctx: RunContext) -> dict:
        decision = ctx.pause_for({"needs": "approval", "amount": state["amount"]})
        return {"approved": decision == "approve", "decision": decision}

    def execute(state: State, ctx: RunContext) -> dict:
        verb = "executed" if state["approved"] else "declined"
        return {"result": verb}

    return (
        Flow[None, str]("refund")
        .step("parse", fn=parse)
        .step("await_approval", fn=await_approval)
        .step("execute", fn=execute)
        .start_at("parse")
        .then("parse", "await_approval")
        .then("await_approval", "execute")
        .then("execute", Flow.DONE)
        .returns("result")
    )


@pytest.mark.asyncio
async def test_flow_pause_surfaces_flow_pause_pending():
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    flow = _approval_flow()

    result = await runner.run(flow, "refund #42", session_id="cust-42")

    assert result.paused is True
    assert len(result.pending) == 1
    pending = result.pending[0]
    assert pending.kind == "flow_pause"
    # The pause_for value is surfaced for the reviewer.
    assert pending.payload == {"needs": "approval", "amount": 5000}


@pytest.mark.asyncio
async def test_flow_pause_respond_resume_threads_decision_value():
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    flow = _approval_flow()

    result = await runner.run(flow, "refund #42", session_id="cust-42")
    assert result.paused

    # Decide via the SAME idiom as tool approval.
    decision = await approvals.respond(result, by="alice", answer="approve", store=store)

    resumed = await runner.run(flow, resume=decision, session_id="cust-42")
    assert resumed.paused is False
    # The decided value flowed back as pause_for's return value, all the way
    # through the rest of the flow.
    assert resumed.output == "executed"


@pytest.mark.asyncio
async def test_flow_pause_resume_with_denial_value():
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    flow = _approval_flow()

    result = await runner.run(flow, "refund #42", session_id="deny-1")
    decision = await approvals.respond(result, by="bob", answer="reject", store=store)
    resumed = await runner.run(flow, resume=decision, session_id="deny-1")
    assert resumed.output == "declined"


@pytest.mark.asyncio
async def test_flow_pause_is_durable_across_runner_instances():
    # A pause persisted on one checkpointer+store is resumable from another
    # Runner sharing the same backends — the cross-replica resume floor.
    store = InMemoryApprovalStore()
    saver = MemorySaver()
    flow = _approval_flow()

    pausing_runner = Runner(run_checkpointer=saver, approval_store=store)
    result = await pausing_runner.run(flow, "refund #42", session_id="dur-1")
    assert result.paused

    decision = await approvals.respond(result, by="alice", answer="approve", store=store)

    # A fresh Runner (think: another pod) with the same backends resumes.
    resuming_runner = Runner(run_checkpointer=saver, approval_store=store)
    resumed = await resuming_runner.run(flow, resume=decision, session_id="dur-1")
    assert resumed.output == "executed"
