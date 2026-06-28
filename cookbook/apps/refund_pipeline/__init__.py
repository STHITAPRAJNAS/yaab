"""Refund pipeline — a durable business workflow with a human gate.

A ``Flow`` parses a refund amount and branches: small refunds auto-approve, large
ones **pause for a human** (``ctx.pause_for``) and resume — on a fresh runner
call, even a different process — once a reviewer decides. The pause is a durable
checkpoint + an approval row, so the run survives a restart.

Capabilities: **Flow** (branch), **HITL pause→decide→resume**, durable
**checkpoint + approval store**, and the *same* ``approvals.respond`` idiom used
for tool approvals.

    python -m cookbook.apps.refund_pipeline
"""

from __future__ import annotations

import asyncio
from typing import Any

from cookbook._harness import expect
from yaab import Flow, Runner
from yaab.governance import approvals
from yaab.governance.approvals import InMemoryApprovalStore
from yaab.graph.checkpoint import MemorySaver


def build(amount: int) -> Flow:
    """A refund flow that auto-approves under $100 and gates the rest for a human."""

    def parse(state: Any, ctx: Any) -> dict:
        return {"amount": amount}

    def gate(state: Any, ctx: Any) -> dict:
        if state["amount"] < 100:
            return {"result": f"auto-refunded ${state['amount']}"}
        # Large refund: suspend the whole run until a human decides.
        decision = ctx.pause_for({"needs": "approval", "amount": state["amount"]})
        verb = "refunded" if decision == "approve" else "declined"
        return {"result": f"{verb} ${state['amount']}"}

    return (
        Flow[None, str]("refund")
        .step("parse", fn=parse)
        .step("gate", fn=gate)
        .start_at("parse")
        .then("parse", "gate")
        .then("gate", Flow.DONE)
        .returns("result")
    )


def _runner(store: InMemoryApprovalStore) -> Runner:
    return Runner(run_checkpointer=MemorySaver(), approval_store=store)


async def run() -> dict:
    # Small refund: completes without a human.
    store = InMemoryApprovalStore()
    small = await _runner(store).run(build(40), "refund order #1", session_id="cust-1")
    expect(not small.paused, "a small refund should not pause")
    expect("auto-refunded" in str(small.output), f"unexpected small result: {small.output!r}")

    # Large refund: pauses for approval, then resumes with the decision.
    store2 = InMemoryApprovalStore()
    runner = _runner(store2)
    paused = await runner.run(build(500), "refund order #2", session_id="cust-2")
    expect(paused.paused, "a $500 refund must pause for a human")
    expect(paused.pending[0].kind == "flow_pause", "the pause should surface as a flow_pause")

    decision = await approvals.respond(paused, by="alice", answer="approve", store=store2)
    resumed = await runner.run(build(500), resume=decision, session_id="cust-2")
    expect(not resumed.paused, "the run should finish after approval")
    expect("refunded $500" in str(resumed.output), f"unexpected resumed result: {resumed.output!r}")

    return {
        "small": str(small.output),
        "large_after_approval": str(resumed.output),
        "paused_kind": paused.pending[0].kind,
    }


if __name__ == "__main__":
    print(asyncio.run(run()))
