"""Approval pipeline: a durable graph that pauses for human approval.

Pattern: a workflow that must stop mid-run for a human decision and resume
later — e.g. a payment or content-publish gate. Uses StateGraph + a checkpointer
so the pause survives a process restart, and ctx.interrupt() for the gate.
"""

from __future__ import annotations

from typing import Any

from yaab.graph import END, START, Channel, CompiledGraph, MemorySaver, StateGraph


def build(checkpointer: Any = None) -> CompiledGraph:
    """Return a compiled draft → approve → execute graph."""

    def draft(state: dict, ctx: Any) -> dict:
        return {"draft": f"Wire transfer of ${state['amount']}"}

    def approve(state: dict, ctx: Any) -> dict:
        # Pause for a human; on resume this returns the supplied decision.
        decision = ctx.interrupt({"review": state["draft"], "amount": state["amount"]})
        return {"approved": bool(decision)}

    def execute(state: dict, ctx: Any) -> dict:
        return {"status": "EXECUTED" if state["approved"] else "REJECTED"}

    g = StateGraph(channels={"amount": Channel(default=0)})
    g.add_node("draft", draft)
    g.add_node("approve", approve)
    g.add_node("execute", execute)
    g.add_edge(START, "draft")
    g.add_edge("draft", "approve")
    g.add_edge("approve", "execute")
    g.set_finish_point("execute")
    return g.compile(checkpointer=checkpointer or MemorySaver())


def run(amount: int = 10_000, approve_decision: bool = True) -> dict[str, Any]:
    """Run the gate: first call pauses; the second resumes with the decision."""
    app = build()
    paused = app.invoke({"amount": amount}, thread_id="txn-1")
    assert paused.interrupted, "expected a human-approval pause"
    done = app.invoke(thread_id="txn-1", resume=approve_decision)
    return done.state


if __name__ == "__main__":
    print(run())
