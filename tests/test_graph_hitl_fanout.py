"""Graph HITL + fan-out: an interrupt must not re-apply already-run nodes.

When a superstep holds several nodes (a fan-out) and a *later* one interrupts,
resuming must not re-execute the nodes that already ran in that superstep —
otherwise their updates apply twice. With an accumulating reducer ("add"/
"append") that silently corrupts state; with "last_value" it's masked.
"""

from __future__ import annotations

import pytest

from yaab.graph import START, Channel, MemorySaver, StateGraph


@pytest.mark.asyncio
async def test_fanout_interrupt_does_not_double_apply():
    # seed -> {A, B} in one superstep. 'hits' uses the "add" reducer.
    # B interrupts; A must contribute its +1 exactly once across pause+resume.
    def seed(state, ctx):
        return {}

    def node_a(state, ctx):
        return {"hits": 1}

    def node_b(state, ctx):
        decision = ctx.interrupt({"need": "ok"})
        return {"hits": 10, "approved": decision}

    g = StateGraph(channels={"hits": Channel("add", default=0)})
    g.add_node("seed", seed)
    g.add_node("A", node_a)
    g.add_node("B", node_b)
    g.add_edge(START, "seed")
    g.add_edge("seed", "A")
    g.add_edge("seed", "B")
    g.set_finish_point("A")
    g.set_finish_point("B")

    app = g.compile(checkpointer=MemorySaver())
    paused = await app.ainvoke({}, thread_id="t")
    assert paused.interrupted

    done = await app.ainvoke(thread_id="t", resume=True)
    # A=+1 (once) and B=+10 (once) -> 11, NOT 12 (A double-applied).
    assert done.state["hits"] == 11, done.state["hits"]
    assert done.state["approved"] is True


@pytest.mark.asyncio
async def test_fanout_interrupt_preserves_executed_successors():
    # A already-run node's successor must still execute after resume.
    # seed -> {A, B}; A -> C (a follow-on node). B interrupts. C must still run.
    def seed(state, ctx):
        return {}

    def node_a(state, ctx):
        return {"a_ran": True}

    def node_c(state, ctx):
        return {"c_ran": True}

    def node_b(state, ctx):
        ctx.interrupt({"need": "ok"})
        return {"b_ran": True}

    g = StateGraph()
    g.add_node("seed", seed)
    g.add_node("A", node_a)
    g.add_node("B", node_b)
    g.add_node("C", node_c)
    g.add_edge(START, "seed")
    g.add_edge("seed", "A")
    g.add_edge("seed", "B")
    g.add_edge("A", "C")
    g.set_finish_point("B")
    g.set_finish_point("C")

    app = g.compile(checkpointer=MemorySaver())
    paused = await app.ainvoke({}, thread_id="t2")
    assert paused.interrupted
    done = await app.ainvoke(thread_id="t2", resume=True)
    # C (successor of the already-run A) must not be lost across the interrupt.
    assert done.state.get("c_ran") is True, done.state
    assert done.state.get("b_ran") is True, done.state
