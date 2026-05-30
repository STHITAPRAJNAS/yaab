"""Tests for the durable graph engine."""

from __future__ import annotations

from yaab.graph import END, START, Channel, MemorySaver, StateGraph


def test_linear_graph():
    g = StateGraph()
    g.add_node("a", lambda s: {"x": 1})
    g.add_node("b", lambda s: {"y": s["x"] + 1})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.set_finish_point("b")
    result = g.compile().invoke({})
    assert result.state["x"] == 1
    assert result.state["y"] == 2
    assert not result.interrupted


def test_cyclic_graph_with_reducer():
    g = StateGraph(channels={"count": Channel("add", default=0)})
    g.add_node("inc", lambda s: {"count": 1})
    g.add_edge(START, "inc")
    g.add_conditional_edges(
        "inc", lambda s: "inc" if s["count"] < 3 else END, {"inc": "inc", END: END}
    )
    result = g.compile().invoke({})
    assert result.state["count"] == 3


def test_parallel_superstep():
    g = StateGraph(channels={"hits": Channel("append", default=[])})
    g.add_node("fan", lambda s: {})
    g.add_node("left", lambda s: {"hits": "L"})
    g.add_node("right", lambda s: {"hits": "R"})
    g.add_edge(START, "fan")
    g.add_edge("fan", "left")
    g.add_edge("fan", "right")
    g.set_finish_point("left")
    g.set_finish_point("right")
    result = g.compile().invoke({})
    assert sorted(result.state["hits"]) == ["L", "R"]


def test_human_in_the_loop_interrupt_and_resume():
    def gate(state, ctx):
        decision = ctx.interrupt({"need": "approval"})
        return {"approved": decision}

    g = StateGraph()
    g.add_node("gate", gate)
    g.add_edge(START, "gate")
    g.set_finish_point("gate")
    app = g.compile(checkpointer=MemorySaver())

    first = app.invoke({}, thread_id="t1")
    assert first.interrupted is True
    assert first.interrupt_value == {"need": "approval"}

    resumed = app.invoke(thread_id="t1", resume=True)
    assert resumed.interrupted is False
    assert resumed.state["approved"] is True


def test_checkpoint_history_time_travel():
    saver = MemorySaver()
    g = StateGraph(channels={"count": Channel("add", default=0)})
    g.add_node("inc", lambda s: {"count": 1})
    g.add_edge(START, "inc")
    g.add_conditional_edges(
        "inc", lambda s: "inc" if s["count"] < 3 else END, {"inc": "inc", END: END}
    )
    g.compile(checkpointer=saver).invoke({}, thread_id="th")
    history = saver.history("th")
    assert len(history) >= 3  # one checkpoint per superstep
