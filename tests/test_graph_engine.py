"""Tests for the selectable graph engine (python vs rust superstep advancement)."""

from __future__ import annotations

import pytest

from yaab import _core
from yaab.exceptions import YaabError
from yaab.graph import END, START, Channel, MemorySaver, StateGraph


def _counter_graph():
    g = StateGraph(channels={"count": Channel("add", default=0)})
    g.add_node("inc", lambda s: {"count": 1})
    g.add_edge(START, "inc")
    g.add_conditional_edges(
        "inc", lambda s: "inc" if s["count"] < 3 else END, {"inc": "inc", END: END}
    )
    return g


def _fanout_graph():
    g = StateGraph(channels={"hits": Channel("append", default=[])})
    g.add_node("fan", lambda s: {})
    g.add_node("left", lambda s: {"hits": "L"})
    g.add_node("right", lambda s: {"hits": "R"})
    g.add_edge(START, "fan")
    g.add_edge("fan", "left")
    g.add_edge("fan", "right")
    g.set_finish_point("left")
    g.set_finish_point("right")
    return g


def test_engine_auto_resolves():
    app = _counter_graph().compile()
    assert app.engine == ("rust" if _core.RUST else "python")


def test_explicit_python_engine():
    app = _counter_graph().compile(engine="python")
    assert app.engine == "python"
    assert app.invoke({}).state["count"] == 3


def test_explicit_rust_engine_when_available():
    if not _core.RUST:
        pytest.skip("rust core not built")
    app = _counter_graph().compile(engine="rust")
    assert app.engine == "rust"
    assert app.invoke({}).state["count"] == 3


def test_rust_engine_requires_extension():
    if _core.RUST:
        pytest.skip("rust core is present; cannot test the missing-extension error")
    with pytest.raises(YaabError):
        _counter_graph().compile(engine="rust")


def test_unknown_engine_rejected():
    with pytest.raises(YaabError):
        _counter_graph().compile(engine="java")


def test_both_engines_agree_cyclic():
    py = _counter_graph().compile(engine="python").invoke({}).state
    if _core.RUST:
        rs = _counter_graph().compile(engine="rust").invoke({}).state
        assert py == rs
    assert py["count"] == 3


def test_both_engines_agree_fanout():
    py = _fanout_graph().compile(engine="python").invoke({}).state
    assert sorted(py["hits"]) == ["L", "R"]
    if _core.RUST:
        rs = _fanout_graph().compile(engine="rust").invoke({}).state
        assert sorted(rs["hits"]) == ["L", "R"]


def test_rust_engine_hitl_still_works():
    if not _core.RUST:
        pytest.skip("rust core not built")

    def gate(state, ctx):
        decision = ctx.interrupt({"need": "ok"})
        return {"approved": decision}

    g = StateGraph()
    g.add_node("gate", gate)
    g.add_edge(START, "gate")
    g.set_finish_point("gate")
    app = g.compile(checkpointer=MemorySaver(), engine="rust")

    paused = app.invoke({}, thread_id="t1")
    assert paused.interrupted
    resumed = app.invoke(thread_id="t1", resume=True)
    assert resumed.state["approved"] is True
