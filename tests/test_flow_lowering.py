"""Flow lowers onto the graph engine — the builder is a thin compiler.

These tests pin the *structural* contract: each builder method lowers onto the
exact engine primitive the design names (``.step`` -> ``add_node``, ``.then`` ->
``add_edge``, ``.route`` -> ``add_conditional_edges``, ``.loop`` -> a cyclic edge
plus a conditional exit). They run the lowered graph through the SAME engine the
rest of the SDK uses, with plain function steps (no model), so they prove the
compilation without any Runner delegation.
"""

from __future__ import annotations

import pytest

from yaab import Flow, State
from yaab.graph.state import END, START, CompiledGraph, StateGraph


def test_step_then_lowers_to_nodes_and_edges():
    flow = (
        Flow[None, int]("pipe")
        .step("a", fn=lambda state, ctx: {"x": 1})
        .step("b", fn=lambda state, ctx: {"y": state["x"] + 1})
        .start_at("a")
        .then("a", "b")
        .then("b", Flow.DONE)
        .returns("y")
    )
    graph = flow.lower()
    assert isinstance(graph, StateGraph)
    # Steps became nodes.
    assert set(graph.nodes) >= {"a", "b"}
    # .then became an edge; .start_at set the entry.
    assert "b" in graph.edges["a"]
    assert graph.entry == "a"
    # .then(..., Flow.DONE) routes to END (the terminal marker).
    assert END in graph.edges["b"]


def test_compiled_flow_runs_through_the_engine():
    flow = (
        Flow[None, int]("pipe")
        .step("a", fn=lambda state, ctx: {"x": 1})
        .step("b", fn=lambda state, ctx: {"y": state["x"] + 1})
        .start_at("a")
        .then("a", "b")
        .then("b", Flow.DONE)
        .returns("y")
    )
    compiled = flow.lower().compile()
    assert isinstance(compiled, CompiledGraph)
    result = compiled.invoke({})
    assert result.state["x"] == 1
    assert result.state["y"] == 2


def test_route_lowers_to_conditional_edges():
    flow = (
        Flow[None, str]("router")
        .step("parse", fn=lambda state, ctx: {"amount": 50})
        .route(
            "parse",
            lambda state, ctx: "high" if state["amount"] >= 100 else "low",
            to={"high": "big", "low": "small"},
        )
        .step("big", fn=lambda state, ctx: {"path": "big"})
        .step("small", fn=lambda state, ctx: {"path": "small"})
        .then("big", Flow.DONE)
        .then("small", Flow.DONE)
        .start_at("parse")
        .returns("path")
    )
    graph = flow.lower()
    assert "parse" in graph.conditional
    _router, mapping = graph.conditional["parse"]
    assert mapping == {"high": "big", "low": "small"}
    result = flow.lower().compile().invoke({})
    assert result.state["path"] == "small"


def test_route_picker_receives_readonly_state_and_cannot_mutate():
    seen: dict[str, object] = {}

    def picker(state, ctx):
        # A picker reads state; the lowering hands it a read-only view.
        seen["type"] = type(state).__name__
        return "done" if state["n"] == 1 else "loop"

    flow = (
        Flow[None, int]("ro")
        .step("seed", fn=lambda state, ctx: {"n": 1})
        .route("seed", picker, to={"done": Flow.DONE, "loop": "seed"})
        .start_at("seed")
        .returns("n")
    )
    flow.lower().compile().invoke({})
    assert seen["type"] == "ReadonlyState"


def test_route_picker_that_mutates_raises_typeerror():
    def bad_picker(state, ctx):
        state["sneaky"] = True  # the read-only view forbids this
        return "done"

    flow = (
        Flow[None, int]("ro")
        .step("seed", fn=lambda state, ctx: {"n": 1})
        .route("seed", bad_picker, to={"done": Flow.DONE})
        .start_at("seed")
        .returns("n")
    )
    with pytest.raises(TypeError):
        flow.lower().compile().invoke({})


def test_loop_lowers_to_cycle_plus_conditional_exit():
    def refine(state, ctx):
        score = state.get("score", 0) + 1
        return {"score": score}

    flow = (
        Flow[None, int]("refine")
        .step("refine", fn=refine)
        .loop("refine", until=lambda state, ctx: state["score"] >= 3, max_iterations=10)
        .start_at("refine")
        .returns("score")
    )
    graph = flow.lower()
    # The loop step has a conditional exit (the until picker).
    assert "refine" in graph.conditional
    result = flow.lower().compile().invoke({})
    # Accumulates on the ONE state across iterations until until= fires.
    assert result.state["score"] == 3


def test_loop_respects_max_iterations_cap():
    def refine(state, ctx):
        return {"score": state.get("score", 0) + 1}

    flow = (
        Flow[None, int]("refine")
        .step("refine", fn=refine)
        .loop("refine", until=lambda state, ctx: state["score"] >= 100, max_iterations=4)
        .start_at("refine")
        .returns("score")
    )
    result = flow.lower().compile().invoke({})
    # until= never fires; the cap stops the cycle at max_iterations.
    assert result.state["score"] == 4


def test_returns_and_done_entry_markers():
    assert Flow.DONE == END
    assert Flow.ENTRY == START


def test_when_guard_lowers_to_skip_sink():
    ran: list[str] = []

    def must_run(state, ctx):
        ran.append("yes")
        return {"did": True}

    flow = (
        Flow[None, bool]("guarded")
        .step("seed", fn=lambda state, ctx: {"intent": "other"})
        .step("guarded", fn=must_run, when="state.intent == 'refund'")
        .start_at("seed")
        .then("seed", "guarded")
        .then("guarded", Flow.DONE)
        .returns("did")
    )
    result = flow.lower().compile().invoke({})
    # The guard is false, so the guarded step is skipped (its body never ran).
    assert ran == []
    assert "did" not in result.state


def test_node_returning_dict_folds_into_state():
    # A plain step body that returns a dict updates the shared State via the
    # engine fold (FL8) — no positional assembly, keys are explicit.
    flow = (
        Flow[None, dict]("folds")
        .step("a", fn=lambda state, ctx: {"k1": "v1", "k2": 2})
        .start_at("a")
        .returns("k2")
    )
    result = flow.lower().compile().invoke({})
    assert result.state["k1"] == "v1"
    assert result.state["k2"] == 2


def test_step_state_view_is_the_one_state():
    captured: dict[str, object] = {}

    def step(state, ctx):
        captured["state_type"] = type(state).__name__
        # temp: writes route to the temp store on the same State object.
        state["temp:scratch"] = "x"
        return {"out": 1}

    flow = Flow[None, int]("one_state").step("a", fn=step).start_at("a").returns("out")
    flow.lower().compile().invoke({})
    assert captured["state_type"] == State.__name__
