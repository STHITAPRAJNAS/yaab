"""Flow as the seventh workflow pattern — end-to-end over the Runner.

These exercise the user-facing surface: a Flow is a ``_WorkflowBase`` that
threads the ONE shared State, captures step output with ``writes=``, routes with
a Part 2 ``Condition``, rolls usage up across agent-steps, and nests as a tool /
inside another workflow agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from yaab import Agent, Flow, RunContext, SequentialAgent, State
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import ToolCall


@dataclass
class ReviewDeps:
    threshold: int = 100


def test_simple_pipeline_writes_capture_and_returns():
    drafter = Agent("drafter", model=TestModel("a draft"), writes="draft")
    polisher = Agent("polisher", model=TestModel("the final"), writes="final")

    flow: Flow[ReviewDeps, str] = (
        Flow[ReviewDeps, str]("reply_pipeline")
        .step("draft", agent=drafter)
        .step("polish", agent=polisher)
        .start_at("draft")
        .then("draft", "polish")
        .then("polish", Flow.DONE)
        .returns("final")
    )
    result = flow.run_sync("refund #42", deps=ReviewDeps())
    assert result.output == "the final"
    # Both sub-agents' usage rolled up.
    assert result.usage.requests >= 2


def test_step_level_writes_overrides_agent_writes():
    drafter = Agent("drafter", model=TestModel("draft text"), writes="draft")
    flow = (
        Flow[None, str]("ov")
        .step("draft", agent=drafter, writes="override_key")
        .start_at("draft")
        .returns("override_key")
    )
    result = flow.run_sync("go")
    assert result.output == "draft text"


def test_function_step_reads_and_writes_one_state():
    def parse(state: State, ctx: RunContext) -> dict:
        return {"amount": 250}

    flow = Flow[None, int]("parse_flow").step("parse", fn=parse).start_at("parse").returns("amount")
    result = flow.run_sync("the amount is 250")
    assert result.output == 250


def test_conditional_route_twelve_line_example():
    # The runnable 3-step flow with a conditional route.
    def parse(state, ctx):
        return {"amount": 50}

    flow = (
        Flow[ReviewDeps, str]("refund_router")
        .step("parse", fn=parse)
        .route(
            "parse",
            lambda state, ctx: "human" if state["amount"] >= ctx.deps.threshold else "auto",
            to={"auto": "execute", "human": "await_approval"},
        )
        .step("execute", fn=lambda state, ctx: {"done": "auto-executed"})
        .step("await_approval", fn=lambda state, ctx: {"done": "needs-human"})
        .then("execute", Flow.DONE)
        .then("await_approval", Flow.DONE)
        .start_at("parse")
        .returns("done")
    )
    result = flow.run_sync("refund", deps=ReviewDeps(threshold=100))
    assert result.output == "auto-executed"  # 50 < 100 -> auto branch


def test_route_picker_branches_to_human_path():
    def parse(state, ctx):
        return {"amount": 5000}

    flow = (
        Flow[ReviewDeps, str]("refund_router")
        .step("parse", fn=parse)
        .route(
            "parse",
            lambda state, ctx: "human" if state["amount"] >= ctx.deps.threshold else "auto",
            to={"auto": "execute", "human": "await_approval"},
        )
        .step("execute", fn=lambda state, ctx: {"done": "auto-executed"})
        .step("await_approval", fn=lambda state, ctx: {"done": "needs-human"})
        .then("execute", Flow.DONE)
        .then("await_approval", Flow.DONE)
        .start_at("parse")
        .returns("done")
    )
    result = flow.run_sync("refund", deps=ReviewDeps(threshold=100))
    assert result.output == "needs-human"  # 5000 >= 100 -> human branch


def test_route_unknown_label_rejected_at_build_time():
    flow = Flow[None, str]("bad").step("a", fn=lambda s, c: {})
    with pytest.raises(ValueError):
        # The picker can only return labels present in ``to`` — an unknown one in
        # ``to`` referencing a non-existent step is fine, but a picker target not
        # in ``to`` is validated at build via the route's mapping completeness.
        flow.route("a", lambda s, c: "zzz", to={}).start_at("a").lower()


def test_loop_accumulates_on_one_state():
    def refine(state, ctx):
        return {"score": state.get("score", 0) + 1}

    flow = (
        Flow[None, int]("refine")
        .step("refine", fn=refine)
        .loop("refine", until="state.score >= 3", max_iterations=10)
        .start_at("refine")
        .returns("score")
    )
    result = flow.run_sync("seed")
    assert result.output == 3


def test_flow_as_tool_nested_in_agent():
    drafter = Agent("drafter", model=TestModel("drafted"), writes="draft")
    flow = (
        Flow[None, str]("process").step("draft", agent=drafter).start_at("draft").returns("draft")
    )
    tool = flow.as_tool(name="process_refund")
    assert tool.name == "process_refund"

    supervisor = Agent(
        "ops",
        model=TestModel(
            responses=[
                ModelResponse(
                    tool_calls=[ToolCall(name="process_refund", arguments={"prompt": "go"})],
                    finish_reason="tool_calls",
                ),
                "supervised done",
            ]
        ),
        tools=[tool],
    )
    result = supervisor.run_sync("handle refund")
    assert result.output == "supervised done"


def test_flow_nested_in_sequential_shares_state():
    classifier = Agent("classifier", model=TestModel("refund"), writes="intent")

    def use_intent(state, ctx):
        # The Flow shares the parent's State, so the classifier's writes= is here.
        return {"final": f"handling: {state['intent']}"}

    flow = Flow[None, str]("inner").step("use", fn=use_intent).start_at("use").returns("final")
    pipeline = SequentialAgent("intake", [classifier, flow], pipe_output=False)
    result = pipeline.run_sync("a refund please")
    # The Flow read state["intent"] that the classifier wrote — one shared State.
    assert "refund" in str(result.output)


def test_flow_is_seventh_workflow_pattern_in_docstring():
    import yaab.multiagent as ma

    assert "Flow" in (ma.__doc__ or "")
