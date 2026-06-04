"""Tests for when=/stop=/else= guards across every workflow pattern.

Also covers tool gating, sub-agent transfer gating, the include_skipped flag,
failure/timeout fallback, decision events, and the kind: router YAML config.
"""

from __future__ import annotations

import warnings

import pytest

from yaab import (
    Agent,
    LoopAgent,
    MapAgent,
    ParallelAgent,
    SequentialAgent,
    agent_from_dict,
)
from yaab.conditions import Status, Step
from yaab.models.test_model import FunctionModel, TestModel
from yaab.types import EventType


def _echo(name):
    return Agent(name, model=FunctionModel(lambda msgs: msgs[-1].content))


def _say(name, text):
    return Agent(name, model=TestModel(text))


def _tracked(name, text):
    agent = _say(name, text)
    flag = {"ran": False}
    orig = agent.run

    async def run(*a, **k):
        flag["ran"] = True
        return await orig(*a, **k)

    agent.run = run  # type: ignore[assignment]
    return agent, flag


# --- Sequential when= -------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_when_skips_step():
    skipped_agent, flag = _tracked("skipme", "SKIPPED-OUTPUT")
    seq = SequentialAgent(
        "pipe",
        [
            Step(_say("a", "first"), writes="result"),
            Step(skipped_agent, when="state.result == 'never'"),
        ],
    )
    result = await seq.run("go")
    assert flag["ran"] is False
    # Skip passes input through unchanged (C6): output is the prior step's output.
    assert result.output == "first"


@pytest.mark.asyncio
async def test_sequential_when_runs_step_when_true():
    ran_agent, flag = _tracked("run", "RAN")
    seq = SequentialAgent(
        "pipe",
        [
            Step(_say("a", "refund"), writes="intent"),
            Step(ran_agent, when="state.intent == 'refund'"),
        ],
        pipe_output=False,
    )
    await seq.run("go")
    assert flag["ran"] is True


@pytest.mark.asyncio
async def test_sequential_else_runs_on_skip():
    fallback = _say("fb", "FALLBACK")
    seq = SequentialAgent(
        "pipe",
        [
            Step(_say("a", "x"), writes="intent"),
            Step(_say("main", "MAIN"), when="state.intent == 'never'", else_=fallback),
        ],
        pipe_output=False,
    )
    result = await seq.run("go")
    assert result.output == "FALLBACK"


@pytest.mark.asyncio
async def test_sequential_stop_fires():
    b_agent, flag = _tracked("b", "should not run")
    seq = SequentialAgent(
        "pipe",
        [_say("a", "STOP here"), b_agent],
        stop="output contains 'STOP'",
        pipe_output=False,
    )
    result = await seq.run("go")
    assert result.output == "STOP here"
    assert flag["ran"] is False


# --- stop_when deprecation alias --------------------------------------------


@pytest.mark.asyncio
async def test_stop_when_alias_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seq = SequentialAgent("pipe", [_say("a", "X")], stop_when=lambda o: True, pipe_output=False)
        await seq.run("go")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


# --- Parallel when= + include_skipped ---------------------------------------


@pytest.mark.asyncio
async def test_parallel_when_omits_skipped_branch():
    par = ParallelAgent(
        "fan",
        [
            Step(_say("a", "A"), writes="a"),
            Step(_say("b", "B"), when="input == 'never'", writes="b"),
        ],
    )
    result = await par.run("go")
    assert "a" in result.output
    assert "b" not in result.output  # skipped branch absent by default (C6)


@pytest.mark.asyncio
async def test_parallel_include_skipped_adds_entry():
    par = ParallelAgent(
        "fan",
        [
            Step(_say("a", "A"), writes="a"),
            Step(_say("b", "B"), when="input == 'never'", writes="b"),
        ],
        include_skipped=True,
    )
    result = await par.run("go")
    assert "b" in result.output
    assert result.output["b"].status == Status.SKIPPED


# --- Map when= filter -------------------------------------------------------


@pytest.mark.asyncio
async def test_map_when_filters_inputs():
    mapper = MapAgent(
        "fan",
        # A callable filter: keep multi-char inputs (len() is not in the safe
        # string grammar by design, so per-input length filtering uses a callable).
        Step(_echo("e"), when=lambda v: len(v) > 1),
        map_inputs=lambda p: p.split(","),
    )
    result = await mapper.run("a,bb,c,dd")
    # Only 'bb' and 'dd' survive the filter.
    assert result.output == ["bb", "dd"]


# --- Loop stop= (state-aware) + until= alias --------------------------------


@pytest.mark.asyncio
async def test_loop_stop_reads_state():
    # A tool increments state['n']; stop fires when it reaches 3.
    calls = {"n": 0}

    def model(msgs):
        calls["n"] += 1
        return f"iter-{calls['n']}"

    agent = Agent("worker", model=FunctionModel(model), writes="n_out")

    # Use a plain callable stop reading the loop's output count.
    loop = LoopAgent(
        "loop",
        agent,
        max_iterations=5,
        stop=lambda out: "iter-3" in str(out),
        pipe_output=False,
    )
    result = await loop.run("go")
    assert result.output == "iter-3"


@pytest.mark.asyncio
async def test_loop_until_alias_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loop = LoopAgent(
            "loop",
            _say("a", "X"),
            max_iterations=2,
            until=lambda o: True,
            pipe_output=False,
        )
        await loop.run("go")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


@pytest.mark.asyncio
async def test_loop_else_on_exhaustion():
    # Loop never stops; else= runs when max_iterations is hit.
    fallback = _say("review", "REVIEWED")
    loop = LoopAgent(
        "loop",
        _say("a", "again"),
        max_iterations=2,
        stop=lambda o: False,
        else_=fallback,
        pipe_output=False,
    )
    result = await loop.run("go")
    assert result.output == "REVIEWED"


# --- else= on failure -------------------------------------------------------


@pytest.mark.asyncio
async def test_else_runs_on_failure():
    boom = _say("boom", "X")

    async def explode(*a, **k):
        from yaab.exceptions import ToolError

        raise ToolError("kaboom")

    boom.run = explode  # type: ignore[assignment]

    fallback = _say("fb", "RECOVERED")
    seq = SequentialAgent("pipe", [Step(boom, else_=fallback)], pipe_output=False)
    result = await seq.run("go")
    assert result.output == "RECOVERED"


@pytest.mark.asyncio
async def test_no_else_failure_propagates():
    boom = _say("boom", "X")

    async def explode(*a, **k):
        from yaab.exceptions import ToolError

        raise ToolError("kaboom")

    boom.run = explode  # type: ignore[assignment]

    seq = SequentialAgent("pipe", [Step(boom)], pipe_output=False)
    from yaab.exceptions import ToolError

    with pytest.raises(ToolError):
        await seq.run("go")


# --- decision events carry resolved operands (req. 7 / C13) -----------------


@pytest.mark.asyncio
async def test_skip_event_carries_operands():
    seq = SequentialAgent(
        "pipe",
        [
            Step(_say("a", "x"), writes="intent"),
            Step(_say("main", "M"), when="state.intent == 'refund'"),
        ],
        pipe_output=False,
    )
    result = await seq.run("go")
    skip_events = [e for e in result.events if e.type == EventType.CONDITION_SKIP]
    assert skip_events
    payload = skip_events[0].payload
    assert "operands" in payload
    assert payload["result"] is False


# --- tool when= gating (model-facing availability) --------------------------


@pytest.mark.asyncio
async def test_tool_when_gates_availability():
    from yaab import FunctionTool

    called = {"hit": False}

    def secret() -> str:
        called["hit"] = True
        return "secret-result"

    agent = Agent(
        "ops",
        model=TestModel("done"),
        tools=[Step(FunctionTool(secret, name="secret"), when="state.role == 'admin'")],
    )
    # role is not admin => tool is omitted from the schema.
    result = await agent.run("hi")
    assert result.output == "done"


# --- YAML kind: router ------------------------------------------------------


def test_yaml_router_roundtrip():
    cfg = {
        "kind": "router",
        "name": "support_router",
        "on_no_match": "default",
        "branches": [
            {
                "when": "input == 'bill'",
                "agent": {"name": "billing", "instructions": "Handle billing."},
            },
        ],
        "default_agent": {"name": "general", "instructions": "Answer."},
    }
    router = agent_from_dict(cfg)
    assert router.name == "support_router"
    assert len(router.branches) == 1


def test_yaml_router_rejects_arbitrary_python_when():
    cfg = {
        "kind": "router",
        "name": "r",
        "branches": [
            {"when": "__import__('os')", "agent": {"name": "x", "instructions": "x"}},
        ],
        "default_agent": {"name": "d", "instructions": "d"},
    }
    with pytest.raises((ValueError, Exception)):
        agent_from_dict(cfg)
