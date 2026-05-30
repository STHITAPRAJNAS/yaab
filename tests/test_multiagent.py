"""Tests for multi-agent workflow patterns."""

from __future__ import annotations

import pytest

from yaab import Agent, LoopAgent, ParallelAgent, SequentialAgent, Swarm
from yaab.multiagent import SwarmState
from yaab.testing import TestModel


@pytest.mark.asyncio
async def test_sequential_pipes_output():
    a = Agent("a", model=TestModel("step-a"))
    b = Agent("b", model=TestModel("step-b"))
    seq = SequentialAgent("pipe", [a, b])
    result = await seq.run("start")
    assert result.output == "step-b"
    assert result.usage.requests == 2


@pytest.mark.asyncio
async def test_parallel_returns_map():
    a = Agent("a", model=TestModel("ans-a"))
    b = Agent("b", model=TestModel("ans-b"))
    par = ParallelAgent("fan", [a, b])
    result = await par.run("q")
    assert result.output == {"a": "ans-a", "b": "ans-b"}
    assert result.usage.requests == 2


@pytest.mark.asyncio
async def test_loop_stops_on_condition():
    calls = {"n": 0}

    def make_response(messages):
        calls["n"] += 1
        return "done" if calls["n"] >= 2 else "keep going"

    from yaab.models.test_model import FunctionModel

    agent = Agent("a", model=FunctionModel(make_response))
    loop = LoopAgent("loop", agent, max_iterations=5, until=lambda out: out == "done")
    result = await loop.run("go")
    assert result.output == "done"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_swarm_handoff():
    # Triage hands off to specialist on its first turn.
    triage_model = TestModel(custom_output="routed", call_tools=["handoff_to_specialist"])
    triage = Agent("triage", model=triage_model)
    specialist = Agent("specialist", model=TestModel("specialist answer"))

    swarm = Swarm("support", [triage, specialist], entry="triage")
    result = await swarm.run("I need help", deps=SwarmState())
    assert result.output == "specialist answer"


@pytest.mark.asyncio
async def test_workflow_agent_as_tool():
    a = Agent("a", model=TestModel("inner"))
    seq = SequentialAgent("seq", [a])
    # A workflow agent can be exposed as a tool to a parent agent.
    parent_model = TestModel(custom_output="parent-done", call_tools=["call_seq"])
    parent = Agent("parent", model=parent_model, tools=[seq.as_tool()])
    result = await parent.run("delegate")
    assert result.output == "parent-done"
