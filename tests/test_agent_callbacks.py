"""Per-agent entry/exit callbacks that fire even inside a composition.

``before_agent`` / ``after_agent`` run around an agent's own loop, so they fire
once per agent whether it runs standalone or as a child of a workflow agent
(where children call ``agent.run()`` directly and bypass the parent Runner's
plugin chain). Both sync and async callables are accepted.
"""

from __future__ import annotations

import pytest

from yaab import Agent, SequentialAgent
from yaab.testing import TestModel


@pytest.mark.asyncio
async def test_before_and_after_fire_once_standalone():
    events: list[str] = []
    agent = Agent(
        "a",
        model=TestModel("hi"),
        before_agent=lambda ag, prompt: events.append(f"before:{ag.name}:{prompt}"),
        after_agent=lambda ag, result: events.append(f"after:{ag.name}:{result.output}"),
    )
    await agent.run("go")
    assert events == ["before:a:go", "after:a:hi"]


@pytest.mark.asyncio
async def test_async_callbacks_supported():
    events: list[str] = []

    async def before(ag, prompt):
        events.append("before")

    async def after(ag, result):
        events.append("after")

    agent = Agent("a", model=TestModel("x"), before_agent=before, after_agent=after)
    await agent.run("go")
    assert events == ["before", "after"]


@pytest.mark.asyncio
async def test_callbacks_fire_per_child_in_a_workflow():
    # The whole point: workflow children call agent.run() directly, so their own
    # before/after callbacks must still fire (the parent Runner's plugins don't).
    seen: list[str] = []
    a = Agent("a", model=TestModel("A"), before_agent=lambda ag, p: seen.append("a"))
    b = Agent("b", model=TestModel("B"), before_agent=lambda ag, p: seen.append("b"))
    await SequentialAgent("pipe", [a, b]).run("go")
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_before_agent_can_observe_then_run_proceeds():
    calls = {"n": 0}
    agent = Agent(
        "a",
        model=TestModel("done"),
        before_agent=lambda ag, p: calls.__setitem__("n", calls["n"] + 1),
    )
    result = await agent.run("go")
    assert calls["n"] == 1
    assert result.output == "done"
