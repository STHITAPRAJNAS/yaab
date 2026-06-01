"""Run lifecycle control (the #1 cross-framework ask): cancel a running agent,
enforce a wall-clock budget, and reset an agent for reuse.
"""

from __future__ import annotations

import asyncio

import pytest

from yaab import Agent, UsageLimits, tool
from yaab.exceptions import RunCancelled, UsageLimitExceeded
from yaab.limits import CancellationToken
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import ToolCall


def _loop_model(tool_name: str, n: int) -> TestModel:
    """A model that asks for the tool n times, then answers 'done'."""
    calls = [
        ModelResponse(
            tool_calls=[ToolCall(name=tool_name, arguments={})], finish_reason="tool_calls"
        )
        for _ in range(n)
    ]
    return TestModel(responses=[*calls, "done"])


# --- external mid-run cancellation -------------------------------------
@pytest.mark.asyncio
async def test_external_cancel_stops_run():
    token = CancellationToken()

    @tool
    def step() -> str:
        """one step"""
        token.cancel("user_stop")  # simulate an external cancel mid-run
        return "ok"

    agent = Agent("a", model=_loop_model("step", 3), tools=[step])
    with pytest.raises(RunCancelled):
        await agent.run("go", cancellation=token)


@pytest.mark.asyncio
async def test_external_cancel_stops_stream_events():
    from yaab import EventType

    token = CancellationToken()

    @tool
    def step() -> str:
        """one step"""
        token.cancel()
        return "ok"

    agent = Agent("a", model=_loop_model("step", 3), tools=[step])
    saw_error = False
    async for e in agent.stream_events("go", cancellation=token):
        if e.type is EventType.ERROR:
            saw_error = isinstance(e.payload["error"], RunCancelled)
    assert saw_error


@pytest.mark.asyncio
async def test_cancel_from_concurrent_task():
    started = asyncio.Event()
    release = asyncio.Event()
    token = CancellationToken()

    @tool
    async def waits() -> str:
        """waits for release"""
        started.set()
        await release.wait()
        return "ok"

    agent = Agent("a", model=_loop_model("waits", 3), tools=[waits])

    async def canceller():
        await started.wait()
        token.cancel("stop")
        release.set()  # let the tool finish; next step check raises

    with pytest.raises(RunCancelled):
        await asyncio.gather(agent.run("go", cancellation=token), canceller())


# --- wall-clock budget (max_wall_seconds was previously dead) ----------
@pytest.mark.asyncio
async def test_max_wall_seconds_enforced():
    @tool
    async def slow() -> str:
        """slow"""
        await asyncio.sleep(0.06)
        return "x"

    agent = Agent("a", model=_loop_model("slow", 5), tools=[slow])
    with pytest.raises(UsageLimitExceeded) as exc:
        await agent.run("go", usage_limits=UsageLimits(max_wall_seconds=0.05))
    assert exc.value.limit == "wall_seconds"


@pytest.mark.asyncio
async def test_wall_seconds_not_tripped_when_fast():
    agent = Agent("a", model=TestModel("done"))
    r = await agent.run("go", usage_limits=UsageLimits(max_wall_seconds=10))
    assert r.output == "done"


# --- Agent.reset() -----------------------------------------------------
@pytest.mark.asyncio
async def test_agent_reset_clears_model_cache():
    agent = Agent("a", model=TestModel("one"))
    _ = agent.model  # resolve + cache
    assert agent._model is not None
    out = agent.reset()
    assert agent._model is None
    assert out is agent  # chainable
