"""Parallel tool execution + per-tool timeout (Phase A1 + A2).

Concurrency is proven deterministically with an asyncio.Barrier: two tools that
each wait on a 2-party barrier can only both complete if they run concurrently;
under sequential execution the first would block forever.
"""

from __future__ import annotations

import asyncio

import pytest

from yaab import Agent, tool
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import EventType, ToolCall


def _two_tool_then_done(a: str, b: str) -> TestModel:
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[
                    ToolCall(name=a, arguments={}),
                    ToolCall(name=b, arguments={}),
                ],
                finish_reason="tool_calls",
            ),
            "done",
        ]
    )


@pytest.mark.asyncio
async def test_tools_run_concurrently():
    barrier = asyncio.Barrier(2)

    @tool
    async def alpha() -> str:
        """alpha"""
        await barrier.wait()  # only proceeds if beta also arrives -> concurrent
        return "A"

    @tool
    async def beta() -> str:
        """beta"""
        await barrier.wait()
        return "B"

    agent = Agent("a", model=_two_tool_then_done("alpha", "beta"), tools=[alpha, beta])
    # If tools ran sequentially, alpha would wait on the barrier forever.
    result = await asyncio.wait_for(agent.run("go"), timeout=5)
    assert result.output == "done"


@pytest.mark.asyncio
async def test_parallel_preserves_event_and_result_order():
    @tool
    async def first() -> str:
        """first"""
        await asyncio.sleep(0.02)  # finishes AFTER second despite being first
        return "first-result"

    @tool
    async def second() -> str:
        """second"""
        return "second-result"

    agent = Agent("a", model=_two_tool_then_done("first", "second"), tools=[first, second])
    events = [e async for e in agent._get_runner().run_stream(agent, "go")]
    calls = [e.payload["name"] for e in events if e.type is EventType.TOOL_CALL]
    results = [e.payload["name"] for e in events if e.type is EventType.TOOL_RESULT]
    # Model's call order is preserved in the event stream regardless of finish order.
    assert calls == ["first", "second"]
    assert results == ["first", "second"]


@pytest.mark.asyncio
async def test_parallel_tools_opt_out_is_sequential():
    order = []

    @tool
    async def one() -> str:
        """one"""
        await asyncio.sleep(0.02)
        order.append("one")
        return "1"

    @tool
    async def two() -> str:
        """two"""
        order.append("two")
        return "2"

    agent = Agent(
        "a", model=_two_tool_then_done("one", "two"), tools=[one, two], parallel_tools=False
    )
    result = await agent.run("go")
    assert result.output == "done"
    # Sequential: 'one' completes (and appends) before 'two' starts.
    assert order == ["one", "two"]


@pytest.mark.asyncio
async def test_max_parallel_tools_bounds_concurrency():
    active = {"now": 0, "peak": 0}

    @tool
    async def worker_a() -> str:
        """a"""
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return "a"

    @tool
    async def worker_b() -> str:
        """b"""
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return "b"

    agent = Agent(
        "a",
        model=_two_tool_then_done("worker_a", "worker_b"),
        tools=[worker_a, worker_b],
        max_parallel_tools=1,
    )
    await agent.run("go")
    assert active["peak"] == 1  # capped at one at a time


# --- A2: per-tool timeout ----------------------------------------------
@pytest.mark.asyncio
async def test_per_tool_timeout_becomes_error_result():
    @tool
    async def slow() -> str:
        """slow"""
        await asyncio.sleep(5)
        return "never"

    slow.timeout = 0.05  # type: ignore[attr-defined]

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="slow", arguments={})], finish_reason="tool_calls"
            ),
            "recovered",
        ]
    )
    agent = Agent("a", model=model, tools=[slow])
    result = await asyncio.wait_for(agent.run("go"), timeout=3)
    # The timeout surfaces as a tool error fed back to the model, not a crash.
    assert result.output == "recovered"
    tool_results = [
        m.content for m in result.messages if getattr(m, "name", None) == "slow"
    ]
    assert tool_results and "timed out" in tool_results[0].lower()


@pytest.mark.asyncio
async def test_runner_default_tool_timeout():
    from yaab import Runner

    @tool
    async def slow() -> str:
        """slow"""
        await asyncio.sleep(5)
        return "never"

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="slow", arguments={})], finish_reason="tool_calls"
            ),
            "ok",
        ]
    )
    runner = Runner(default_tool_timeout=0.05)
    result = await asyncio.wait_for(runner.run(Agent("a", model=model, tools=[slow]), "go"), 3)
    assert result.output == "ok"
