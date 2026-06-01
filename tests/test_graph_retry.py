"""Tests for per-node retry policies in the durable graph engine.

These exercise the workflow-runtime feature where a node may declare
a :class:`RetryPolicy` so transient failures retry with backoff instead of
aborting the whole graph. Control-flow ``Interrupt`` must be exempt — it is not
an error, so it must never be retried or swallowed.
"""

from __future__ import annotations

import asyncio

import pytest

from yaab.exceptions import Interrupt
from yaab.graph import (
    START,
    Channel,
    MemorySaver,
    RetryPolicy,
    StateGraph,
)


class Boom(RuntimeError):
    """A transient error used to drive retries in these tests."""


def _flaky_node(fail_times: int):
    """Build a node that raises ``fail_times`` before succeeding.

    Returns the node fn and a mutable ``calls`` list so a test can assert how
    many times the node body actually ran.
    """
    calls: list[int] = []

    def node(state):
        calls.append(1)
        if len(calls) <= fail_times:
            raise Boom(f"transient {len(calls)}")
        return {"ok": True}

    return node, calls


# --- RetryPolicy data model ------------------------------------------------


def test_retry_policy_defaults():
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.backoff == 0.5
    assert policy.backoff_multiplier == 2.0
    assert policy.retry_on == (Exception,)


def test_retry_policy_backoff_schedule():
    # Attempt N (1-indexed) sleeps backoff * multiplier**(N-1).
    policy = RetryPolicy(backoff=0.5, backoff_multiplier=2.0)
    assert policy.backoff_for(1) == pytest.approx(0.5)
    assert policy.backoff_for(2) == pytest.approx(1.0)
    assert policy.backoff_for(3) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_retry_policy_sleep_uses_asyncio_sleep(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    policy = RetryPolicy(backoff=0.25, backoff_multiplier=3.0)
    await policy.sleep(1)
    await policy.sleep(2)
    assert slept == [pytest.approx(0.25), pytest.approx(0.75)]


# --- retry behavior in the engine -----------------------------------------


@pytest.mark.asyncio
async def test_node_fails_twice_then_succeeds(monkeypatch):
    # No real waiting in tests.
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    node, calls = _flaky_node(fail_times=2)

    g = StateGraph()
    g.add_node("flaky", node, retry=RetryPolicy(max_attempts=3))
    g.add_edge(START, "flaky")
    g.set_finish_point("flaky")

    result = await g.compile().ainvoke({})
    assert result.state["ok"] is True
    assert result.interrupted is False
    assert len(calls) == 3  # 2 failures + 1 success
    # Two retries were needed and they are observable on the result.
    assert result.retries == {"flaky": 2}


@pytest.mark.asyncio
async def test_retries_exhausted_propagates_original_exception(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    node, calls = _flaky_node(fail_times=99)  # always fails

    g = StateGraph()
    g.add_node("flaky", node, retry=RetryPolicy(max_attempts=3))
    g.add_edge(START, "flaky")
    g.set_finish_point("flaky")

    with pytest.raises(Boom):
        await g.compile().ainvoke({})
    assert len(calls) == 3  # exactly max_attempts attempts, no more


@pytest.mark.asyncio
async def test_retry_on_filters_exception_types(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    calls: list[int] = []

    def node(state):
        calls.append(1)
        raise ValueError("not retryable here")

    g = StateGraph()
    # Only retry Boom; a ValueError must propagate on the first attempt.
    g.add_node("n", node, retry=RetryPolicy(max_attempts=5, retry_on=(Boom,)))
    g.add_edge(START, "n")
    g.set_finish_point("n")

    with pytest.raises(ValueError):
        await g.compile().ainvoke({})
    assert len(calls) == 1  # not retried


@pytest.mark.asyncio
async def test_node_without_policy_is_unchanged():
    # A node with no retry policy still aborts immediately on error.
    calls: list[int] = []

    def node(state):
        calls.append(1)
        raise Boom("once")

    g = StateGraph()
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.set_finish_point("n")

    with pytest.raises(Boom):
        await g.compile().ainvoke({})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_backoff_sleeps_happen(monkeypatch):
    slept: list[float] = []

    async def record_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    node, _calls = _flaky_node(fail_times=2)

    g = StateGraph()
    g.add_node(
        "flaky",
        node,
        retry=RetryPolicy(max_attempts=3, backoff=0.5, backoff_multiplier=2.0),
    )
    g.add_edge(START, "flaky")
    g.set_finish_point("flaky")

    await g.compile().ainvoke({})
    # Two failures -> two backoff sleeps: 0.5 then 1.0.
    assert slept == [pytest.approx(0.5), pytest.approx(1.0)]


# --- Interrupt is control flow, never retried ------------------------------


@pytest.mark.asyncio
async def test_interrupt_is_never_retried(monkeypatch):
    slept: list[float] = []

    async def record_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    calls: list[int] = []

    def gate(state, ctx):
        calls.append(1)
        decision = ctx.interrupt({"need": "approval"})
        return {"approved": decision}

    g = StateGraph()
    # A generous retry policy must NOT cause the interrupt to be retried.
    g.add_node("gate", gate, retry=RetryPolicy(max_attempts=5, retry_on=(Exception,)))
    g.add_edge(START, "gate")
    g.set_finish_point("gate")
    app = g.compile(checkpointer=MemorySaver())

    paused = await app.ainvoke({}, thread_id="t1")
    assert paused.interrupted is True
    assert paused.interrupt_value == {"need": "approval"}
    assert len(calls) == 1  # the node body ran exactly once before pausing
    assert slept == []  # no backoff sleep on an interrupt

    resumed = await app.ainvoke(thread_id="t1", resume=True)
    assert resumed.interrupted is False
    assert resumed.state["approved"] is True


@pytest.mark.asyncio
async def test_interrupt_subclass_of_retry_on_still_not_retried(monkeypatch):
    # Even when retry_on is the broad (Exception,) — and Interrupt IS an
    # Exception — control flow must win. This guards the precedence ordering.
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    calls: list[int] = []

    def gate(state, ctx):
        calls.append(1)
        raise Interrupt({"pause": "now"})

    g = StateGraph()
    g.add_node("gate", gate, retry=RetryPolicy(max_attempts=5))
    g.add_edge(START, "gate")
    g.set_finish_point("gate")
    app = g.compile(checkpointer=MemorySaver())

    paused = await app.ainvoke({}, thread_id="t2")
    assert paused.interrupted is True
    assert len(calls) == 1


# --- retries are observable and accumulate across supersteps ---------------


@pytest.mark.asyncio
async def test_no_retries_recorded_when_nothing_retries():
    g = StateGraph()
    g.add_node("a", lambda s: {"x": 1})
    g.add_edge(START, "a")
    g.set_finish_point("a")
    result = await g.compile().ainvoke({})
    assert result.retries == {}


@pytest.mark.asyncio
async def test_retries_recorded_in_parallel_superstep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    left, left_calls = _flaky_node(fail_times=1)
    right, right_calls = _flaky_node(fail_times=2)

    g = StateGraph(channels={"hits": Channel("append", default=[])})
    g.add_node("fan", lambda s: {})
    g.add_node("left", lambda s: {"hits": "L"} if left(s) else {"hits": "L"}, retry=RetryPolicy())
    g.add_node("right", lambda s: {"hits": "R"} if right(s) else {"hits": "R"}, retry=RetryPolicy())
    g.add_edge(START, "fan")
    g.add_edge("fan", "left")
    g.add_edge("fan", "right")
    g.set_finish_point("left")
    g.set_finish_point("right")

    result = await g.compile().ainvoke({})
    assert sorted(result.state["hits"]) == ["L", "R"]
    assert result.retries == {"left": 1, "right": 2}
    assert len(left_calls) == 2
    assert len(right_calls) == 3


async def _noop_sleep(_seconds):
    """A no-op replacement for ``asyncio.sleep`` so tests don't actually wait."""
    return None
