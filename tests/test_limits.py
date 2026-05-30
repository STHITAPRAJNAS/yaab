"""Tests for UsageLimits and run cancellation/timeout (Tier 1)."""

from __future__ import annotations

import pytest

from yaab import (
    Agent,
    CancellationToken,
    RunCancelled,
    UsageLimitExceeded,
    UsageLimits,
)
from yaab.models.test_model import TestModel
from yaab.types import Usage


def test_usage_limits_check_usage_caps():
    limits = UsageLimits(max_total_tokens=10, max_requests=1)
    limits.check_usage(Usage(requests=1, total_tokens=5))  # ok
    with pytest.raises(UsageLimitExceeded) as ei:
        limits.check_usage(Usage(requests=1, total_tokens=11))
    assert ei.value.limit == "total_tokens"
    with pytest.raises(UsageLimitExceeded) as ei:
        limits.check_usage(Usage(requests=2, total_tokens=1))
    assert ei.value.limit == "requests"


def test_usage_limits_per_tool():
    limits = UsageLimits(max_tool_calls=5, per_tool_calls={"charge": 1})
    limits.check_tool_call("charge", {"charge": 1})  # ok at the cap
    with pytest.raises(UsageLimitExceeded) as ei:
        limits.check_tool_call("charge", {"charge": 2})
    assert ei.value.limit == "tool:charge"
    with pytest.raises(UsageLimitExceeded) as ei:
        limits.check_tool_call("x", {"a": 3, "b": 3})
    assert ei.value.limit == "tool_calls"


@pytest.mark.asyncio
async def test_runner_enforces_request_limit():
    # TestModel that keeps calling a tool would loop; cap requests at 1.
    model = TestModel(custom_output="done", call_tools=["noop"])

    from yaab import tool

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    agent = Agent("a", model=model, tools=[noop], max_steps=10)
    with pytest.raises(UsageLimitExceeded):
        await agent.run("go", usage_limits=UsageLimits(max_requests=1))


@pytest.mark.asyncio
async def test_runner_enforces_per_tool_limit():
    model = TestModel(custom_output="done", call_tools=["charge"])

    from yaab import tool

    @tool
    def charge() -> str:
        """Charge once."""
        return "charged"

    agent = Agent("a", model=model, tools=[charge], max_steps=10)
    with pytest.raises(UsageLimitExceeded) as ei:
        # The model requests `charge` on the first response; allow 0 calls.
        await agent.run("go", usage_limits=UsageLimits(per_tool_calls={"charge": 0}))
    assert "charge" in ei.value.limit


def test_cancellation_token_explicit():
    tok = CancellationToken()
    assert not tok.cancelled
    tok.cancel("user_stop")
    assert tok.cancelled
    with pytest.raises(RunCancelled) as ei:
        tok.raise_if_cancelled()
    assert ei.value.reason == "user_stop"


def test_cancellation_token_timeout():
    tok = CancellationToken.with_timeout(-1)  # already past deadline
    assert tok.cancelled
    with pytest.raises(RunCancelled) as ei:
        tok.raise_if_cancelled()
    assert ei.value.reason == "timeout"


@pytest.mark.asyncio
async def test_runner_honors_cancellation():
    tok = CancellationToken()
    tok.cancel()
    agent = Agent("a", model=TestModel("hi"))
    with pytest.raises(RunCancelled):
        await agent.run("go", cancellation=tok)


@pytest.mark.asyncio
async def test_runner_honors_timeout():
    agent = Agent("a", model=TestModel("hi"))
    with pytest.raises(RunCancelled) as ei:
        await agent.run("go", timeout=-1)  # immediate deadline
    assert ei.value.reason == "timeout"


@pytest.mark.asyncio
async def test_no_limits_runs_normally():
    agent = Agent("a", model=TestModel("fine"))
    result = await agent.run("go")
    assert result.output == "fine"
