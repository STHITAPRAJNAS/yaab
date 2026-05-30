"""Tests for resilience wrappers and declarative agent config."""

from __future__ import annotations

import pytest

from yaab import Agent, agent_from_dict
from yaab.exceptions import ModelError
from yaab.models.resilient import CircuitBreaker, RateLimiter, ResilientModel
from yaab.models.test_model import TestModel
from yaab.types import Message, Role


# --- rate limiter ------------------------------------------------------
@pytest.mark.asyncio
async def test_rate_limiter_allows_burst_then_throttles():
    import time

    rl = RateLimiter(rate=2, per=1.0)
    start = time.monotonic()
    await rl.acquire()
    await rl.acquire()  # burst of 2 is instant
    await rl.acquire()  # third must wait for a token
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3  # had to wait for refill


# --- circuit breaker ---------------------------------------------------
def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(threshold=2, cooldown=0.2)
    assert cb.state == "closed"
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"
    with pytest.raises(ModelError):
        cb.check()
    import time

    time.sleep(0.25)
    assert cb.state == "half_open"
    cb.check()  # half-open allows a probe
    cb.record_success()
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_resilient_model_records_failure_and_opens():
    class Failing:
        name = "failing"

        async def complete(self, messages, **kwargs):
            raise RuntimeError("provider down")

        def stream(self, messages, **kwargs): ...

    cb = CircuitBreaker(threshold=1, cooldown=10)
    model = ResilientModel(Failing(), circuit_breaker=cb)
    with pytest.raises(RuntimeError):
        await model.complete([Message(role=Role.USER, content="hi")])
    # breaker is now open -> fails fast with ModelError
    assert cb.state == "open"
    with pytest.raises(ModelError):
        await model.complete([Message(role=Role.USER, content="hi")])


@pytest.mark.asyncio
async def test_resilient_model_passes_through_success():
    cb = CircuitBreaker(threshold=2)
    model = ResilientModel(TestModel("ok"), rate_limiter=RateLimiter(100), circuit_breaker=cb)
    resp = await model.complete([Message(role=Role.USER, content="hi")])
    assert resp.content == "ok"
    assert cb.state == "closed"


# --- declarative config ------------------------------------------------
def test_agent_from_dict_builds_agent():
    agent = agent_from_dict(
        {
            "name": "support-bot",
            "model": "openai/gpt-4o",
            "instructions": "Be helpful.",
            "tools": ["calculator", "current_time"],
            "max_steps": 5,
            "registry_id": "support-bot",
        }
    )
    assert isinstance(agent, Agent)
    assert agent.name == "support-bot"
    assert agent.max_steps == 5
    assert agent.registry_id == "support-bot"
    assert {t.name for t in agent.tools} == {"calculator", "current_time"}


def test_agent_from_dict_requires_name():
    with pytest.raises(ValueError):
        agent_from_dict({"model": "openai/gpt-4o"})


def test_agent_from_dict_unknown_tool_fails_loudly():
    with pytest.raises(ValueError):
        agent_from_dict({"name": "x", "tools": ["nonexistent_tool"]})


def test_agent_from_yaml_string():
    from yaab import agent_from_yaml

    pytest.importorskip("yaml")
    yaml_text = """
name: yaml-bot
model: openai/gpt-4o
instructions: From YAML.
tools:
  - calculator
"""
    agent = agent_from_yaml(yaml_text)
    assert agent.name == "yaml-bot"
    assert agent.tools[0].name == "calculator"
