"""Tests for MapAgent fan-out and SequentialAgent early-stop (Tier 2c)."""

from __future__ import annotations

import pytest

from yaab import Agent, MapAgent, SequentialAgent
from yaab.models.test_model import FunctionModel, TestModel


@pytest.mark.asyncio
async def test_mapagent_explicit_list():
    agent = Agent("echo", model=FunctionModel(lambda msgs: f"got: {msgs[-1].content}"))
    mapper = MapAgent("fan", agent)
    result = await mapper.run(["a", "b", "c"])
    assert result.output == ["got: a", "got: b", "got: c"]
    assert result.usage.requests == 3


@pytest.mark.asyncio
async def test_mapagent_with_map_inputs():
    agent = Agent("echo", model=FunctionModel(lambda msgs: msgs[-1].content.upper()))
    mapper = MapAgent("fan", agent, map_inputs=lambda p: p.split(","))
    result = await mapper.run("x,y,z")
    assert result.output == ["X", "Y", "Z"]


@pytest.mark.asyncio
async def test_mapagent_bounded_concurrency():
    agent = Agent("echo", model=FunctionModel(lambda msgs: msgs[-1].content))
    mapper = MapAgent("fan", agent, max_concurrency=2)
    result = await mapper.run(["1", "2", "3", "4", "5"])
    assert result.output == ["1", "2", "3", "4", "5"]


@pytest.mark.asyncio
async def test_sequential_early_stop():
    a = Agent("a", model=TestModel("STOP here"))
    b = Agent("b", model=TestModel("should not run"))
    ran = {"b": False}

    # Wrap b.run to detect if it executes.
    orig = b.run

    async def tracking_run(*args, **kwargs):
        ran["b"] = True
        return await orig(*args, **kwargs)

    b.run = tracking_run  # type: ignore[assignment]

    seq = SequentialAgent("pipe", [a, b], stop_when=lambda out: "STOP" in str(out))
    result = await seq.run("go")
    assert result.output == "STOP here"
    assert ran["b"] is False  # pipeline stopped before b
