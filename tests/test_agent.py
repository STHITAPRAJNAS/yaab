"""Tests for the Agent / Runner fast path."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from yaab import Agent, EventType, RunContext, tool
from yaab.models.test_model import TestModel
from yaab.testing import TestModel as ExportedTestModel


def test_simple_run():
    agent = Agent("a", model=TestModel("hello world"))
    result = agent.run_sync("hi")
    assert result.output == "hello world"
    assert result.usage.requests == 1


def test_testmodel_exported():
    assert ExportedTestModel is TestModel


@pytest.mark.asyncio
async def test_tool_loop():
    calls = {"n": 0}

    @tool
    def ping() -> str:
        """Return pong."""
        calls["n"] += 1
        return "pong"

    model = TestModel(custom_output="done", call_tools=["ping"])
    agent = Agent("a", model=model, tools=[ping])
    result = await agent.run("go")
    assert calls["n"] == 1
    assert result.output == "done"


@pytest.mark.asyncio
async def test_tool_receives_context_and_deps():
    class Deps(BaseModel):
        user: str

    seen = {}

    @tool
    def whoami(ctx: RunContext) -> str:
        """Report the current user from deps."""
        seen["user"] = ctx.deps.user
        return ctx.deps.user

    model = TestModel(custom_output="ok", call_tools=["whoami"])
    agent = Agent("a", model=model, tools=[whoami], deps_type=Deps)
    await agent.run("hi", deps=Deps(user="alice"))
    assert seen["user"] == "alice"


@pytest.mark.asyncio
async def test_structured_output_validation():
    class Weather(BaseModel):
        city: str
        temp_c: int

    model = TestModel(structured_output={"city": "Paris", "temp_c": 21})
    agent = Agent("w", model=model, output_type=Weather)
    result = await agent.run("weather in Paris")
    assert isinstance(result.output, Weather)
    assert result.output.city == "Paris"
    assert result.output.temp_c == 21


@pytest.mark.asyncio
async def test_event_stream():
    agent = Agent("a", model=TestModel("final"))
    types = [ev.type async for ev in agent._get_runner().run_stream(agent, "hi")]
    assert EventType.RUN_START in types
    assert EventType.FINAL_OUTPUT in types
    assert types[-1] == EventType.RUN_END


@pytest.mark.asyncio
async def test_agent_as_tool():
    inner = Agent("inner", model=TestModel("inner-answer"))
    outer_model = TestModel(custom_output="outer-done", call_tools=["call_inner"])
    outer = Agent("outer", model=outer_model, tools=[inner.as_tool()])
    result = await outer.run("delegate")
    assert result.output == "outer-done"
