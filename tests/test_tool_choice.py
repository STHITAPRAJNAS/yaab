"""Tests for tool_choice, reasoning capture, and tool-arg repair (Tier 1b)."""

from __future__ import annotations

import pytest

from yaab import Agent, EventType, tool
from yaab.models.test_model import TestModel
from yaab.plugins import Plugin
from yaab.runner import _normalize_tool_choice


@tool
def ping(value: int = 0) -> str:
    """Return pong."""
    return f"pong-{value}"


def _schemas():
    return [ping.schema()]


def test_normalize_tool_choice_passthrough():
    assert _normalize_tool_choice("auto", _schemas()) == "auto"
    assert _normalize_tool_choice("required", _schemas()) == "required"
    assert _normalize_tool_choice("none", _schemas()) == "none"
    d = {"type": "function", "function": {"name": "ping"}}
    assert _normalize_tool_choice(d, _schemas()) == d


def test_normalize_tool_choice_named_tool_expands():
    out = _normalize_tool_choice("ping", _schemas())
    assert out == {"type": "function", "function": {"name": "ping"}}


def test_normalize_tool_choice_no_tools_is_none():
    assert _normalize_tool_choice("ping", None) is None


@pytest.mark.asyncio
async def test_agent_passes_tool_choice_to_model():
    model = TestModel(custom_output="done", call_tools=["ping"])
    agent = Agent("a", model=model, tools=[ping], tool_choice="required")
    await agent.run("go")
    # The first model call should have received tool_choice="required".
    assert model.tool_choices[0] == "required"


@pytest.mark.asyncio
async def test_named_tool_choice_expanded_to_dict():
    model = TestModel(custom_output="done", call_tools=["ping"])
    agent = Agent("a", model=model, tools=[ping], tool_choice="ping")
    await agent.run("go")
    assert model.tool_choices[0] == {"type": "function", "function": {"name": "ping"}}


@pytest.mark.asyncio
async def test_reasoning_trace_emitted_as_event():
    model = TestModel("the answer", reasoning="let me think... 2+2=4")
    agent = Agent("a", model=model)
    events = [e async for e in agent._get_runner().run_stream(agent, "2+2?")]
    deltas = [e for e in events if e.type is EventType.MODEL_DELTA]
    assert deltas
    assert "2+2=4" in deltas[0].payload["reasoning"]


@pytest.mark.asyncio
async def test_repair_tool_args_hook():
    captured = {}

    @tool
    def add(a: int, b: int) -> int:
        """Add two ints."""
        captured["a"], captured["b"] = a, b
        return a + b

    class RepairPlugin(Plugin):
        async def repair_tool_args(self, ctx, agent, tool, args):
            # Model emitted strings; coerce to ints before validation.
            if tool == "add":
                return {k: int(v) for k, v in args.items()}
            return None

    from yaab import Runner
    from yaab.models.base import ModelResponse
    from yaab.types import ToolCall

    # Model asks for add with string args, then finalizes.
    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="add", arguments={"a": "2", "b": "3"})],
                finish_reason="tool_calls",
            ),
            "sum is 5",
        ]
    )
    runner = Runner(plugins=[RepairPlugin()])
    agent = Agent("a", model=model, tools=[add])
    result = await runner.run(agent, "add 2 and 3")
    assert captured == {"a": 2, "b": 3}
    assert result.output == "sum is 5"
