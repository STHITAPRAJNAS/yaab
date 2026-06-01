"""Streaming through the tool loop (Phase C).

`Runner.stream_run` / `Agent.stream_events` yield a unified event stream where
token deltas (TEXT_DELTA) arrive live during each answering turn AND tools
execute mid-run (TOOL_CALL/TOOL_RESULT), across multiple steps — not just the
single answering turn that `agent.stream()` covers.
"""

from __future__ import annotations

import pytest

from yaab import Agent, EventType, tool
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import ToolCall


@pytest.mark.asyncio
async def test_stream_run_emits_text_deltas_and_final():
    agent = Agent("a", model=TestModel("hello world foo"))
    events = [e async for e in agent.stream_events("hi")]
    types = [e.type for e in events]
    assert EventType.RUN_START in types
    assert types[-1] is EventType.RUN_END
    deltas = [e.payload["delta"] for e in events if e.type is EventType.TEXT_DELTA]
    assert "".join(deltas).strip() == "hello world foo"
    # The final result is carried on RUN_END.
    final = events[-1].payload["result"]
    assert final.output == "hello world foo"


@pytest.mark.asyncio
async def test_stream_run_interleaves_tools_then_streams_final():
    calls = {"n": 0}

    @tool
    def lookup(q: str = "") -> str:
        """look up"""
        calls["n"] += 1
        return "42"

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="lookup", arguments={"q": "x"})],
                finish_reason="tool_calls",
            ),
            "the answer is 42",
        ]
    )
    agent = Agent("a", model=model, tools=[lookup])
    events = [e async for e in agent.stream_events("go")]
    types = [e.type for e in events]
    # Tools executed mid-stream...
    assert EventType.TOOL_CALL in types and EventType.TOOL_RESULT in types
    assert calls["n"] == 1
    # ...then the final turn streamed tokens.
    tool_idx = types.index(EventType.TOOL_RESULT)
    later_deltas = [e.payload["delta"] for e in events[tool_idx:] if e.type is EventType.TEXT_DELTA]
    assert "".join(later_deltas).strip() == "the answer is 42"
    assert events[-1].payload["result"].output == "the answer is 42"


@pytest.mark.asyncio
async def test_stream_run_tool_call_order_preserved():
    @tool
    def a_tool() -> str:
        """a"""
        return "ra"

    @tool
    def b_tool() -> str:
        """b"""
        return "rb"

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[
                    ToolCall(name="a_tool", arguments={}),
                    ToolCall(name="b_tool", arguments={}),
                ],
                finish_reason="tool_calls",
            ),
            "done",
        ]
    )
    agent = Agent("a", model=model, tools=[a_tool, b_tool])
    events = [e async for e in agent.stream_events("go")]
    called = [e.payload["name"] for e in events if e.type is EventType.TOOL_CALL]
    assert called == ["a_tool", "b_tool"]


@pytest.mark.asyncio
async def test_stream_events_is_run_consistent():
    # Streaming and non-streaming produce the same final output.
    @tool
    def echo(x: str = "") -> str:
        """echo"""
        return x

    model_for_run = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="echo", arguments={"x": "hi"})],
                finish_reason="tool_calls",
            ),
            "final",
        ]
    )
    run_result = await Agent("a", model=model_for_run, tools=[echo]).run("go")

    model_for_stream = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="echo", arguments={"x": "hi"})],
                finish_reason="tool_calls",
            ),
            "final",
        ]
    )
    events = [e async for e in Agent("a", model=model_for_stream, tools=[echo]).stream_events("go")]
    assert events[-1].payload["result"].output == run_result.output == "final"
