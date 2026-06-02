"""Per-step model/tool/token/latency detail on the event stream.

The trace console renders a span waterfall with model name, finish reason,
per-call token deltas, and per-step latency — all driven by the structured
events the runner emits. These tests pin the enrichment at the source:

* ``MODEL_RESPONSE`` carries ``model``, ``finish_reason``, and a per-call
  ``usage`` delta;
* ``MODEL_RESPONSE`` / ``TOOL_RESULT`` / ``RUN_END`` carry a ``duration_ms``;
* when a trace store is configured, every emitted event is appended to it
  keyed by ``(run_id, seq)`` in a JSON-safe shape — and a run without one is
  byte-for-byte the existing behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from yaab import Agent
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool
from yaab.types import EventType, ToolCall


def ping_impl() -> str:
    """Return pong."""
    return "pong"


ping = FunctionTool(ping_impl, name="ping")


class FakeTraceStore:
    """A duck-typed trace store: records every appended event in order."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int, dict[str, Any]]] = []

    def append(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        self.rows.append((run_id, seq, event))


def _tool_then_answer() -> TestModel:
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="ping", arguments={})],
                finish_reason="tool_calls",
                model="gpt-test",
            ),
            ModelResponse(content="done", finish_reason="stop", model="gpt-test"),
        ]
    )


@pytest.mark.asyncio
async def test_model_response_carries_model_and_finish_reason():
    runner = Runner()
    agent = Agent("a", model=_tool_then_answer(), tools=[ping], runner=runner)

    events = [ev async for ev in runner.run_stream(agent, "go")]
    model_evs = [e for e in events if e.type is EventType.MODEL_RESPONSE]
    assert model_evs
    for ev in model_evs:
        assert ev.payload.get("model") == "gpt-test"
        assert ev.payload.get("finish_reason") in ("tool_calls", "stop")
        # A per-call usage delta is attached (this single call's tokens).
        assert "usage" in ev.payload
        assert ev.payload["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_durations_on_model_tool_and_run_end():
    runner = Runner()
    agent = Agent("a", model=_tool_then_answer(), tools=[ping], runner=runner)

    events = [ev async for ev in runner.run_stream(agent, "go")]
    by_type = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    for etype in (EventType.MODEL_RESPONSE, EventType.TOOL_RESULT, EventType.RUN_END):
        evs = by_type.get(etype, [])
        assert evs, f"expected at least one {etype} event"
        for ev in evs:
            assert ev.duration_ms is not None
            assert ev.duration_ms >= 0.0


@pytest.mark.asyncio
async def test_trace_store_receives_every_event_json_safe():
    trace = FakeTraceStore()
    runner = Runner(trace_store=trace)
    agent = Agent("a", model=_tool_then_answer(), tools=[ping], runner=runner)

    events = [ev async for ev in runner.run_stream(agent, "go")]

    # Every emitted event was appended, in order, keyed by (run_id, seq).
    assert len(trace.rows) == len(events)
    run_id = events[0].run_id
    for i, (rid, seq, payload) in enumerate(trace.rows):
        assert rid == run_id
        assert seq == i
        # JSON-safe: type is a plain string, payload is a dict.
        assert isinstance(payload["type"], str)
        assert isinstance(payload["payload"], dict)

    # The RUN_END row's result is JSON-safe (not a live RunResult object).
    end_rows = [r for r in trace.rows if r[2]["type"] == EventType.RUN_END.value]
    assert end_rows
    end_payload = end_rows[-1][2]["payload"]
    assert "result" in end_payload
    # usage is surfaced for cost attribution.
    assert end_payload["result"]["usage"]["total_tokens"] >= 0


@pytest.mark.asyncio
async def test_no_trace_store_is_unchanged():
    runner = Runner()
    assert runner.trace_store is None
    agent = Agent("a", model=TestModel("hi"), runner=runner)
    result = await runner.run(agent, "go")
    assert result.output == "hi"
