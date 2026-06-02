"""Persisted run history + computed span/waterfall over HTTP.

A run's per-step timeline (model calls, tool calls, transfers, approvals) is
recorded in a trace store; these endpoints surface it so a debugger can replay a
run with model/tool/token/cost/latency detail:

* ``GET /runs/{id}/events`` returns the full persisted event trace;
* ``GET /runs/{id}/trace`` returns a computed waterfall of typed spans plus
  token/cost/latency rollups.

Both ``404`` cleanly when no trace store is configured. We also pin the
``_safe_event_payload`` fix that surfaces ``usage`` and ``duration_ms`` on the
live SSE stream.

Offline: a real ``Runner`` with an ``InMemoryTraceStore`` produces the event
shape, and the endpoints are exercised over a seeded store so the waterfall math
is deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runner import Runner  # noqa: E402
from yaab.runs.trace import InMemoryTraceStore  # noqa: E402
from yaab.serve import _compute_trace, _safe_event_payload, fastapi_server_app  # noqa: E402
from yaab.types import Event, EventType, RunResult, Usage  # noqa: E402


def _agent(out: str = "served-output", **kw) -> Agent:
    return Agent("svc", model=TestModel(out), registry_id="svc", **kw)


# --- endpoints 404 cleanly without a trace store ----------------------
def test_events_endpoint_404_without_trace_store():
    client = TestClient(fastapi_server_app(_agent()))
    assert client.get("/runs/r1/events").status_code == 404
    assert client.get("/runs/r1/trace").status_code == 404


def _seed_real_trace(trace: InMemoryTraceStore) -> tuple[str, RunResult]:
    """Persist a real run's events into ``trace`` and return (run_id, result).

    Drives a genuine tool-calling run through a ``Runner`` to capture the exact
    JSON-safe event shape the runner emits (``_safe_event``), then appends each
    event to the trace store. This decouples the endpoint assertions from the
    runner's internal trace-append timing while still exercising the real payload
    fields (model name, finish reason, per-call usage, durations).
    """
    from yaab.runner import _safe_event

    @tool
    async def lookup(ctx) -> str:
        """look something up"""
        return "looked-up"

    agent = Agent(
        "svc",
        model=TestModel(custom_output="final", call_tools=["lookup"]),
        tools=[lookup],
        registry_id="svc",
    )
    runner = Runner()

    async def go() -> tuple[str, RunResult]:
        # Capture events from the live stream (before ``result.events`` is
        # attached) so dumping the RUN_END result doesn't recurse into itself.
        run_id = ""
        result: RunResult | None = None
        seq = 0
        async for ev in runner.run_stream(agent, "hi"):
            run_id = ev.run_id
            if ev.type is EventType.RUN_END:
                result = ev.payload["result"]
            payload = _safe_event(ev)
            payload["seq"] = seq
            await trace.append(run_id, seq, payload)
            seq += 1
        assert result is not None
        return run_id, result

    return asyncio.run(go())


# --- a real run's events are persisted and retrievable ----------------
def test_run_events_persisted_and_retrievable():
    trace = InMemoryTraceStore()
    run_id, _ = _seed_real_trace(trace)

    client = TestClient(fastapi_server_app(_agent(), trace_store=trace))
    r = client.get(f"/runs/{run_id}/events")
    assert r.status_code == 200
    events = r.json()["events"]
    types = [e["type"] for e in events]
    assert "model_response" in types
    assert "tool_result" in types
    assert "run_end" in types
    # The model_response carries the model name and finish reason.
    mr = next(e for e in events if e["type"] == "model_response")
    assert mr["payload"]["model"] == "test"
    assert mr["payload"]["finish_reason"] == "tool_calls"


def test_run_trace_waterfall_and_rollups():
    trace = InMemoryTraceStore()
    run_id, result = _seed_real_trace(trace)

    client = TestClient(fastapi_server_app(_agent(), trace_store=trace))
    body = client.get(f"/runs/{run_id}/trace").json()
    assert body["run_id"] == run_id
    span_types = [s["type"] for s in body["spans"]]
    assert "model_call" in span_types
    assert "tool_call" in span_types
    # Model spans carry the model name; durations are present (non-null).
    model_span = next(s for s in body["spans"] if s["type"] == "model_call")
    assert model_span["model"] == "test"
    assert model_span["duration_ms"] is not None
    # Token totals match the run's aggregate usage.
    assert body["totals"]["total_tokens"] == result.usage.total_tokens
    # Per-model and per-tool rollups are present.
    assert "test" in body["models"]
    assert "lookup" in body["tools"]


# --- _compute_trace is deterministic on a seeded trace ----------------
def test_compute_trace_math():
    events = [
        {
            "type": "model_response",
            "payload": {
                "model": "m1",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.002,
                },
            },
            "duration_ms": 120.0,
            "timestamp": 1.0,
        },
        {
            "type": "tool_result",
            "payload": {"name": "search", "result": "x"},
            "duration_ms": 30.0,
            "timestamp": 1.2,
        },
        {
            "type": "run_end",
            "payload": {
                "result": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "cost_usd": 0.002,
                    }
                }
            },
            "duration_ms": 200.0,
            "timestamp": 1.5,
        },
    ]
    trace = _compute_trace(events)
    assert [s["type"] for s in trace["spans"]] == ["model_call", "tool_call"]
    assert trace["totals"]["total_tokens"] == 15
    assert trace["totals"]["cost_usd"] == pytest.approx(0.002)
    assert trace["models"]["m1"]["calls"] == 1
    assert trace["tools"]["search"]["calls"] == 1


# --- _safe_event_payload surfaces usage + duration on RUN_END ---------
def test_safe_event_payload_includes_usage_and_duration():
    usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01)
    result = RunResult(output="done", usage=usage, run_id="r1")
    event = Event(
        type=EventType.RUN_END,
        agent="svc",
        run_id="r1",
        payload={"result": result},
        duration_ms=250.0,
    )
    payload = _safe_event_payload(event)
    assert payload["output"] == "done"
    assert payload["usage"]["total_tokens"] == 15
    assert payload["usage"]["cost_usd"] == pytest.approx(0.01)
    assert payload["duration_ms"] == 250.0


def test_safe_event_payload_model_response_keeps_fields():
    event = Event(
        type=EventType.MODEL_RESPONSE,
        agent="svc",
        run_id="r1",
        payload={"model": "test", "finish_reason": "stop", "usage": {"total_tokens": 7}},
        duration_ms=12.5,
    )
    payload = _safe_event_payload(event)
    assert payload["model"] == "test"
    assert payload["finish_reason"] == "stop"
    assert payload["duration_ms"] == 12.5


# --- the live SSE stream now carries usage on run_end -----------------
def test_run_stream_sse_includes_usage_on_run_end():
    client = TestClient(fastapi_server_app(_agent("streamed")))
    with client.stream("POST", "/run/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: run_end" in body
    # The run_end data line includes the usage object (the dropped-usage fix).
    assert '"usage"' in body
