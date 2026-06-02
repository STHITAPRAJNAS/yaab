"""Tests for the trace-debug additions to the `yaab web` dev console (yaab.web).

The console gains a span/waterfall **Trace** tab and a **State** inspector tab,
upgrades the **Runs** tab with per-run Trace + Replay actions, adds an
**Approvals** tab for out-of-band human sign-off, and enriches the **Events**
tab with a per-run usage summary header. All of it is the same vanilla-JS,
no-build ``_PAGE`` template, and it degrades gracefully when the matching
durable stores aren't configured.

These tests assert, via a FastAPI TestClient, that:

* the page exposes the new tab markers (trace, state, approvals) alongside the
  original four, and the JS handlers each tab relies on;
* ``web_app`` forwards the durable stores (trace/approval/run/cron) to the
  underlying serve app so the new endpoints are reachable *through* the console;
* the Trace/State/Approvals endpoints are mounted and return the documented
  shapes through the web app, and degrade to a clean 404 when the store is
  absent (the UI's graceful-degradation contract).

Offline only: a ``TestModel``/``FunctionModel`` agent, an ``InMemoryTraceStore``
seeded from a real run, and in-memory approval/run stores.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.governance.approval import ToolApprovalPlugin  # noqa: E402
from yaab.governance.approvals import InMemoryApprovalStore  # noqa: E402
from yaab.graph.checkpoint import MemorySaver  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runner import Runner, _safe_event  # noqa: E402
from yaab.runs.memory import InMemoryRunStore  # noqa: E402
from yaab.runs.trace import InMemoryTraceStore  # noqa: E402
from yaab.types import EventType, RunResult  # noqa: E402
from yaab.web import web_app  # noqa: E402


def _agent(out: str = "hi", *, instructions: str = "Be helpful.") -> Agent:
    @tool
    def adder(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    return Agent(
        "console",
        model=TestModel(out),
        tools=[adder],
        instructions=instructions,
        registry_id="console",
    )


def _client(agent: Agent | None = None, **kw) -> TestClient:
    return TestClient(web_app(agent or _agent(), **kw))


def _page(**kw) -> str:
    return _client(**kw).get("/").text


def _seed_real_trace(trace: InMemoryTraceStore) -> tuple[str, RunResult]:
    """Persist a real tool-calling run's events into ``trace``; return (id, result)."""

    @tool
    async def lookup(ctx) -> str:
        """look something up"""
        return "looked-up"

    agent = Agent(
        "console",
        model=TestModel(custom_output="final", call_tools=["lookup"]),
        tools=[lookup],
        registry_id="console",
    )
    runner = Runner()

    async def go() -> tuple[str, RunResult]:
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


# --- the page still serves and keeps the original tabs -----------------
def test_page_keeps_original_tabs_and_serves_html():
    page = _page()
    assert "<!doctype html>" in page.lower()
    for tab in ("chat", "events", "runs", "agent"):
        assert f'data-tab="{tab}"' in page, f"missing original tab marker {tab!r}"


def test_page_has_new_trace_state_approvals_tab_markers():
    """The trace-debug console adds Trace, State and Approvals tabs."""
    page = _page()
    for tab in ("trace", "state", "approvals"):
        assert f'data-tab="{tab}"' in page, f"missing new tab marker {tab!r}"


def test_page_ships_new_js_handlers():
    """The load-bearing client logic for the new/upgraded tabs must be present."""
    page = _page()
    for fn in (
        "function loadTrace",  # TRACE tab: GET /runs/{id}/trace -> waterfall
        "function loadState",  # STATE tab: GET /sessions/{id}/state -> KV table
        "function loadApprovals",  # APPROVALS tab: GET /approvals?status=pending
        "function decideApproval",  # APPROVALS tab: POST approve/deny
        "function replayRun",  # RUNS tab: GET /runs/{id}/events -> re-render
    ):
        assert fn in page, f"missing JS handler {fn!r}"
    # The original handlers must survive the upgrade.
    for fn in ("function streamRun", "function refreshRuns", "function switchTab"):
        assert fn in page, f"upgrade dropped original handler {fn!r}"


def test_page_references_new_endpoints():
    """The page wires the new endpoints by path so they can't silently vanish."""
    page = _page()
    assert "/trace" in page  # GET /runs/{id}/trace
    assert "/events" in page  # GET /runs/{id}/events (replay)
    assert "/sessions/" in page and "/state" in page  # GET /sessions/{id}/state
    assert "/approvals" in page  # GET /approvals + approve/deny


def test_trace_tab_renders_span_token_cost_affordances():
    """The Trace tab template carries the span/latency/token/cost render hooks."""
    page = _page()
    # A span waterfall keyed by type, latency bars, and token/cost badges.
    assert "renderSpan" in page or "span" in page
    assert "duration_ms" in page  # latency bars driven by per-span duration
    assert "cost_usd" in page  # cost badge on model spans
    assert "total_tokens" in page  # rollup totals header


def test_trace_tab_draws_latency_bars():
    """Each span gets a proportional latency bar, not just a millisecond label.

    The waterfall is only readable at a glance when slow steps are visibly wider
    than fast ones, so the template must carry a bar element whose width is driven
    by the span's ``duration_ms``.
    """
    page = _page()
    assert "lat-bar" in page  # a dedicated latency-bar element class
    # The bar width is computed from a span's duration relative to the slowest.
    assert "maxDur" in page


def test_state_tab_has_refresh_button():
    """The State inspector can re-pull a session's state without retyping the id.

    A reviewer watching state change between turns needs a one-click refresh, so
    the template ships a refresh affordance wired to the same loader.
    """
    page = _page()
    assert "refreshState" in page  # a one-click re-pull of the current session
    assert "Refresh" in page


def test_events_tab_has_usage_summary_header():
    """The Events tab gains a run-summary header (tokens/cost/latency on RUN_END)."""
    page = _page()
    # The summary parses the now-complete RUN_END payload.
    assert "usage" in page
    assert "run_end" in page


def test_events_tab_shows_model_name_chips():
    """Model-response events surface which model answered, as a chip on the row.

    Scanning a timeline for *which* model handled a step is far faster with a
    visible model-name chip than expanding every event, so the renderer pulls the
    model name from the (possibly nested) event payload and renders a chip.
    """
    page = _page()
    assert "chip" in page  # a per-event model-name chip
    assert "model" in page  # the chip is keyed off the event's model field


# --- web_app forwards durable stores to the serve app ------------------
def test_web_app_forwards_trace_store_endpoints():
    trace = InMemoryTraceStore()
    run_id, result = _seed_real_trace(trace)
    client = _client(trace_store=trace)

    ev = client.get(f"/runs/{run_id}/events")
    assert ev.status_code == 200
    assert any(e["type"] == "tool_result" for e in ev.json()["events"])

    tr = client.get(f"/runs/{run_id}/trace")
    assert tr.status_code == 200
    body = tr.json()
    assert body["run_id"] == run_id
    assert body["totals"]["total_tokens"] == result.usage.total_tokens
    assert any(s["type"] == "model_call" for s in body["spans"])


def test_trace_endpoints_404_without_store_through_web_app():
    """Graceful degradation: no trace store -> the trace endpoints 404 cleanly."""
    client = _client()  # no trace_store
    assert client.get("/runs/r1/events").status_code == 404
    assert client.get("/runs/r1/trace").status_code == 404


def test_web_app_forwards_state_inspector():
    agent = _agent()
    client = _client(agent)
    # Drive a run so the default in-memory session service has a session.
    client.post("/run", json={"prompt": "hi", "session_id": "s1"})
    r = client.get("/sessions/s1/state")
    assert r.status_code == 200
    assert r.json()["session_id"] == "s1"
    assert "state" in r.json()


def test_state_inspector_404_for_unknown_session():
    client = _client()
    assert client.get("/sessions/nope/state").status_code == 404


# --- Approvals tab end-to-end through the web app ----------------------
def _approval_agent() -> Agent:
    @tool
    async def wire(ctx, amount: int = 100) -> str:
        """move money"""
        return f"wired {amount}"

    return Agent(
        "console",
        model=TestModel(custom_output="all done", call_tools=["wire"]),
        tools=[wire],
        registry_id="console",
    )


def _approval_web_app(agent, approvals, runs):
    plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=approvals)
    runner = Runner(run_checkpointer=MemorySaver(), plugins=[plugin])
    return web_app(agent, runner=runner, approval_store=approvals, run_store=runs)


def test_web_app_forwards_approvals_list_and_decide():
    import time

    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    with TestClient(_approval_web_app(_approval_agent(), approvals, runs)) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]

        # Wait for the run to park and a pending approval to surface.
        deadline = time.monotonic() + 5.0
        pending: list = []
        while time.monotonic() < deadline:
            pending = client.get("/approvals?status=pending").json()
            if pending:
                break
            time.sleep(0.01)
        assert pending, "no pending approval appeared for the Approvals tab"
        ap = pending[0]
        assert ap["tool"] == "wire"

        # The Approvals tab approves via POST; the run then resumes to completion.
        r = client.post(f"/approvals/{ap['approval_id']}/approve", json={"reviewer": "alice"})
        assert r.status_code == 200
        assert r.json()["decision"] == "approved"

        deadline = time.monotonic() + 5.0
        final: dict = {}
        while time.monotonic() < deadline:
            final = client.get(f"/runs/{run_id}").json()
            if final["status"] == "completed":
                break
            time.sleep(0.01)
        assert final["status"] == "completed"
        assert final["output"] == "all done"


def test_approvals_endpoint_404_without_store_through_web_app():
    client = _client()  # no approval_store
    assert client.get("/approvals").status_code == 404


# --- the original console wiring is untouched --------------------------
def test_original_serve_endpoints_still_reachable():
    client = _client(_agent("done"))
    client.post("/run", json={"prompt": "hi"})
    runs = client.get("/runs").json()
    assert isinstance(runs, list) and runs
    assert {"id", "status", "started_at"} <= set(runs[0])
    assert client.get("/agent/info").json()["name"] == "console"
    assert client.get("/health").json()["status"] == "ok"
