"""Tests for the multi-pane `yaab web` dev console (yaab.web).

The web app wraps the serve app and adds a single self-contained HTML page with
four tabs (Chat, Events, Runs, Agent) plus a `/agent/info` introspection
endpoint mounted on the web app itself. These tests assert, via a FastAPI
TestClient, that:

* the HTML page exposes the four tab markers and the JS handlers the tabs need
  (SSE parsing for /run/stream, run polling, agent-card rendering);
* `/agent/info` returns the agent's name, stringified model spec, tool
  list (name + description) and instructions;
* the underlying serve endpoints (/run/stream, /runs, /chat/stream, the agent
  card) are all reachable *through* the web app, since web.py mounts serve.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
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


def _client(agent: Agent | None = None) -> TestClient:
    return TestClient(web_app(agent or _agent()))


# --- the page itself ---------------------------------------------------
def test_page_served_as_html():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<!doctype html>" in r.text.lower()
    # Agent name is rendered into the title/header.
    assert "console" in r.text


def test_page_has_four_tab_markers():
    """All four dev-console tabs must be present as data-tab markers."""
    page = _client().get("/").text
    for tab in ("chat", "events", "runs", "agent"):
        assert f'data-tab="{tab}"' in page, f"missing tab marker for {tab!r}"


def test_page_includes_sse_and_polling_js_handlers():
    """The page must ship the vanilla-JS handlers the tabs rely on.

    We assert function names are present so the load-bearing client logic
    (POST-SSE streaming via fetch ReadableStream, run polling, agent-card
    rendering) can't silently vanish.
    """
    page = _client().get("/").text
    for fn in (
        "function streamRun",  # EVENTS tab: POST /run/stream via fetch reader
        "function refreshRuns",  # RUNS tab: GET /runs + auto-refresh
        "function cancelRun",  # RUNS tab: POST /runs/{id}/cancel
        "function loadAgent",  # AGENT tab: GET /agent/info + /.well-known card
        "function switchTab",  # tab switching
    ):
        assert fn in page, f"missing JS handler {fn!r}"
    # POST SSE needs a streaming reader, not EventSource.
    assert "getReader" in page
    assert "/run/stream" in page


def test_chat_tab_still_streams_tokens_minimum():
    """Graceful-degradation floor: the chat tab keeps its /chat/stream wiring."""
    page = _client().get("/").text
    assert "/chat/stream" in page
    assert 'data-tab="chat"' in page


# --- /agent/info introspection endpoint --------------------------------
def test_agent_info_shape():
    r = _client().get("/agent/info")
    assert r.status_code == 200
    info = r.json()
    assert info["name"] == "console"
    assert isinstance(info["model"], str) and info["model"]  # str(agent._model_spec)
    assert info["instructions"] == "Be helpful."
    # Tools are listed as {name, description}.
    names = {t["name"]: t["description"] for t in info["tools"]}
    assert "adder" in names
    assert "Add two integers" in names["adder"]


def test_agent_info_model_is_stringified_spec():
    agent = _agent()
    info = _client(agent).get("/agent/info").json()
    assert info["model"] == str(agent._model_spec)


def test_agent_info_handles_callable_instructions():
    """instructions may be a callable; the endpoint must still return a string-ish value."""

    def dyn(ctx) -> str:
        return "dynamic"

    agent = Agent("c2", model=TestModel("x"), instructions=dyn, registry_id="c2")
    info = TestClient(web_app(agent)).get("/agent/info").json()
    # Callable instructions can't be sent verbatim; endpoint must not 500 and must
    # return a JSON-serializable value (string).
    assert isinstance(info["instructions"], str)


def test_agent_info_empty_tools():
    agent = Agent("bare", model=TestModel("x"), registry_id="bare")
    info = TestClient(web_app(agent)).get("/agent/info").json()
    assert info["tools"] == []


# --- the serve endpoints are mounted through the web app ----------------
def test_run_stream_reachable_through_web_app():
    with _client(_agent("streamed")).stream("POST", "/run/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())
    assert "event: run_start" in body
    assert "event: run_end" in body


def test_runs_endpoint_reachable_through_web_app():
    client = _client(_agent("done"))
    # A completed run registers, so /runs lists it.
    client.post("/run", json={"prompt": "hi"})
    r = client.get("/runs")
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list) and len(items) >= 1
    assert {"id", "status", "started_at"} <= set(items[0])


def test_chat_stream_reachable_through_web_app():
    with _client(_agent("a b c")).stream("POST", "/chat/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "data:" in body and "[DONE]" in body


def test_agent_card_reachable_through_web_app():
    card = _client().get("/.well-known/agent.json").json()
    assert card["name"] == "console"


def test_health_reachable_through_web_app():
    assert _client().get("/health").json()["status"] == "ok"
