"""Comprehensive test of the out-of-the-box FastAPI server (yaab.serve).

Covers every endpoint fastapi_server_app exposes — health, A2A discovery card,
/run, /run/stream (SSE events), /chat/stream (token SSE), A2A task submit + poll
+ 404 + stream — plus pluggable auth enforcement (401/200) and identity flow.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.auth import BearerTokenAuth  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.serve import fastapi_server_app  # noqa: E402


def _agent(out: str = "served-output") -> Agent:
    return Agent("svc", model=TestModel(out), registry_id="svc")


# --- discovery + health ------------------------------------------------
def test_health():
    client = TestClient(fastapi_server_app(_agent()))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["agent"] == "svc"


def test_agent_card_discovery():
    client = TestClient(fastapi_server_app(_agent(), base_url="http://svc"))
    r = client.get("/.well-known/agent.json")
    assert r.status_code == 200
    card = r.json()
    assert card["name"] == "svc"
    assert "x-yaab-governance" in card
    assert "securitySchemes" in card  # auth scheme advertised


# --- /run --------------------------------------------------------------
def test_run_endpoint():
    client = TestClient(fastapi_server_app(_agent("hello from run")))
    r = client.post("/run", json={"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["output"] == "hello from run"
    assert "run_id" in body and "usage" in body
    assert body["usage"]["requests"] >= 1


def test_run_accepts_input_alias():
    client = TestClient(fastapi_server_app(_agent("ok")))
    assert client.post("/run", json={"input": "hi"}).json()["output"] == "ok"


# --- auth enforcement --------------------------------------------------
def test_auth_rejects_missing_token():
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(_agent(), auth=auth))
    # No Authorization header -> 401.
    assert client.post("/run", json={"prompt": "hi"}).status_code == 401


def test_auth_rejects_bad_token():
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(_agent(), auth=auth))
    r = client.post("/run", json={"prompt": "hi"}, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_auth_accepts_valid_token_and_flows_identity():
    seen = {}

    @tool
    def whoami(ctx) -> str:
        """report identity"""
        seen["id"] = ctx.identity
        return ctx.identity or "none"

    agent = Agent("svc", model=TestModel(custom_output="done", call_tools=["whoami"]),
                  tools=[whoami], registry_id="svc")
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(agent, auth=auth))
    r = client.post("/run", json={"prompt": "hi"}, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    assert seen["id"] == "alice"  # identity flowed from the token into the run context


def test_card_advertises_configured_scheme():
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(_agent(), auth=auth))
    card = client.get("/.well-known/agent.json").json()
    assert "bearer" in card["securitySchemes"]


# --- A2A task endpoints ------------------------------------------------
def test_a2a_task_submit_poll_and_404():
    client = TestClient(fastapi_server_app(_agent("a2a-result")))
    # Submit a task.
    r = client.post("/a2a/tasks", json={"message": {"parts": [{"text": "do it"}]}})
    assert r.status_code == 200
    task = r.json()
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"][0]["text"] == "a2a-result"
    tid = task["id"]
    # Poll it back.
    got = client.get(f"/a2a/tasks/{tid}")
    assert got.status_code == 200 and got.json()["id"] == tid
    # Unknown task -> 404.
    assert client.get("/a2a/tasks/nope").status_code == 404


def test_a2a_task_respects_explicit_id():
    client = TestClient(fastapi_server_app(_agent()))
    r = client.post("/a2a/tasks", json={"id": "my-task", "message": {"parts": [{"text": "x"}]}})
    assert r.json()["id"] == "my-task"


# --- SSE streaming endpoints -------------------------------------------
def test_run_stream_sse_events():
    client = TestClient(fastapi_server_app(_agent("streamed")))
    with client.stream("POST", "/run/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())
    assert "event: run_start" in body
    assert "event: run_end" in body
    assert "event: done" in body


def test_chat_stream_tokens():
    client = TestClient(fastapi_server_app(_agent("one two three")))
    with client.stream("POST", "/chat/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    # Token deltas arrive as SSE data lines, terminated by [DONE].
    assert "data:" in body and "[DONE]" in body


def test_a2a_task_stream_status_events():
    client = TestClient(fastapi_server_app(_agent("final-answer")))
    with client.stream(
        "POST", "/a2a/tasks/stream", json={"message": {"parts": [{"text": "hi"}]}}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    # working -> completed task, then done.
    assert '"state": "working"' in body
    assert '"state": "completed"' in body
    payloads = [json.loads(line[5:]) for line in body.splitlines()
                if line.startswith("data:") and line[5:].strip().startswith("{")]
    assert any(p.get("status", {}).get("state") == "completed" for p in payloads)
