"""Regression tests for the serve-layer security fixes.

* Cross-tenant isolation: identity A cannot read identity B's run, trace,
  events, session state, or resume it (findings 10).
* Reviewer authorization: an approval pinned to ``allowed_reviewers`` can only
  be decided by an allowed identity (finding 11).
* Argument redaction: secret-looking tool arguments are masked in the trace /
  events responses (finding 12).
* SSRF: webhook URLs to loopback / link-local / private hosts and non-https
  schemes are rejected (finding 13).
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent  # noqa: E402
from yaab.auth import BearerTokenAuth  # noqa: E402
from yaab.governance.approvals import ApprovalRequest, InMemoryApprovalStore  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runs.base import RunRecord, RunStatus  # noqa: E402
from yaab.runs.memory import InMemoryRunStore  # noqa: E402
from yaab.runs.trace import InMemoryTraceStore  # noqa: E402
from yaab.serve import (  # noqa: E402
    _redact_arguments,
    _validate_webhook,
    fastapi_server_app,
)


def _agent() -> Agent:
    return Agent("svc", model=TestModel("ok"), registry_id="svc")


def _auth() -> BearerTokenAuth:
    return BearerTokenAuth({"tok-a": "alice", "tok-b": "bob"})


def _hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _run_record(run_id: str, identity: str, session_id: str | None = None) -> RunRecord:
    now = time.time()
    return RunRecord(
        run_id=run_id,
        agent="svc",
        status=RunStatus.RUNNING,
        prompt="hi",
        identity=identity,
        session_id=session_id,
        created_at=now,
        updated_at=now,
    )


# --- finding 10: cross-tenant run/trace/events/state isolation -------------
async def test_get_run_cross_tenant_forbidden():
    runs = InMemoryRunStore()
    await runs.create(_run_record("r1", "alice"))
    app = fastapi_server_app(_agent(), auth=_auth(), run_store=runs, worker=False)
    with TestClient(app) as client:
        # The owner reads their run.
        assert client.get("/runs/r1", headers=_hdr("tok-a")).status_code == 200
        # Another identity is forbidden, not served the data.
        assert client.get("/runs/r1", headers=_hdr("tok-b")).status_code == 403


async def test_trace_and_events_cross_tenant_forbidden():
    runs = InMemoryRunStore()
    traces = InMemoryTraceStore()
    await runs.create(_run_record("r1", "alice"))
    await traces.append("r1", 0, {"type": "tool_call", "payload": {"name": "wire"}})
    app = fastapi_server_app(
        _agent(), auth=_auth(), run_store=runs, trace_store=traces, worker=False
    )
    with TestClient(app) as client:
        assert client.get("/runs/r1/trace", headers=_hdr("tok-a")).status_code == 200
        assert client.get("/runs/r1/trace", headers=_hdr("tok-b")).status_code == 403
        assert client.get("/runs/r1/events", headers=_hdr("tok-a")).status_code == 200
        assert client.get("/runs/r1/events", headers=_hdr("tok-b")).status_code == 403


async def test_session_state_cross_tenant_forbidden():
    runs = InMemoryRunStore()
    await runs.create(_run_record("r1", "alice", session_id="sess1"))
    app = fastapi_server_app(_agent(), auth=_auth(), run_store=runs, worker=False)
    with TestClient(app) as client:
        # bob owns no run on sess1 -> forbidden.
        assert client.get("/sessions/sess1/state", headers=_hdr("tok-b")).status_code == 403


async def test_resume_cross_tenant_forbidden():
    runs = InMemoryRunStore()
    await runs.create(_run_record("r1", "alice"))
    app = fastapi_server_app(_agent(), auth=_auth(), run_store=runs, worker=False)
    with TestClient(app) as client:
        assert client.post("/runs/r1/resume", headers=_hdr("tok-b")).status_code == 403


# --- finding 12: argument redaction ----------------------------------------
def test_redact_arguments_masks_secret_like_keys():
    out = _redact_arguments(
        {"to": "x@y.com", "api_key": "secret123", "nested": {"password": "p", "ok": 1}}
    )
    assert out["to"] == "x@y.com"
    assert out["api_key"] == "***redacted***"
    assert out["nested"]["password"] == "***redacted***"
    assert out["nested"]["ok"] == 1


async def test_events_endpoint_redacts_arguments():
    runs = InMemoryRunStore()
    traces = InMemoryTraceStore()
    await runs.create(_run_record("r1", "alice"))
    await traces.append(
        "r1",
        0,
        {"type": "tool_call", "payload": {"name": "send", "arguments": {"api_key": "secret123"}}},
    )
    app = fastapi_server_app(
        _agent(), auth=_auth(), run_store=runs, trace_store=traces, worker=False
    )
    with TestClient(app) as client:
        body = client.get("/runs/r1/events", headers=_hdr("tok-a")).json()
        args = body["events"][0]["payload"]["arguments"]
        assert args["api_key"] == "***redacted***"
        assert "secret123" not in str(body)


# --- finding 11: reviewer authorization ------------------------------------
async def test_approval_restricted_reviewer_enforced():
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    await approvals.create(
        ApprovalRequest(
            approval_id="ap1",
            run_id="r1",
            resume_id="r1",
            agent="svc",
            tool="wire",
            allowed_reviewers=["alice"],
        )
    )
    app = fastapi_server_app(
        _agent(), auth=_auth(), approval_store=approvals, run_store=runs, worker=False
    )
    with TestClient(app) as client:
        # bob is not an allowed reviewer -> 403, decision untouched.
        r = client.post("/approvals/ap1/approve", headers=_hdr("tok-b"))
        assert r.status_code == 403
        still = client.get("/approvals/ap1", headers=_hdr("tok-a")).json()
        assert still["decision"] == "pending"
        # alice is allowed -> the decision records her identity as the reviewer.
        r = client.post("/approvals/ap1/approve", headers=_hdr("tok-a"))
        assert r.status_code == 200
        assert r.json()["decision"] == "approved"
        assert r.json()["reviewer"] == "alice"


async def test_unrestricted_approval_any_authenticated_reviewer():
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    await approvals.create(
        ApprovalRequest(approval_id="ap1", run_id="r1", resume_id="r1", agent="svc", tool="wire")
    )
    app = fastapi_server_app(
        _agent(), auth=_auth(), approval_store=approvals, run_store=runs, worker=False
    )
    with TestClient(app) as client:
        r = client.post("/approvals/ap1/deny", headers=_hdr("tok-b"))
        assert r.status_code == 200
        assert r.json()["reviewer"] == "bob"  # authenticated identity, not body


# --- finding 13: SSRF webhook validation -----------------------------------
def test_validate_webhook_rejects_dangerous_targets():
    bad = [
        "http://example.com/hook",  # not https
        "https://localhost/hook",
        "https://127.0.0.1/hook",
        "https://169.254.169.254/latest/meta-data",  # cloud metadata
        "https://10.0.0.5/hook",
        "https://192.168.1.1/hook",
        "https://0.0.0.0/hook",
        "gopher://internal/cmd",
    ]
    for url in bad:
        with pytest.raises(ValueError):
            _validate_webhook(url)


def test_validate_webhook_allows_public_https_and_none():
    assert _validate_webhook(None) is None
    assert _validate_webhook("") == ""
    assert _validate_webhook("https://hooks.example.com/x") == "https://hooks.example.com/x"


def test_post_run_rejects_ssrf_webhook():
    runs = InMemoryRunStore()
    app = fastapi_server_app(_agent(), auth=_auth(), run_store=runs, worker=False)
    with TestClient(app) as client:
        r = client.post(
            "/run",
            headers=_hdr("tok-a"),
            json={"prompt": "p", "background": True, "webhook": "https://127.0.0.1/x"},
        )
        assert r.status_code == 400


def test_post_cron_rejects_ssrf_webhook():
    from yaab.runs.cron import InMemoryCronStore

    crons = InMemoryCronStore()
    app = fastapi_server_app(_agent(), auth=_auth(), cron_store=crons, worker=False)
    with TestClient(app) as client:
        r = client.post(
            "/crons",
            headers=_hdr("tok-a"),
            json={
                "schedule": "every 1 minute",
                "prompt": "p",
                "webhook": "https://10.0.0.1/x",
            },
        )
        assert r.status_code == 400
