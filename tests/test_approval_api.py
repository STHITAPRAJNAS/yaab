"""Out-of-band human sign-off over HTTP — list, approve, deny, resume.

A sensitive tool call can be parked for human review instead of blocking: the
run pauses durably (consuming no compute), a pending approval record is
persisted, and a reviewer on any replica lists it, then approves or denies it.
On approve the run resumes and runs the now-signed-off tool; on deny it resumes
with the denial fed back to the model — in both cases without re-requesting the
captured model turns.

Endpoints covered:

* ``GET  /approvals?status=pending`` / ``GET /approvals/{id}``
* ``POST /approvals/{id}/approve`` / ``POST /approvals/{id}/deny``
* ``POST /runs/{id}/resume``

All ``404`` cleanly when no approval store is configured.

Offline: a ``TestModel`` that calls a guarded tool, an in-memory approval +
run store, a ``MemorySaver`` checkpointer, and a ``TestClient`` whose lifespan
runs the in-process worker that drains and pauses the background run.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.governance.approval import ToolApprovalPlugin  # noqa: E402
from yaab.governance.approvals import InMemoryApprovalStore  # noqa: E402
from yaab.graph.checkpoint import MemorySaver  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runner import Runner  # noqa: E402
from yaab.runs.memory import InMemoryRunStore  # noqa: E402
from yaab.serve import fastapi_server_app  # noqa: E402


def _plain_agent() -> Agent:
    return Agent("svc", model=TestModel("ok"), registry_id="svc")


def _approval_agent() -> Agent:
    """An agent whose first turn calls a guarded ``wire`` tool, then answers."""

    @tool
    async def wire(ctx, amount: int = 100) -> str:
        """move money"""
        return f"wired {amount}"

    return Agent(
        "svc",
        model=TestModel(custom_output="all done", call_tools=["wire"]),
        tools=[wire],
        registry_id="svc",
    )


def _approval_app(agent: Agent, approval_store, run_store):
    """Build a served app whose runner queues guarded tools for sign-off."""
    plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=approval_store)
    runner = Runner(run_checkpointer=MemorySaver(), plugins=[plugin])
    return fastapi_server_app(
        agent,
        runner=runner,
        approval_store=approval_store,
        run_store=run_store,
    )


def _poll_until(client: TestClient, run_id: str, *, want: set[str], timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        last = r.json()
        if last["status"] in want:
            return last
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached {want}; last={last}")


def _wait_for_pending(client: TestClient, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = client.get("/approvals?status=pending").json()
        if pending:
            return pending[0]
        time.sleep(0.01)
    raise AssertionError("no pending approval appeared")


# --- endpoints 404 cleanly without an approval store ------------------
def test_approval_endpoints_404_without_store():
    client = TestClient(fastapi_server_app(_plain_agent()))
    assert client.get("/approvals").status_code == 404
    assert client.get("/approvals/x").status_code == 404
    assert client.post("/approvals/x/approve").status_code == 404
    assert client.post("/approvals/x/deny").status_code == 404


def test_resume_404_without_run_store():
    approvals = InMemoryApprovalStore()
    client = TestClient(fastapi_server_app(_plain_agent(), approval_store=approvals))
    assert client.post("/runs/r1/resume").status_code == 404


# --- a guarded background run pauses and surfaces a pending approval ----
def test_background_run_pauses_and_lists_pending():
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    with TestClient(_approval_app(_approval_agent(), approvals, runs)) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})

        pending = _wait_for_pending(client)
        assert pending["tool"] == "wire"
        assert pending["decision"] == "pending"

        # The single-request fetch returns the tool + arguments for the reviewer.
        one = client.get(f"/approvals/{pending['approval_id']}").json()
        assert one["approval_id"] == pending["approval_id"]
        assert one["tool"] == "wire"


# --- approve -> run resumes, runs the tool, completes -----------------
def test_approve_resumes_and_completes():
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    with TestClient(_approval_app(_approval_agent(), approvals, runs)) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})
        pending = _wait_for_pending(client)

        r = client.post(
            f"/approvals/{pending['approval_id']}/approve",
            json={"reviewer": "alice"},
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "approved"
        assert r.json()["reviewer"] == "alice"

        final = _poll_until(client, run_id, want={"completed"})
        assert final["output"] == "all done"
        # The approval is no longer pending.
        assert client.get("/approvals?status=pending").json() == []


# --- deny -> run resumes with the denial, tool never runs -------------
def test_deny_resumes_with_denial():
    seen: dict[str, bool] = {"ran": False}

    @tool
    async def wire(ctx, amount: int = 100) -> str:
        """move money"""
        seen["ran"] = True
        return f"wired {amount}"

    agent = Agent(
        "svc",
        model=TestModel(custom_output="acknowledged", call_tools=["wire"]),
        tools=[wire],
        registry_id="svc",
    )
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    with TestClient(_approval_app(agent, approvals, runs)) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})
        pending = _wait_for_pending(client)

        r = client.post(
            f"/approvals/{pending['approval_id']}/deny",
            json={"reviewer": "bob", "reason": "too risky"},
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "denied"

        final = _poll_until(client, run_id, want={"completed"})
        # The model still produced its answer; the guarded tool never executed.
        assert final["output"] == "acknowledged"
        assert seen["ran"] is False


# --- approve/deny of an unknown approval 404 --------------------------
def test_decide_unknown_approval_404():
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    client = TestClient(_approval_app(_approval_agent(), approvals, runs))
    assert client.post("/approvals/nope/approve").status_code == 404
    assert client.post("/approvals/nope/deny").status_code == 404


# --- manual resume endpoint re-runs a paused run ----------------------
def test_manual_resume_endpoint():
    approvals = InMemoryApprovalStore()
    runs = InMemoryRunStore()
    with TestClient(_approval_app(_approval_agent(), approvals, runs)) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})
        # Resume with an explicit approved decision (idempotent manual hook).
        r = client.post(f"/runs/{run_id}/resume", json={"decision": "approved"})
        assert r.status_code == 200
        final = _poll_until(client, run_id, want={"completed"})
        assert final["output"] == "all done"
