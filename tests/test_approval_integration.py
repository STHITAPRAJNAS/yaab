"""End-to-end approval integration with durable SQLite stores and trace replay.

* Finding 19: a complete background-run + approve + complete flow against
  *SQLite* approval and run stores (not just in-memory), with a second store
  instance over the same files proving cross-pod visibility of the paused run
  and pending approval.
* Finding 20: a durable trace store captures the ``APPROVAL_REQUIRED`` event and
  the post-resume ``TOOL_RESULT``, so the full approval+resume timeline is
  replayable via ``GET /runs/{id}/trace``.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.governance.approval import ToolApprovalPlugin  # noqa: E402
from yaab.governance.approvals import SQLiteApprovalStore  # noqa: E402
from yaab.graph.checkpoint import SQLiteSaver  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runner import Runner  # noqa: E402
from yaab.runs.sqlite import SQLiteRunStore  # noqa: E402
from yaab.runs.trace import InMemoryTraceStore, SQLiteTraceStore  # noqa: E402
from yaab.serve import fastapi_server_app  # noqa: E402


def _approval_agent() -> Agent:
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


# --- finding 19: SQLite-backed end-to-end approval flow + cross-pod view ---
def test_sqlite_backed_approval_flow_visible_across_instances(tmp_path):
    approvals_path = str(tmp_path / "approvals.db")
    runs_path = str(tmp_path / "runs.db")
    ckpt_path = str(tmp_path / "ckpt.db")

    approvals = SQLiteApprovalStore(path=approvals_path)
    runs = SQLiteRunStore(path=runs_path)
    plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=approvals)
    runner = Runner(run_checkpointer=SQLiteSaver(path=ckpt_path), plugins=[plugin])
    app = fastapi_server_app(
        _approval_agent(), runner=runner, approval_store=approvals, run_store=runs
    )

    with TestClient(app) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})
        pending = _wait_for_pending(client)
        assert pending["tool"] == "wire"

        # A *second* store instance over the same files (the "other pod") sees the
        # paused run and the pending approval it never created.
        runs_b = SQLiteRunStore(path=runs_path)
        approvals_b = SQLiteApprovalStore(path=approvals_path)

        async def _cross_pod_view():
            rec = await runs_b.get(run_id)
            pend = await approvals_b.list_pending()
            return rec, pend

        import asyncio

        rec, pend = asyncio.run(_cross_pod_view())
        assert rec is not None and rec.status.value == "paused"
        assert [p.approval_id for p in pend] == [pending["approval_id"]]

        # Approve -> the in-process worker/resume drives it to completion.
        r = client.post(f"/approvals/{pending['approval_id']}/approve", json={"reviewer": "alice"})
        assert r.status_code == 200
        final = _poll_until(client, run_id, want={"completed"})
        assert final["output"] == "all done"

        # Both store instances now see the terminal run.
        async def _terminal():
            return await runs_b.get(run_id)

        rec_b = asyncio.run(_terminal())
        assert rec_b is not None and rec_b.status.value == "completed"


# --- finding 20: APPROVAL_REQUIRED + TOOL_RESULT persisted in the trace ----
def test_approval_flow_with_trace_store_records_timeline(tmp_path):
    approvals = SQLiteApprovalStore(path=str(tmp_path / "ap.db"))
    runs = SQLiteRunStore(path=str(tmp_path / "runs.db"))
    traces = SQLiteTraceStore(path=str(tmp_path / "trace.db"))
    plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=approvals)
    runner = Runner(
        run_checkpointer=SQLiteSaver(path=str(tmp_path / "ckpt.db")),
        plugins=[plugin],
        trace_store=traces,
    )
    app = fastapi_server_app(
        _approval_agent(),
        runner=runner,
        approval_store=approvals,
        run_store=runs,
        trace_store=traces,
    )

    with TestClient(app) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})
        pending = _wait_for_pending(client)

        # The pause emitted an APPROVAL_REQUIRED event into the durable trace.
        events = client.get(f"/runs/{run_id}/events").json()["events"]
        types = [e.get("type") for e in events]
        assert "approval_required" in types

        client.post(f"/approvals/{pending['approval_id']}/approve", json={"reviewer": "alice"})
        _poll_until(client, run_id, want={"completed"})

        # After resume the trace also carries the TOOL_RESULT for the run tool,
        # and the computed trace surfaces the approval span.
        events = client.get(f"/runs/{run_id}/events").json()["events"]
        types = [e.get("type") for e in events]
        assert "tool_result" in types
        trace = client.get(f"/runs/{run_id}/trace").json()
        span_types = {s["type"] for s in trace["spans"]}
        assert "approval" in span_types
        assert "tool_call" in span_types


# --- a trace store also persists the approval event for the in-memory path --
def test_approval_required_persisted_to_in_memory_trace(tmp_path):
    approvals = SQLiteApprovalStore(path=str(tmp_path / "ap.db"))
    runs = SQLiteRunStore(path=str(tmp_path / "runs.db"))
    traces = InMemoryTraceStore()
    plugin = ToolApprovalPlugin(tools=["wire"], mode="queue", store=approvals)
    runner = Runner(
        run_checkpointer=SQLiteSaver(path=str(tmp_path / "ckpt.db")),
        plugins=[plugin],
        trace_store=traces,
    )
    app = fastapi_server_app(
        _approval_agent(),
        runner=runner,
        approval_store=approvals,
        run_store=runs,
        trace_store=traces,
    )
    with TestClient(app) as client:
        run_id = client.post("/run", json={"prompt": "pay", "background": True}).json()["run_id"]
        _poll_until(client, run_id, want={"paused"})

        import asyncio

        async def _events():
            return await traces.get(run_id)

        events = asyncio.run(_events())
        assert any(e.get("type") == "approval_required" for e in events)
