"""Run lifecycle management on the FastAPI server.

Covers the gap-closing endpoints that let a remote caller submit a run in the
background, poll its status, list runs, and cancel an in-flight run (the ADK
"cancel runs" / OpenAI "interrupt active run" parity feature):

* ``POST /run`` with ``{"background": true}`` -> 202 + ``{run_id, status}``;
* ``GET  /runs/{run_id}``  -> status (+ output/usage or error when finished);
* ``POST /runs/{run_id}/cancel`` -> cancel an in-flight run (no-op when done);
* ``GET  /runs`` -> recent runs, newest first.

The sync ``POST /run`` keeps its exact prior contract while also registering
itself so it too is cancellable mid-flight from another request.

Background runs require the persistent event loop the ``TestClient`` keeps while
used as a context manager (``with TestClient(app) as client``); a per-request
loop would tear down the ``asyncio.create_task`` the moment the 202 returns.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent, tool  # noqa: E402
from yaab.auth import BearerTokenAuth  # noqa: E402
from yaab.exceptions import RunCancelled  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.serve import fastapi_server_app  # noqa: E402


def _agent(out: str = "served-output") -> Agent:
    return Agent("svc", model=TestModel(out), registry_id="svc")


def _poll_until(client: TestClient, run_id: str, *, want: set[str], timeout: float = 5.0) -> dict:
    """Poll GET /runs/{run_id} until status is in ``want`` or we time out."""
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


# --- sync /run is unchanged -------------------------------------------
def test_sync_run_unchanged_contract():
    """The non-background path keeps its exact response shape (200 + fields)."""
    client = TestClient(fastapi_server_app(_agent("hello from run")))
    r = client.post("/run", json={"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["output"] == "hello from run"
    assert "run_id" in body and "usage" in body
    assert body["usage"]["requests"] >= 1


def test_sync_run_is_registered_and_pollable():
    """A completed sync run is queryable afterwards (it registered itself)."""
    client = TestClient(fastapi_server_app(_agent("done")))
    body = client.post("/run", json={"prompt": "hi"}).json()
    got = client.get(f"/runs/{body['run_id']}")
    assert got.status_code == 200
    assert got.json()["status"] == "completed"
    assert got.json()["output"] == "done"


# --- background submission + polling ----------------------------------
def test_background_run_accepted_and_completes():
    with TestClient(fastapi_server_app(_agent("bg-output"))) as client:
        r = client.post("/run", json={"prompt": "hi", "background": True})
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "running"
        run_id = body["run_id"]

        final = _poll_until(client, run_id, want={"completed"})
        assert final["output"] == "bg-output"
        assert final["usage"]["requests"] >= 1


def test_background_run_unknown_id_404():
    client = TestClient(fastapi_server_app(_agent()))
    assert client.get("/runs/nope").status_code == 404


# --- cancellation of an in-flight background run ----------------------
class _Gate:
    """Cross-call handle holding asyncio events created on the app's own loop.

    The tool lazily creates the events on first entry (so they bind to the loop
    the app runs on), records them here, signals ``released``, then parks on
    ``gate``. The test reaches them via the portal — the only safe way to touch
    a loop-bound primitive from the test thread.
    """

    def __init__(self) -> None:
        self.gate: asyncio.Event | None = None
        self.released: asyncio.Event | None = None

    def ensure(self) -> None:
        if self.gate is None:
            self.gate = asyncio.Event()
            self.released = asyncio.Event()


def _slow_agent(handle: _Gate) -> Agent:
    @tool
    async def slow(ctx) -> str:
        """block until the test releases the gate"""
        handle.ensure()
        assert handle.released is not None and handle.gate is not None
        handle.released.set()
        await handle.gate.wait()
        return "tool-finished"

    return Agent(
        "svc",
        model=TestModel(custom_output="done", call_tools=["slow"]),
        tools=[slow],
        registry_id="svc",
    )


def test_cancel_inflight_background_run():
    """Cancelling a parked background run flips it to 'cancelled' and the
    underlying run raises RunCancelled (surfaced as the registry error)."""
    handle = _Gate()
    agent = _slow_agent(handle)
    with TestClient(fastapi_server_app(agent)) as client:
        run_id = client.post("/run", json={"prompt": "hi", "background": True}).json()["run_id"]

        # Wait (on the app loop) for the run to be parked inside the slow tool.
        async def _wait_released() -> None:
            for _ in range(500):
                if handle.released is not None and handle.released.is_set():
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("run never reached the slow tool")

        client.portal.call(_wait_released)

        # Cancel it; the run is still parked, so this is a live cancellation.
        r = client.post(f"/runs/{run_id}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] in {"cancelled", "running"}

        # Release the gate so the tool returns and the loop reaches the next
        # cancellation checkpoint, where RunCancelled is raised.
        async def _release() -> None:
            assert handle.gate is not None
            handle.gate.set()

        client.portal.call(_release)

        final = _poll_until(client, run_id, want={"cancelled"})
        assert final["status"] == "cancelled"
        # The internal RunCancelled is recorded as the error for the run.
        assert "cancel" in (final.get("error") or "").lower()


def test_cancel_finished_run_is_noop():
    client = TestClient(fastapi_server_app(_agent("done")))
    run_id = client.post("/run", json={"prompt": "hi"}).json()["run_id"]
    # Run already completed synchronously; cancelling is a no-op.
    r = client.post(f"/runs/{run_id}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    # Status is unchanged on a second read.
    assert client.get(f"/runs/{run_id}").json()["status"] == "completed"


def test_cancel_unknown_run_404():
    client = TestClient(fastapi_server_app(_agent()))
    assert client.post("/runs/nope/cancel").status_code == 404


def test_runcancelled_is_importable_and_carries_reason():
    """Guard: RunCancelled carries a reason so the registry error is meaningful."""
    exc = RunCancelled("run api_cancel", reason="api_cancel")
    assert exc.reason == "api_cancel"


# --- /runs listing -----------------------------------------------------
def test_list_runs_newest_first():
    client = TestClient(fastapi_server_app(_agent("ok")))
    first = client.post("/run", json={"prompt": "a"}).json()["run_id"]
    second = client.post("/run", json={"prompt": "b"}).json()["run_id"]

    listing = client.get("/runs").json()
    assert isinstance(listing, list)
    ids = [item["id"] for item in listing]
    assert first in ids and second in ids
    # Newest first: 'second' precedes 'first'.
    assert ids.index(second) < ids.index(first)
    for item in listing:
        assert {"id", "status", "started_at"} <= set(item)


# --- auth enforcement on the new endpoints ----------------------------
def test_auth_enforced_on_run_management_endpoints():
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(_agent(), auth=auth))
    hdr = {"Authorization": "Bearer secret"}

    # Submit a sync run *with* auth so there is a real id to probe.
    run_id = client.post("/run", json={"prompt": "hi"}, headers=hdr).json()["run_id"]

    # Without a token every new endpoint is 401.
    assert client.get("/runs").status_code == 401
    assert client.get(f"/runs/{run_id}").status_code == 401
    assert client.post(f"/runs/{run_id}/cancel").status_code == 401

    # With a valid token they succeed.
    assert client.get("/runs", headers=hdr).status_code == 200
    assert client.get(f"/runs/{run_id}", headers=hdr).status_code == 200
    assert client.post(f"/runs/{run_id}/cancel", headers=hdr).status_code == 200
