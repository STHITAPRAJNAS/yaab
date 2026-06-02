"""Session/run state inspector over HTTP.

The state inspector surfaces a session's structured key-value state so a
developer can see what a run accumulated:

* ``GET /sessions/{id}/state`` returns the KV snapshot for a session;
* ``GET /runs/{id}/state`` resolves the run's session (via the run store) and
  returns that session's state.

Both ``404`` cleanly for an unknown session/run. Offline: an in-memory session
service seeded directly, read back through the served runner's session service.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runner import Runner  # noqa: E402
from yaab.runs.base import RunRecord, RunStatus  # noqa: E402
from yaab.runs.memory import InMemoryRunStore  # noqa: E402
from yaab.serve import fastapi_server_app  # noqa: E402
from yaab.sessions.memory import InMemorySessionService  # noqa: E402


def _agent() -> Agent:
    return Agent("svc", model=TestModel("ok"), registry_id="svc")


def _runner_with_session(session_id: str, state: dict) -> tuple[Runner, InMemorySessionService]:
    """Build a runner whose session service holds a seeded session state."""
    service = InMemorySessionService()

    async def seed() -> None:
        session = await service.get_or_create(session_id)
        session.state.update(state)
        await service.save(session)

    asyncio.run(seed())
    return Runner(session_service=service), service


# --- session state snapshot -------------------------------------------
def test_session_state_returns_kv_snapshot():
    runner, _ = _runner_with_session("s1", {"counter": 3, "name": "alice"})
    client = TestClient(fastapi_server_app(_agent(), runner=runner))
    r = client.get("/sessions/s1/state")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s1"
    assert body["state"] == {"counter": 3, "name": "alice"}


def test_session_state_unknown_session_404():
    runner, _ = _runner_with_session("s1", {"k": "v"})
    client = TestClient(fastapi_server_app(_agent(), runner=runner))
    assert client.get("/sessions/does-not-exist/state").status_code == 404


# --- run -> session state resolution ----------------------------------
def test_run_state_resolves_session_via_run_store():
    runner, _ = _runner_with_session("s1", {"progress": "step-2"})
    store = InMemoryRunStore()

    async def seed_run() -> None:
        now = 1.0
        await store.create(
            RunRecord(
                run_id="r1",
                agent="svc",
                status=RunStatus.RUNNING,
                prompt="hi",
                session_id="s1",
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(seed_run())

    client = TestClient(fastapi_server_app(_agent(), runner=runner, run_store=store))
    r = client.get("/runs/r1/state")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "r1"
    assert body["session_id"] == "s1"
    assert body["state"] == {"progress": "step-2"}


def test_run_state_unknown_run_404():
    runner, _ = _runner_with_session("s1", {"k": "v"})
    store = InMemoryRunStore()
    client = TestClient(fastapi_server_app(_agent(), runner=runner, run_store=store))
    assert client.get("/runs/nope/state").status_code == 404


def test_run_state_run_without_session_404():
    runner, _ = _runner_with_session("s1", {"k": "v"})
    store = InMemoryRunStore()

    async def seed_run() -> None:
        await store.create(
            RunRecord(
                run_id="r2",
                agent="svc",
                status=RunStatus.RUNNING,
                prompt="hi",
                session_id=None,
                created_at=1.0,
                updated_at=1.0,
            )
        )

    asyncio.run(seed_run())
    client = TestClient(fastapi_server_app(_agent(), runner=runner, run_store=store))
    assert client.get("/runs/r2/state").status_code == 404


# --- state endpoints require auth like the rest -----------------------
def test_state_endpoints_enforce_auth():
    from yaab.auth import BearerTokenAuth

    runner, _ = _runner_with_session("s1", {"k": "v"})
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(_agent(), runner=runner, auth=auth))
    assert client.get("/sessions/s1/state").status_code == 401
    assert (
        client.get("/sessions/s1/state", headers={"Authorization": "Bearer secret"}).status_code
        == 200
    )
