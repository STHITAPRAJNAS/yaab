"""Re-attach to a run's event stream + durable schedules over HTTP.

``GET /runs/{id}/stream`` decouples a request's lifetime from a run's: a caller
joins an in-flight or finished background run and receives the events it already
emitted (replayed from the trace store) followed by a terminal ``done`` marker
once the run record reaches a terminal status. ``POST/GET/DELETE /crons`` manage
durable schedules that the worker materializes into queued runs.

Both groups ``404`` cleanly when their backing store is absent.

Offline: an in-memory trace + run + cron store, seeded directly, exercised
through a ``TestClient``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent  # noqa: E402
from yaab.models.test_model import TestModel  # noqa: E402
from yaab.runs.base import RunRecord, RunStatus  # noqa: E402
from yaab.runs.cron import InMemoryCronStore  # noqa: E402
from yaab.runs.memory import InMemoryRunStore  # noqa: E402
from yaab.runs.trace import InMemoryTraceStore  # noqa: E402
from yaab.serve import fastapi_server_app  # noqa: E402


def _agent() -> Agent:
    return Agent("svc", model=TestModel("ok"), registry_id="svc")


def _seed_trace(trace: InMemoryTraceStore, run_id: str) -> None:
    async def go() -> None:
        events = [
            {"type": "run_start", "payload": {"prompt": "hi"}, "seq": 0, "timestamp": 1.0},
            {
                "type": "model_response",
                "payload": {"model": "test", "finish_reason": "stop"},
                "seq": 1,
                "timestamp": 1.1,
            },
            {
                "type": "run_end",
                "payload": {"result": {"output": "done"}},
                "seq": 2,
                "timestamp": 1.2,
            },
        ]
        for ev in events:
            await trace.append(run_id, ev["seq"], ev)

    asyncio.run(go())


def _seed_terminal_run(store: InMemoryRunStore, run_id: str) -> None:
    async def go() -> None:
        await store.create(
            RunRecord(
                run_id=run_id,
                agent="svc",
                status=RunStatus.COMPLETED,
                prompt="hi",
                created_at=1.0,
                updated_at=1.0,
                finished_at=1.2,
            )
        )

    asyncio.run(go())


# --- join stream replays the persisted trace then ends ----------------
def test_join_stream_replays_persisted_trace():
    trace = InMemoryTraceStore()
    runs = InMemoryRunStore()
    _seed_trace(trace, "r1")
    _seed_terminal_run(runs, "r1")

    client = TestClient(fastapi_server_app(_agent(), trace_store=trace, run_store=runs))
    with client.stream("GET", "/runs/r1/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    # The full persisted trace is replayed, terminated by a done marker.
    assert "event: run_start" in body
    assert "event: model_response" in body
    assert "event: run_end" in body
    assert "event: done" in body
    # The replayed event data is the persisted JSON-safe payload.
    payloads = [
        json.loads(line[5:])
        for line in body.splitlines()
        if line.startswith("data:") and line[5:].strip().startswith("{")
    ]
    assert any(p.get("type") == "run_end" for p in payloads)


def test_join_stream_terminates_on_terminal_run_via_run_end_event():
    # No run store: the stream stops once the trace shows a run_end event.
    trace = InMemoryTraceStore()
    _seed_trace(trace, "r1")
    client = TestClient(fastapi_server_app(_agent(), trace_store=trace))
    with client.stream("GET", "/runs/r1/stream") as resp:
        body = "".join(resp.iter_text())
    assert "event: run_end" in body
    assert "event: done" in body


def test_join_stream_404_without_trace_store():
    client = TestClient(fastapi_server_app(_agent()))
    with client.stream("GET", "/runs/r1/stream") as resp:
        assert resp.status_code == 404


# --- crons ------------------------------------------------------------
def test_crons_create_list_delete():
    crons = InMemoryCronStore()
    client = TestClient(fastapi_server_app(_agent(), cron_store=crons))

    r = client.post(
        "/crons",
        json={"schedule": "every 5 minutes", "prompt": "nightly", "cron_id": "c1"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["cron_id"] == "c1"
    assert body["schedule"] == "every 5 minutes"
    assert body["agent"] == "svc"  # defaulted to the served agent

    listing = client.get("/crons").json()
    assert any(c["cron_id"] == "c1" for c in listing)

    d = client.delete("/crons/c1")
    assert d.status_code == 200 and d.json()["deleted"] is True
    assert client.get("/crons").json() == []


def test_cron_invalid_schedule_400():
    crons = InMemoryCronStore()
    client = TestClient(fastapi_server_app(_agent(), cron_store=crons))
    r = client.post("/crons", json={"schedule": "0 0 * * *", "prompt": "x"})
    assert r.status_code == 400


def test_cron_delete_unknown_404():
    crons = InMemoryCronStore()
    client = TestClient(fastapi_server_app(_agent(), cron_store=crons))
    assert client.delete("/crons/nope").status_code == 404


def test_crons_404_without_store():
    client = TestClient(fastapi_server_app(_agent()))
    assert client.get("/crons").status_code == 404
    assert client.post("/crons", json={"schedule": "every 1 minute"}).status_code == 404
    assert client.delete("/crons/x").status_code == 404


# --- crons enforce auth -----------------------------------------------
def test_crons_enforce_auth():
    from yaab.auth import BearerTokenAuth

    crons = InMemoryCronStore()
    auth = BearerTokenAuth({"secret": "alice"})
    client = TestClient(fastapi_server_app(_agent(), cron_store=crons, auth=auth))
    assert client.get("/crons").status_code == 401
    assert client.get("/crons", headers={"Authorization": "Bearer secret"}).status_code == 200
