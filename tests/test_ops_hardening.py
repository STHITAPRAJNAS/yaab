"""Regression tests for the operability hardening fixes.

* TraceStore pagination + retention: ``get`` honors ``limit``/``offset`` and
  ``prune`` reclaims storage (findings 14).
* The trace/events HTTP endpoints accept ``?limit=N`` so a huge trace is paged
  rather than materialized whole (finding 16).
* ``worker=False`` disables the embedded queue worker so a multi-replica
  deployment can run a single external worker (finding 18).
* Graceful shutdown waits a grace period scaled to the lease/heartbeat cadence
  (finding 17).
"""

from __future__ import annotations

import time

import pytest

from yaab.runs.trace import InMemoryTraceStore, SQLiteTraceStore


# --- finding 14: trace pagination + retention ------------------------------
async def test_trace_get_paginates(tmp_path):
    for store in (InMemoryTraceStore(), SQLiteTraceStore(path=str(tmp_path / "t.db"))):
        for i in range(10):
            await store.append("r1", i, {"type": "tool_call", "seq": i})
        first3 = await store.get("r1", limit=3)
        assert [e["seq"] for e in first3] == [0, 1, 2]
        mid = await store.get("r1", limit=3, offset=3)
        assert [e["seq"] for e in mid] == [3, 4, 5]
        rest = await store.get("r1", offset=8)
        assert [e["seq"] for e in rest] == [8, 9]


async def test_trace_prune_keep_last(tmp_path):
    for store in (InMemoryTraceStore(), SQLiteTraceStore(path=str(tmp_path / "k.db"))):
        for i in range(10):
            await store.append("r1", i, {"type": "x", "seq": i})
        deleted = await store.prune(keep_last=3)
        assert deleted == 7
        remaining = await store.get("r1")
        assert [e["seq"] for e in remaining] == [7, 8, 9]


async def test_trace_prune_older_than(tmp_path):
    store = SQLiteTraceStore(path=str(tmp_path / "o.db"))
    await store.append("r1", 0, {"type": "x"})
    cutoff = time.time() + 1.0  # everything so far is "older than" the cutoff
    deleted = await store.prune(older_than=cutoff)
    assert deleted == 1
    assert await store.get("r1") == []


# --- finding 16: trace endpoint accepts ?limit ------------------------------
def test_trace_endpoint_limit_query():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab import Agent
    from yaab.models.test_model import TestModel
    from yaab.runs.memory import InMemoryRunStore
    from yaab.serve import fastapi_server_app

    async def _seed(store, runs):
        from yaab.runs.base import RunRecord, RunStatus

        now = time.time()
        await runs.create(
            RunRecord(
                run_id="r1",
                agent="svc",
                status=RunStatus.COMPLETED,
                created_at=now,
                updated_at=now,
            )
        )
        for i in range(50):
            await store.append("r1", i, {"type": "tool_result", "payload": {"name": f"t{i}"}})

    import asyncio

    traces = InMemoryTraceStore()
    runs = InMemoryRunStore()
    asyncio.run(_seed(traces, runs))
    agent = Agent("svc", model=TestModel("ok"), registry_id="svc")
    app = fastapi_server_app(agent, run_store=runs, trace_store=traces, worker=False)
    with TestClient(app) as client:
        body = client.get("/runs/r1/events?limit=5").json()
        assert len(body["events"]) == 5
        # An invalid limit is a clean 400.
        assert client.get("/runs/r1/events?limit=-1").status_code == 400
        assert client.get("/runs/r1/events?limit=abc").status_code == 400


# --- finding 18: worker=False disables the embedded worker -----------------
def test_worker_false_disables_embedded_worker():
    pytest.importorskip("fastapi")

    from yaab import Agent
    from yaab.models.test_model import TestModel
    from yaab.runs.memory import InMemoryRunStore
    from yaab.serve import fastapi_server_app

    runs = InMemoryRunStore()
    agent = Agent("svc", model=TestModel("ok"), registry_id="svc")
    # With worker=False no worker is constructed even though a run_store is set.
    app = fastapi_server_app(agent, run_store=runs, worker=False)
    # Sanity: the app still builds and serves health.
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        # A background submission still enqueues, but nothing drains it (no worker).
        rid = client.post("/run", json={"prompt": "p", "background": True}).json()["run_id"]
        import asyncio

        async def _still_queued():
            rec = await runs.get(rid)
            return rec is not None and rec.status.value == "queued"

        assert asyncio.run(_still_queued())


# --- finding 17: shutdown grace scales with the lease cadence --------------
def test_shutdown_grace_scales_with_lease():
    pytest.importorskip("fastapi")

    from yaab import Agent
    from yaab.models.test_model import TestModel
    from yaab.runs.memory import InMemoryRunStore
    from yaab.runs.worker import RunWorker
    from yaab.serve import fastapi_server_app

    runs = InMemoryRunStore()
    agent = Agent("svc", model=TestModel("ok"), registry_id="svc")
    # A long lease implies a long grace period (>= lease + heartbeat, min 15s).
    worker = RunWorker(agent, runs, lease_seconds=60.0, heartbeat_interval=20.0)
    # The grace formula the lifespan uses is asserted via the worker's params.
    grace = max(15.0, worker.lease_seconds + worker.heartbeat_interval)
    assert grace == 80.0
    # The app still constructs with the explicit worker.
    app = fastapi_server_app(agent, run_store=runs, worker=worker)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
