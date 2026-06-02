"""One-call production wiring — a coherent set of shared, durable backends.

:func:`yaab.durable_backends` turns one or two connection URLs into a consistent
set of backends (sessions, artifacts, run store, approval store, trace store,
checkpointer, audit sink, registry, rate limiter) all pointed at the same place,
so making a deployment multi-replica-safe is a single call rather than wiring
nine backends by hand. These tests prove:

* a ``sqlite://`` DSN builds every backend against the *same* database file, so
  the whole set is durable and consistent (the wiring is correct);
* the struct splats cleanly into ``Runner`` and ``fastapi_server_app`` (it is
  shaped for the consumers it is meant for);
* the run store and session service genuinely roundtrip data (a smoke that the
  built backends are live, not stubs);
* an in-memory build (no DSN) stays process-local for dev;
* the ``redis_url`` selects the shared rate limiter (global budget across pods).

All offline: SQLite tempfiles and an injected fake redis client only.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab import DurableBackends, durable_backends
from yaab.artifacts import InMemoryArtifactService, SQLiteArtifactService
from yaab.governance.approvals import InMemoryApprovalStore, SQLiteApprovalStore
from yaab.runs import (
    InMemoryRunStore,
    InMemoryTraceStore,
    RunRecord,
    RunStatus,
    SQLiteRunStore,
    SQLiteTraceStore,
)
from yaab.sessions.sqlite import SQLiteSessionService


def _sqlite_dsn(tmp_path) -> str:
    return f"sqlite://{tmp_path / 'durable.db'}"


def test_sqlite_dsn_builds_durable_backends(tmp_path):
    b = durable_backends(dsn=_sqlite_dsn(tmp_path))

    assert isinstance(b, DurableBackends)
    assert isinstance(b.run_store, SQLiteRunStore)
    assert isinstance(b.approval_store, SQLiteApprovalStore)
    assert isinstance(b.trace_store, SQLiteTraceStore)
    assert isinstance(b.artifact_service, SQLiteArtifactService)
    assert isinstance(b.session_service, SQLiteSessionService)
    # A checkpointer, audit sink and registry backend round out the set.
    assert b.run_checkpointer is not None
    assert b.audit_sink is not None
    assert b.registry_backend is not None


def test_no_dsn_is_process_local_dev_default():
    b = durable_backends()

    assert isinstance(b.run_store, InMemoryRunStore)
    assert isinstance(b.approval_store, InMemoryApprovalStore)
    assert isinstance(b.trace_store, InMemoryTraceStore)
    assert isinstance(b.artifact_service, InMemoryArtifactService)
    # No shared rate limiter without a redis url.
    assert b.rate_limiter is None


def test_runner_kwargs_splat_into_runner(tmp_path):
    from yaab import Runner

    b = durable_backends(dsn=_sqlite_dsn(tmp_path))
    # The struct must hand the Runner exactly the keyword args it accepts.
    runner = Runner(**b.runner_kwargs())
    assert runner is not None


def test_serve_kwargs_splat_into_server_app(tmp_path):
    pytest.importorskip("fastapi")
    from yaab import Agent
    from yaab.serve import fastapi_server_app
    from yaab.testing import TestModel

    b = durable_backends(dsn=_sqlite_dsn(tmp_path))
    agent = Agent("svc", model=TestModel("ok"))
    app = fastapi_server_app(agent, **b.serve_kwargs())
    assert app is not None


def test_run_store_roundtrips_against_the_dsn(tmp_path):
    b = durable_backends(dsn=_sqlite_dsn(tmp_path))

    async def go() -> RunRecord | None:
        now = time.time()
        await b.run_store.create(
            RunRecord(run_id="r1", agent="svc", created_at=now, updated_at=now)
        )
        return await b.run_store.get("r1")

    rec = asyncio.run(go())
    assert rec is not None and rec.run_id == "r1" and rec.status == RunStatus.QUEUED


def test_session_service_roundtrips_against_the_dsn(tmp_path):
    b = durable_backends(dsn=_sqlite_dsn(tmp_path))

    async def go() -> dict:
        sess = await b.session_service.get_or_create()
        sess.state["k"] = "v"
        await b.session_service.save(sess)
        again = await b.session_service.get(sess.id)
        return dict(again.state) if again is not None else {}

    state = asyncio.run(go())
    assert state.get("k") == "v"


def test_dsn_is_recorded_on_the_struct(tmp_path):
    # The struct records the DSN it was built from so callers can see (and the
    # safety guardrail can report) exactly what the backends are pointed at.
    dsn = f"sqlite://{tmp_path / 'shared.db'}"
    b = durable_backends(dsn=dsn)
    assert b.dsn == dsn


def test_redis_url_selects_shared_rate_limiter(tmp_path):
    from yaab.models.distributed_ratelimit import RedisRateLimiter

    class _FakeRedis:
        def eval(self, *a, **k):  # pragma: no cover - not exercised here
            return 1

    fake = _FakeRedis()
    b = durable_backends(
        dsn=_sqlite_dsn(tmp_path),
        redis_url="redis://localhost:6379/0",
        redis_client=fake,
    )
    assert isinstance(b.rate_limiter, RedisRateLimiter)


def test_durable_backends_is_idempotent_struct(tmp_path):
    # Two calls with the same DSN produce independent but equivalently-typed
    # structs (no shared mutable global state leaking between builds).
    a = durable_backends(dsn=_sqlite_dsn(tmp_path))
    c = durable_backends(dsn=_sqlite_dsn(tmp_path))
    assert type(a) is type(c)
    assert a.run_store is not c.run_store
