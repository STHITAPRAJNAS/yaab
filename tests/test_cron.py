"""Scheduled runs — a durable cron row materializes a queued run when due.

A schedule is just another durable row: ``(cron_id, schedule, prompt, agent,
enabled, next_run_at)``. On each worker tick, every schedule whose next-run time
has arrived is turned into exactly one queued run and its next-run time is rolled
forward, so a tick is idempotent for a given moment and a disabled schedule is
inert. Schedule parsing stays deliberately small ("every N seconds/minutes/
hours" plus a fixed interval in seconds); richer expressions can come later
without a new dependency.

All offline: in-memory and SQLite cron stores, an in-memory run store, no sleeps.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab import Agent
from yaab.models.test_model import TestModel
from yaab.runs.base import RunStatus
from yaab.runs.cron import (
    CronRecord,
    CronStore,
    InMemoryCronStore,
    SQLiteCronStore,
    parse_schedule,
)
from yaab.runs.memory import InMemoryRunStore
from yaab.runs.worker import RunWorker


def _agent() -> Agent:
    return Agent("svc", model=TestModel("cron-output"), registry_id="svc")


def _memory() -> InMemoryCronStore:
    return InMemoryCronStore()


def _sqlite(tmp_path) -> SQLiteCronStore:
    return SQLiteCronStore(str(tmp_path / "crons.db"))


# --- schedule parsing -------------------------------------------------
def test_parse_schedule_units():
    assert parse_schedule("every 30 seconds") == 30.0
    assert parse_schedule("every 5 minutes") == 300.0
    assert parse_schedule("every 2 hours") == 7200.0
    # Singular forms work too.
    assert parse_schedule("every 1 minute") == 60.0
    assert parse_schedule("every 1 hour") == 3600.0


def test_parse_schedule_fixed_interval_seconds():
    # A bare number is a fixed interval in seconds.
    assert parse_schedule("45") == 45.0
    assert parse_schedule("@every 10s") == 10.0


def test_parse_schedule_rejects_unknown():
    with pytest.raises(ValueError):
        parse_schedule("0 0 * * *")  # full cron-expression: not yet supported
    with pytest.raises(ValueError):
        parse_schedule("every banana minutes")


# --- store protocol conformance ---------------------------------------
def test_backends_satisfy_protocol(tmp_path):
    assert isinstance(_memory(), CronStore)
    assert isinstance(_sqlite(tmp_path), CronStore)


# --- create / get / list / delete -------------------------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_create_get_list_delete(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        now = time.time()
        rec = CronRecord(
            cron_id="c1",
            schedule="every 5 minutes",
            prompt="summarize",
            agent="svc",
            enabled=True,
            next_run_at=now,
            created_at=now,
        )
        await store.create(rec)
        got = await store.get("c1")
        assert got is not None and got.schedule == "every 5 minutes"

        listed = await store.list()
        assert [c.cron_id for c in listed] == ["c1"]

        assert await store.delete("c1") is True
        assert await store.get("c1") is None
        assert await store.delete("c1") is False

    asyncio.run(go())


# --- due() returns only crons whose next_run_at has arrived -----------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_due_filters_by_next_run_and_enabled(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        now = time.time()
        await store.create(
            CronRecord(
                cron_id="past",
                schedule="every 1 minute",
                prompt="p",
                agent="svc",
                enabled=True,
                next_run_at=now - 10,
                created_at=now,
            )
        )
        await store.create(
            CronRecord(
                cron_id="future",
                schedule="every 1 minute",
                prompt="p",
                agent="svc",
                enabled=True,
                next_run_at=now + 3600,
                created_at=now,
            )
        )
        await store.create(
            CronRecord(
                cron_id="disabled",
                schedule="every 1 minute",
                prompt="p",
                agent="svc",
                enabled=False,
                next_run_at=now - 10,
                created_at=now,
            )
        )
        due = await store.due(now=now)
        assert [c.cron_id for c in due] == ["past"]

    asyncio.run(go())


# --- mark_run advances next_run_at by the interval --------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_mark_run_rolls_forward(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        now = 1000.0
        await store.create(
            CronRecord(
                cron_id="c1",
                schedule="every 30 seconds",
                prompt="p",
                agent="svc",
                enabled=True,
                next_run_at=now,
                created_at=now,
            )
        )
        await store.mark_run("c1", now=now)
        rec = await store.get("c1")
        assert rec is not None
        assert rec.next_run_at == now + 30.0
        assert rec.last_run_at == now
        # No longer due at this instant.
        assert await store.due(now=now) == []

    asyncio.run(go())


# --- cron_tick materializes exactly one queued run per due cron -------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cron_tick_materializes_exactly_once(backend, tmp_path):
    cron_store = _memory() if backend == "memory" else _sqlite(tmp_path)
    run_store = InMemoryRunStore()
    worker = RunWorker(_agent(), run_store, cron_store=cron_store)

    async def go() -> None:
        now = time.time()
        await cron_store.create(
            CronRecord(
                cron_id="c1",
                schedule="every 5 minutes",
                prompt="scheduled work",
                agent="svc",
                enabled=True,
                next_run_at=now - 1,
                created_at=now,
            )
        )
        created = await worker.cron_tick(now=now)
        assert len(created) == 1

        queued = await run_store.list(status=RunStatus.QUEUED)
        assert len(queued) == 1
        run = queued[0]
        assert run.prompt == "scheduled work"
        assert run.agent == "svc"
        assert run.background is True

        # A second tick at the same instant materializes nothing — the cron rolled
        # forward, so it is no longer due (exactly-once per due window).
        created_again = await worker.cron_tick(now=now)
        assert created_again == []
        assert len(await run_store.list(status=RunStatus.QUEUED)) == 1

    asyncio.run(go())


# --- a materialized cron run actually drains through the worker -------
def test_cron_run_then_drains():
    cron_store = InMemoryCronStore()
    run_store = InMemoryRunStore()
    worker = RunWorker(_agent(), run_store, cron_store=cron_store)

    async def go() -> None:
        now = time.time()
        await cron_store.create(
            CronRecord(
                cron_id="c1",
                schedule="every 1 minute",
                prompt="do it",
                agent="svc",
                enabled=True,
                next_run_at=now - 1,
                created_at=now,
            )
        )
        await worker.cron_tick(now=now)
        # Drain the single materialized run.
        record = await run_store.claim_next(pod_id=worker.pod_id, lease_seconds=30.0)
        assert record is not None
        await worker._execute(record)
        final = await run_store.get(record.run_id)
        assert final is not None
        assert final.status is RunStatus.COMPLETED
        assert final.output == "cron-output"

    asyncio.run(go())


# --- cron rows persist across two SQLite views (two pods) -------------
def test_sqlite_cron_visible_across_views(tmp_path):
    path = str(tmp_path / "shared-crons.db")
    pod_a = SQLiteCronStore(path)
    pod_b = SQLiteCronStore(path)

    async def go() -> None:
        now = time.time()
        await pod_a.create(
            CronRecord(
                cron_id="c1",
                schedule="every 1 hour",
                prompt="p",
                agent="svc",
                enabled=True,
                next_run_at=now,
                created_at=now,
            )
        )
        seen = await pod_b.get("c1")
        assert seen is not None and seen.schedule == "every 1 hour"
        # A roll-forward on B is visible to A.
        await pod_b.mark_run("c1", now=now)
        from_a = await pod_a.get("c1")
        assert from_a is not None and from_a.last_run_at == now

    asyncio.run(go())
