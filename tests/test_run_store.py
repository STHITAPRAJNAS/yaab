"""Durable run store — the cross-process record of every run's lifecycle.

These tests prove the store survives a restart and behaves identically across
replicas: a run created on one store view is visible to a second view over the
same file, status patches are atomic, listings come back newest-first, the
worker-claim primitive hands a queued row to exactly one claimer under
contention, and expired leases get re-queued for another replica to pick up.

All offline: in-memory dicts and SQLite tempfiles only.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab.extensions import get
from yaab.runs import InMemoryRunStore, RunRecord, RunStatus, SQLiteRunStore
from yaab.runs.base import RunStore


def _record(run_id: str, *, status: RunStatus = RunStatus.QUEUED, agent: str = "svc") -> RunRecord:
    now = time.time()
    return RunRecord(
        run_id=run_id,
        agent=agent,
        status=status,
        prompt="hi",
        created_at=now,
        updated_at=now,
    )


def _memory() -> InMemoryRunStore:
    return InMemoryRunStore()


def _sqlite(tmp_path) -> SQLiteRunStore:
    return SQLiteRunStore(str(tmp_path / "runs.db"))


# --- protocol conformance ---------------------------------------------
def test_backends_satisfy_protocol(tmp_path):
    assert isinstance(_memory(), RunStore)
    assert isinstance(_sqlite(tmp_path), RunStore)


# --- roundtrip on both backends ---------------------------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_create_get_roundtrip(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        rec = _record("r1")
        await store.create(rec)
        got = await store.get("r1")
        assert got is not None
        assert got.run_id == "r1"
        assert got.agent == "svc"
        assert got.status is RunStatus.QUEUED
        assert got.prompt == "hi"

    asyncio.run(go())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_get_unknown_returns_none(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)
    assert asyncio.run(store.get("missing")) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_update_patches_fields_atomically(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        await store.create(_record("r1"))
        updated = await store.update(
            "r1",
            status=RunStatus.COMPLETED,
            output={"answer": 42},
            usage={"requests": 3},
        )
        assert updated is not None
        assert updated.status is RunStatus.COMPLETED
        assert updated.output == {"answer": 42}
        assert updated.usage == {"requests": 3}
        # The patch persisted.
        again = await store.get("r1")
        assert again is not None and again.status is RunStatus.COMPLETED
        # Unpatched fields untouched.
        assert again.prompt == "hi"

    asyncio.run(go())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_update_bumps_updated_at(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        rec = _record("r1")
        rec.updated_at = 1.0
        await store.create(rec)
        updated = await store.update("r1", status=RunStatus.RUNNING)
        assert updated is not None
        assert updated.updated_at > 1.0

    asyncio.run(go())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_update_unknown_returns_none(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)
    assert asyncio.run(store.update("nope", status=RunStatus.RUNNING)) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_list_newest_first_and_status_filter(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        a = _record("a")
        a.created_at = 1.0
        b = _record("b")
        b.created_at = 2.0
        c = _record("c", status=RunStatus.RUNNING)
        c.created_at = 3.0
        await store.create(a)
        await store.create(b)
        await store.create(c)

        listed = await store.list()
        ids = [r.run_id for r in listed]
        # Newest first.
        assert ids == ["c", "b", "a"]

        # Status filter.
        running = await store.list(status=RunStatus.RUNNING)
        assert [r.run_id for r in running] == ["c"]

        # Limit.
        limited = await store.list(limit=2)
        assert len(limited) == 2
        assert limited[0].run_id == "c"

    asyncio.run(go())


# --- cross-replica visibility: two store views over one SQLite file ---
def test_sqlite_visible_across_two_views(tmp_path):
    path = str(tmp_path / "shared.db")
    pod_a = SQLiteRunStore(path)
    pod_b = SQLiteRunStore(path)

    async def go() -> None:
        await pod_a.create(_record("r1"))
        # The "other pod" sees it.
        seen = await pod_b.get("r1")
        assert seen is not None and seen.run_id == "r1"
        # A patch on B is visible to A.
        await pod_b.update("r1", status=RunStatus.RUNNING)
        from_a = await pod_a.get("r1")
        assert from_a is not None and from_a.status is RunStatus.RUNNING

    asyncio.run(go())


# --- request_cancel sets the durable flag -----------------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_request_cancel_sets_flag(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        await store.create(_record("r1"))
        found = await store.request_cancel("r1")
        assert found is True
        rec = await store.get("r1")
        assert rec is not None and rec.cancel_requested is True

        missing = await store.request_cancel("nope")
        assert missing is False

    asyncio.run(go())


# --- claim_next: queued -> running, exclusive --------------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_claim_next_marks_running_with_lease(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        await store.create(_record("r1"))
        claimed = await store.claim_next(pod_id="pod-A", lease_seconds=30.0)
        assert claimed is not None
        assert claimed.run_id == "r1"
        assert claimed.status is RunStatus.RUNNING
        assert claimed.owner_pod == "pod-A"
        assert claimed.lease_expires_at is not None and claimed.lease_expires_at > time.time()
        # No more queued rows.
        assert await store.claim_next(pod_id="pod-B", lease_seconds=30.0) is None

    asyncio.run(go())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_claim_next_empty_returns_none(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)
    assert asyncio.run(store.claim_next(pod_id="p", lease_seconds=30.0)) is None


def test_sqlite_claim_exclusive_under_concurrency(tmp_path):
    """Many concurrent claimers over one SQLite file never claim the same row.

    Simulates a fleet of workers racing for a small queue: each queued row must
    be handed to exactly one claimer.
    """
    path = str(tmp_path / "race.db")
    seeder = SQLiteRunStore(path)
    n = 20

    async def go() -> list[str]:
        for i in range(n):
            await seeder.create(_record(f"r{i}"))

        # Each "pod" is its own connection/store instance.
        stores = [SQLiteRunStore(path) for _ in range(8)]

        async def claim_all(store: SQLiteRunStore, pod: str) -> list[str]:
            got: list[str] = []
            while True:
                rec = await store.claim_next(pod_id=pod, lease_seconds=30.0)
                if rec is None:
                    break
                got.append(rec.run_id)
                await asyncio.sleep(0)  # yield to interleave claimers
            return got

        results = await asyncio.gather(*(claim_all(s, f"pod-{i}") for i, s in enumerate(stores)))
        return [rid for sub in results for rid in sub]

    claimed = asyncio.run(go())
    # Every row claimed exactly once, none lost, none duplicated.
    assert sorted(claimed) == sorted(f"r{i}" for i in range(n))
    assert len(claimed) == len(set(claimed)) == n


# --- heartbeat extends the lease --------------------------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_heartbeat_extends_lease(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        await store.create(_record("r1"))
        await store.claim_next(pod_id="pod-A", lease_seconds=5.0)
        before = (await store.get("r1")).lease_expires_at
        await store.heartbeat("r1", pod_id="pod-A", lease_seconds=100.0)
        after = (await store.get("r1")).lease_expires_at
        assert after is not None and before is not None and after > before

    asyncio.run(go())


# --- reap_expired_leases re-queues abandoned runs ---------------------
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_reap_expired_leases_requeues(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        await store.create(_record("r1"))
        await store.claim_next(pod_id="dead-pod", lease_seconds=30.0)
        # Force the lease into the past (sleep-free).
        await store.update("r1", lease_expires_at=time.time() - 1.0)

        reaped = await store.reap_expired_leases()
        assert "r1" in reaped

        rec = await store.get("r1")
        assert rec is not None
        assert rec.status is RunStatus.QUEUED
        assert rec.owner_pod is None
        assert rec.lease_expires_at is None
        # And it can be claimed again by another pod.
        again = await store.claim_next(pod_id="fresh-pod", lease_seconds=30.0)
        assert again is not None and again.run_id == "r1"

    asyncio.run(go())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_reap_ignores_live_leases(backend, tmp_path):
    store = _memory() if backend == "memory" else _sqlite(tmp_path)

    async def go() -> None:
        await store.create(_record("r1"))
        await store.claim_next(pod_id="pod-A", lease_seconds=300.0)
        reaped = await store.reap_expired_leases()
        assert reaped == []
        rec = await store.get("r1")
        assert rec is not None and rec.status is RunStatus.RUNNING

    asyncio.run(go())


# --- registry lookup ---------------------------------------------------
def test_registry_get_sqlite(tmp_path):
    store = get("run", "sqlite", path=str(tmp_path / "reg.db"))
    assert isinstance(store, SQLiteRunStore)


def test_registry_get_memory():
    store = get("run", "memory")
    assert isinstance(store, InMemoryRunStore)


def test_registry_lists_all_run_backends():
    from yaab.extensions import available

    names = available("run")
    assert {"memory", "sqlite", "postgres", "redis"} <= set(names)
