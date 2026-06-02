"""Regression tests for the Redis run store's atomic claim and bounded list.

A fake Redis (single-threaded, in-process) emulates just the commands the store
uses. It proves two audit findings without a real server:

* The claim is atomic across LMOVE + record flip: a claimed run is RUNNING the
  instant it leaves the queue, so a reaper never re-queues a popped-but-still-
  QUEUED record and a run is never handed out twice (finding 2).
* ``list`` pages the sorted index instead of pulling every id with
  ``zrevrange(0, -1)`` (finding 15).
"""

from __future__ import annotations

import asyncio
import json
import time

from yaab.runs.base import RunRecord, RunStatus
from yaab.runs.redis import RedisRunStore


class FakeRedis:
    """Minimal in-process Redis covering the run store's command surface."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.zrange_calls: list[tuple[int, int]] = []

    # strings
    def set(self, key, value, ex=None):
        self.kv[key] = value

    def get(self, key):
        return self.kv.get(key)

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.lists.pop(k, None)
            self.zsets.pop(k, None)

    def exists(self, key):
        return 1 if key in self.kv else 0

    def publish(self, channel, message):
        return 0

    # lists
    def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)

    def lrange(self, key, start, stop):
        items = self.lists.get(key, [])
        if stop == -1:
            return list(items[start:])
        return list(items[start : stop + 1])

    def lrem(self, key, count, value):
        items = self.lists.get(key, [])
        self.lists[key] = [x for x in items if x != value]

    def lmove(self, src, dst, from_side, to_side):
        items = self.lists.get(src, [])
        if not items:
            return None
        val = items.pop(0) if from_side == "LEFT" else items.pop()
        self.lists.setdefault(dst, []).append(val)
        return val

    # sorted sets
    def zadd(self, key, mapping):
        z = self.zsets.setdefault(key, {})
        z.update({k: float(v) for k, v in mapping.items()})

    def zrevrange(self, key, start, stop):
        self.zrange_calls.append((start, stop))
        z = self.zsets.get(key, {})
        ordered = sorted(z, key=lambda m: z[m], reverse=True)
        if stop == -1:
            return ordered[start:]
        return ordered[start : stop + 1]

    def pipeline(self):
        return _FakePipeline(self)

    def eval(self, script, numkeys, *args):
        """Interpret the store's claim Lua: LMOVE then GET the record."""
        keys = args[:numkeys]
        argv = args[numkeys:]
        queue_key, processing_key = keys[0], keys[1]
        rec_prefix = argv[0]
        run_id = self.lmove(queue_key, processing_key, "LEFT", "RIGHT")
        if run_id is None:
            return None
        raw = self.get(rec_prefix + run_id)
        if raw is None:
            self.lrem(processing_key, 0, run_id)
            return None
        return json.dumps({"run_id": run_id, "data": raw})


class _FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list = []

    def set(self, *a, **kw):
        self._ops.append((self._redis.set, a, kw))
        return self

    def zadd(self, *a, **kw):
        self._ops.append((self._redis.zadd, a, kw))
        return self

    def execute(self):
        for fn, a, kw in self._ops:
            fn(*a, **kw)
        self._ops.clear()


def _record(run_id: str, created_at: float) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        agent="svc",
        status=RunStatus.QUEUED,
        prompt="hi",
        background=True,
        resume_id=run_id,
        created_at=created_at,
        updated_at=created_at,
    )


def test_redis_claim_flips_record_to_running_atomically():
    async def go() -> None:
        store = RedisRunStore(client=FakeRedis())
        await store.create(_record("r1", time.time()))
        claimed = await store.claim_next(pod_id="A", lease_seconds=30)
        assert claimed is not None
        # The record is RUNNING the instant it is returned — no QUEUED window.
        assert claimed.status is RunStatus.RUNNING
        assert claimed.owner_pod == "A"
        assert claimed.lease_generation == 1
        persisted = await store.get("r1")
        assert persisted is not None and persisted.status is RunStatus.RUNNING

    asyncio.run(go())


def test_redis_claim_never_hands_out_same_run_twice():
    async def go() -> None:
        store = RedisRunStore(client=FakeRedis())
        for i in range(10):
            await store.create(_record(f"r{i}", time.time() + i))
        seen: list[str] = []
        while True:
            rec = await store.claim_next(pod_id="A", lease_seconds=30)
            if rec is None:
                break
            seen.append(rec.run_id)
        assert sorted(seen) == sorted(f"r{i}" for i in range(10))
        assert len(set(seen)) == 10  # no duplicates

    asyncio.run(go())


def test_redis_reaper_bumps_generation_and_requeues():
    async def go() -> None:
        client = FakeRedis()
        store = RedisRunStore(client=client)
        await store.create(_record("r1", time.time()))
        claimed = await store.claim_next(pod_id="A", lease_seconds=0)
        assert claimed is not None and claimed.lease_generation == 1
        # Force the lease to be gone and the deadline passed, then reap.
        client.kv.pop(store._lease_key("r1"), None)
        await store.update("r1", lease_expires_at=time.time() - 1)
        reaped = await store.reap_expired_leases()
        assert reaped == ["r1"]
        requeued = await store.get("r1")
        assert requeued is not None
        assert requeued.status is RunStatus.QUEUED
        assert requeued.lease_generation == 2  # fenced past the stale worker

    asyncio.run(go())


def test_redis_list_pages_index_instead_of_full_scan():
    async def go() -> None:
        client = FakeRedis()
        store = RedisRunStore(client=client)
        for i in range(500):
            await store.create(_record(f"r{i}", time.time() + i))
        client.zrange_calls.clear()
        recs = await store.list(limit=10)
        assert len(recs) == 10
        # Every zrevrange call was bounded (never the full 0..-1 scan).
        assert client.zrange_calls, "list should query the index"
        assert all(stop != -1 for _, stop in client.zrange_calls)

    asyncio.run(go())
