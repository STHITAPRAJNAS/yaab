"""Durable per-run trace store — the record a debugger replays a run from.

These tests prove the trace store keeps every run's events durably and in order:
appended events come back ordered by sequence, a trace written by one store view
is visible to a second view over the same file (so a debugger on another replica
sees the same run), events carrying datetimes and enums survive a JSON round
trip unscathed, runs can be listed and deleted, and the backends are selectable
by name through the component registry.

All offline: in-memory dicts, SQLite tempfiles, and an injected fake Redis
client — no network.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import Enum

import pytest

from yaab.extensions import available, get
from yaab.runs.trace import (
    InMemoryTraceStore,
    RedisTraceStore,
    SQLiteTraceStore,
    TraceStore,
)


class _Color(str, Enum):
    RED = "red"
    BLUE = "blue"


def _event(seq: int, **extra) -> dict:
    payload = {"type": "model_response", "seq": seq, "agent": "svc"}
    payload.update(extra)
    return payload


def _memory() -> InMemoryTraceStore:
    return InMemoryTraceStore()


def _sqlite(tmp_path) -> SQLiteTraceStore:
    return SQLiteTraceStore(str(tmp_path / "trace.db"))


# --- a fake redis good enough for the trace store's key operations -------
class _FakeRedis:
    """A minimal in-process stand-in for the redis client this store uses.

    Implements only the string/sorted-set/set operations the trace store calls,
    so tests run offline without a real server.
    """

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._sets: dict[str, set[str]] = {}

    # strings
    def set(self, key, value, **_kw):  # noqa: ANN001
        self._kv[str(key)] = value

    def get(self, key):  # noqa: ANN001
        return self._kv.get(str(key))

    # sorted sets
    def zadd(self, key, mapping):  # noqa: ANN001
        self._zsets.setdefault(str(key), {}).update({str(m): float(s) for m, s in mapping.items()})

    def zrange(self, key, start, end):  # noqa: ANN001
        members = sorted(self._zsets.get(str(key), {}).items(), key=lambda kv: kv[1])
        ordered = [m for m, _ in members]
        if end == -1:
            end = len(ordered)
        else:
            end += 1
        return ordered[start:end]

    def zrevrange(self, key, start, end):  # noqa: ANN001
        members = sorted(self._zsets.get(str(key), {}).items(), key=lambda kv: kv[1], reverse=True)
        ordered = [m for m, _ in members]
        if end == -1:
            end = len(ordered)
        else:
            end += 1
        return ordered[start:end]

    def zrem(self, key, *members):  # noqa: ANN001
        z = self._zsets.get(str(key))
        if z is None:
            return
        for m in members:
            z.pop(str(m), None)

    # sets
    def sadd(self, key, *members):  # noqa: ANN001
        self._sets.setdefault(str(key), set()).update(str(m) for m in members)

    def srem(self, key, *members):  # noqa: ANN001
        s = self._sets.get(str(key))
        if s is None:
            return
        for m in members:
            s.discard(str(m))

    def delete(self, *keys):  # noqa: ANN001
        for key in keys:
            self._kv.pop(str(key), None)
            self._zsets.pop(str(key), None)
            self._sets.pop(str(key), None)

    def scan_iter(self, match=None):  # noqa: ANN001
        # Only the prefixes the trace store scans are exercised; a plain glob on
        # the recorded keys is enough for the fake.
        import fnmatch

        keys = set(self._kv) | set(self._zsets) | set(self._sets)
        for key in list(keys):
            if match is None or fnmatch.fnmatch(key, match):
                yield key


def _redis() -> RedisTraceStore:
    return RedisTraceStore(client=_FakeRedis())


_BACKENDS = ["memory", "sqlite", "redis"]


def _make(backend, tmp_path):
    if backend == "memory":
        return _memory()
    if backend == "sqlite":
        return _sqlite(tmp_path)
    return _redis()


# --- protocol conformance ---------------------------------------------
def test_backends_satisfy_protocol(tmp_path):
    assert isinstance(_memory(), TraceStore)
    assert isinstance(_sqlite(tmp_path), TraceStore)
    assert isinstance(_redis(), TraceStore)


# --- append / get ordering --------------------------------------------
@pytest.mark.parametrize("backend", _BACKENDS)
def test_append_get_ordered_by_seq(backend, tmp_path):
    store = _make(backend, tmp_path)

    async def go() -> None:
        # Append out of order on purpose; get must return ordered by seq.
        await store.append("r1", 2, _event(2))
        await store.append("r1", 0, _event(0))
        await store.append("r1", 1, _event(1))

        events = await store.get("r1")
        assert [e["seq"] for e in events] == [0, 1, 2]
        assert all(e["type"] == "model_response" for e in events)

    asyncio.run(go())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_get_unknown_run_returns_empty(backend, tmp_path):
    store = _make(backend, tmp_path)
    assert asyncio.run(store.get("missing")) == []


@pytest.mark.parametrize("backend", _BACKENDS)
def test_append_preserves_full_payload(backend, tmp_path):
    store = _make(backend, tmp_path)

    async def go() -> None:
        rich = _event(
            0,
            payload={"model": "test", "finish_reason": "stop"},
            usage={"input_tokens": 12, "output_tokens": 3, "cost_usd": 0.001},
            duration_ms=42.5,
        )
        await store.append("r1", 0, rich)
        got = await store.get("r1")
        assert len(got) == 1
        assert got[0]["usage"]["input_tokens"] == 12
        assert got[0]["duration_ms"] == 42.5
        assert got[0]["payload"]["finish_reason"] == "stop"

    asyncio.run(go())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_append_same_seq_overwrites(backend, tmp_path):
    """Re-appending the same (run_id, seq) replaces the event (idempotent retry)."""
    store = _make(backend, tmp_path)

    async def go() -> None:
        await store.append("r1", 0, _event(0, note="first"))
        await store.append("r1", 0, _event(0, note="second"))
        events = await store.get("r1")
        assert len(events) == 1
        assert events[0]["note"] == "second"

    asyncio.run(go())


# --- JSON-safety: datetimes and enums survive the round trip ----------
@pytest.mark.parametrize("backend", _BACKENDS)
def test_json_safe_datetimes_and_enums(backend, tmp_path):
    store = _make(backend, tmp_path)

    async def go() -> None:
        when = datetime(2026, 6, 1, 12, 30, 0, tzinfo=UTC)
        event = {
            "type": _Color.RED,  # an Enum
            "started_at": when,  # a datetime
            "nested": {"color": _Color.BLUE, "at": when},
            "tags": [_Color.RED, when],
        }
        await store.append("r1", 0, event)
        got = await store.get("r1")
        assert len(got) == 1
        e = got[0]
        # Enums are coerced to their value; datetimes to ISO strings; both are
        # plain JSON types now (no live objects survive serialization).
        assert e["type"] == "red"
        assert e["started_at"] == when.isoformat()
        assert e["nested"]["color"] == "blue"
        assert e["nested"]["at"] == when.isoformat()
        assert e["tags"][0] == "red"
        assert e["tags"][1] == when.isoformat()

    asyncio.run(go())


# --- list_runs --------------------------------------------------------
@pytest.mark.parametrize("backend", _BACKENDS)
def test_list_runs_newest_first(backend, tmp_path):
    store = _make(backend, tmp_path)

    async def go() -> None:
        # Three runs; the most recently first-seen run sorts first.
        await store.append("a", 0, _event(0))
        await store.append("b", 0, _event(0))
        await store.append("c", 0, _event(0))

        runs = await store.list_runs(limit=10)
        assert set(runs) == {"a", "b", "c"}
        # Newest first (c was appended last).
        assert runs[0] == "c"

        limited = await store.list_runs(limit=2)
        assert len(limited) == 2
        assert limited[0] == "c"

    asyncio.run(go())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_list_runs_empty(backend, tmp_path):
    store = _make(backend, tmp_path)
    assert asyncio.run(store.list_runs(limit=10)) == []


# --- delete -----------------------------------------------------------
@pytest.mark.parametrize("backend", _BACKENDS)
def test_delete_removes_trace(backend, tmp_path):
    store = _make(backend, tmp_path)

    async def go() -> None:
        await store.append("r1", 0, _event(0))
        await store.append("r1", 1, _event(1))
        await store.append("r2", 0, _event(0))

        await store.delete("r1")
        assert await store.get("r1") == []
        # The untouched run survives, and it is no longer listed.
        assert await store.get("r2") != []
        assert "r1" not in await store.list_runs(limit=10)
        assert "r2" in await store.list_runs(limit=10)

    asyncio.run(go())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_delete_unknown_is_noop(backend, tmp_path):
    store = _make(backend, tmp_path)
    # Deleting a run that never existed must not raise.
    asyncio.run(store.delete("nope"))


# --- cross-instance visibility over one SQLite file -------------------
def test_sqlite_visible_across_two_views(tmp_path):
    """A trace written by one store view is visible to a second view.

    Two store instances over one file model two replicas: a run traced on one is
    fully readable on the other, so a debugger anywhere sees the same history.
    """
    path = str(tmp_path / "shared.db")
    pod_a = SQLiteTraceStore(path)
    pod_b = SQLiteTraceStore(path)

    async def go() -> None:
        await pod_a.append("r1", 0, _event(0, note="hello"))
        await pod_a.append("r1", 1, _event(1, note="world"))

        # The "other pod" sees the full ordered trace.
        seen = await pod_b.get("r1")
        assert [e["seq"] for e in seen] == [0, 1]
        assert seen[0]["note"] == "hello"
        assert "r1" in await pod_b.list_runs(limit=10)

        # A delete on B is visible to A.
        await pod_b.delete("r1")
        assert await pod_a.get("r1") == []

    asyncio.run(go())


# --- registry lookup --------------------------------------------------
def test_registry_get_sqlite(tmp_path):
    store = get("trace", "sqlite", path=str(tmp_path / "reg.db"))
    assert isinstance(store, SQLiteTraceStore)


def test_registry_get_memory():
    store = get("trace", "memory")
    assert isinstance(store, InMemoryTraceStore)


def test_registry_get_redis():
    store = get("trace", "redis", client=_FakeRedis())
    assert isinstance(store, RedisTraceStore)


def test_registry_lists_all_trace_backends():
    names = available("trace")
    assert {"memory", "sqlite", "postgres", "redis"} <= set(names)
