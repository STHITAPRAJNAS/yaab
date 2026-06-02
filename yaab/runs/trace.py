"""Durable per-run trace store — the record a debugger replays a run from.

Every run emits a stream of typed events (model calls, tool calls, transfers,
approvals) carrying the per-step model name, finish reason, token usage, cost,
and latency. The trace store keeps that stream durably, keyed by ``run_id`` and
ordered by a per-run sequence number, so a run can be inspected step by step
long after it finished and from any replica — the source data behind a debugger
that replays a run with full per-step detail.

The contract is deliberately small and storage-agnostic:

* :meth:`TraceStore.append` records one event at ``(run_id, seq)``; re-appending
  the same position overwrites it, so a retried append is idempotent.
* :meth:`TraceStore.get` returns a run's events ordered by ``seq``.
* :meth:`TraceStore.list_runs` lists recent runs, newest-first.
* :meth:`TraceStore.delete` drops a run's whole trace.

Events are stored JSON-safe: enums collapse to their value and datetimes to ISO
strings, so no live object ever has to round-trip through a backend. Backends
mirror the run/session trio exactly — an in-memory dict for dev, SQLite for a
single durable node, Postgres for true multi-replica history, and Redis for a
distributed store with an injectable client for offline tests. Each is
registered under the ``trace`` component kind, so it can be selected by name:
``yaab.extensions.get("trace", "sqlite", path=...)``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# Alias for ``list[...]`` used after ``def list_runs`` and friends, kept for a
# consistent style with the sibling run-store modules.
_List = list


def _json_safe(value: Any) -> Any:
    """Coerce a value into plain JSON types, recursively.

    Enums collapse to their ``value`` and datetimes (anything with an
    ``isoformat``) to an ISO string, so an event carrying live objects can be
    persisted and read back as ordinary JSON. Pydantic models are dumped via
    ``model_dump``; anything else unknown falls back to ``str``.
    """
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):  # datetime / date / time
        return value.isoformat()
    if hasattr(value, "model_dump"):  # pydantic model
        return _json_safe(value.model_dump())
    return str(value)


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    """Render an event dict fully JSON-safe (top-level always a dict)."""
    return {str(k): _json_safe(v) for k, v in event.items()}


@runtime_checkable
class TraceStore(Protocol):
    """Pluggable, durable backend for per-run event traces.

    Implementations are interchangeable: an in-memory dict for single-process
    dev, SQLite for a single durable node, Postgres for shared multi-replica
    history, Redis for a distributed store. Swapping one in is a one-line change.
    """

    async def append(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        """Record one event at ``(run_id, seq)``.

        Re-appending the same position overwrites the prior event, so a retried
        append is idempotent. The event is stored JSON-safe.
        """
        ...

    async def get(self, run_id: str) -> _List[dict[str, Any]]:
        """Return the run's events ordered by ``seq`` (empty if unknown)."""
        ...

    async def list_runs(self, limit: int = 100) -> _List[str]:
        """Return recent run ids that have a trace, newest-first."""
        ...

    async def delete(self, run_id: str) -> None:
        """Drop a run's whole trace. A no-op if the run is unknown."""
        ...


# ---------------------------------------------------------------------------
# In-memory backend (default for dev and tests).
# ---------------------------------------------------------------------------
class InMemoryTraceStore:
    """Hold run traces in a process-local dict.

    Not durable across restarts and not shared across replicas — single-process
    only — but the reference implementation for the protocol.
    """

    def __init__(self) -> None:
        # run_id -> {seq -> event}
        self._traces: dict[str, dict[int, dict[str, Any]]] = {}
        # run_id -> first-seen monotonic counter, for newest-first listing.
        self._order: dict[str, int] = {}
        self._counter = 0

    async def append(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        run = self._traces.setdefault(run_id, {})
        run[int(seq)] = _safe_event(event)
        if run_id not in self._order:
            self._order[run_id] = self._counter
            self._counter += 1

    async def get(self, run_id: str) -> _List[dict[str, Any]]:
        run = self._traces.get(run_id)
        if not run:
            return []
        return [run[s] for s in sorted(run)]

    async def list_runs(self, limit: int = 100) -> _List[str]:
        ordered = sorted(self._order, key=lambda r: self._order[r], reverse=True)
        return ordered[:limit]

    async def delete(self, run_id: str) -> None:
        self._traces.pop(run_id, None)
        self._order.pop(run_id, None)


# ---------------------------------------------------------------------------
# SQLite backend — durable on a single node, shared across store views.
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_trace_run_seq ON trace_events (run_id, seq)"


class SQLiteTraceStore:
    """Persist run traces in a SQLite ``trace_events`` table keyed by run+seq.

    A trace survives a restart and is visible to every store view over the same
    file, so two processes on one host behave as two replicas sharing one
    history. Each event is stored as a JSON ``payload`` column.
    """

    def __init__(self, path: str = "yaab_trace.db") -> None:
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)

    async def append(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trace_events (run_id, seq, created_at, payload) "
            "VALUES (?, ?, ?, ?)",
            (run_id, int(seq), time.time(), json.dumps(_safe_event(event))),
        )

    async def get(self, run_id: str) -> _List[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT payload FROM trace_events WHERE run_id = ? ORDER BY seq ASC",
            (run_id,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    async def list_runs(self, limit: int = 100) -> _List[str]:
        # Newest-first by the most recent event each run recorded.
        rows = self._conn.execute(
            "SELECT run_id FROM trace_events GROUP BY run_id ORDER BY MAX(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    async def delete(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM trace_events WHERE run_id = ?", (run_id,))


# ---------------------------------------------------------------------------
# Postgres backend — shared multi-replica history.
# ---------------------------------------------------------------------------
def _require_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "psycopg is required for the Postgres backends. "
            "Install with `pip install 'yaab-sdk[postgres]'`."
        ) from exc
    return psycopg


class PostgresTraceStore:
    """Persist run traces in a Postgres table every replica shares.

    A run traced on one replica is fully readable on any other, so a debugger
    anywhere sees the same per-step history. Each event is a JSONB ``payload``
    keyed by ``(run_id, seq)``; uses ``psycopg`` (v3), imported lazily so it is
    only required when this backend is actually constructed.
    """

    def __init__(self, dsn: str, *, table: str = "yaab_trace_events") -> None:
        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"run_id TEXT NOT NULL, "
            f"seq INTEGER NOT NULL, "
            f"created_at DOUBLE PRECISION NOT NULL, "
            f"payload JSONB NOT NULL, "
            f"PRIMARY KEY (run_id, seq))"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_run_seq ON {table} (run_id, seq)"
        )

    async def append(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        self._conn.execute(
            f"INSERT INTO {self._table} (run_id, seq, created_at, payload) "
            f"VALUES (%s, %s, %s, %s) "
            f"ON CONFLICT (run_id, seq) DO UPDATE SET "
            f"created_at = EXCLUDED.created_at, payload = EXCLUDED.payload",
            (run_id, int(seq), time.time(), json.dumps(_safe_event(event))),
        )

    async def get(self, run_id: str) -> _List[dict[str, Any]]:
        rows = self._conn.execute(
            f"SELECT payload FROM {self._table} WHERE run_id = %s ORDER BY seq ASC",
            (run_id,),
        ).fetchall()
        return [r[0] for r in rows]

    async def list_runs(self, limit: int = 100) -> _List[str]:
        rows = self._conn.execute(
            f"SELECT run_id FROM {self._table} "
            f"GROUP BY run_id ORDER BY MAX(created_at) DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    async def delete(self, run_id: str) -> None:
        self._conn.execute(f"DELETE FROM {self._table} WHERE run_id = %s", (run_id,))


# ---------------------------------------------------------------------------
# Redis backend — a distributed, durable trace store.
# ---------------------------------------------------------------------------
class RedisTraceStore:
    """Persist run traces in Redis.

    Each event is a JSON value under a per-(run, seq) key; a per-run sorted set
    orders the run's events by ``seq``, and a global sorted set indexes runs by
    last-write time for newest-first listing. A client can be injected via
    ``client=`` for offline tests against a fake.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "yaab:trace",
        client: Any = None,
    ) -> None:
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            try:
                import redis  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "redis is required for RedisTraceStore. `pip install 'yaab-sdk[redis]'`."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    # --- key helpers ------------------------------------------------------
    def _event_key(self, run_id: str, seq: int) -> str:
        return f"{self._prefix}:ev:{run_id}:{seq}"

    def _seq_index_key(self, run_id: str) -> str:
        return f"{self._prefix}:seq:{run_id}"

    @property
    def _runs_index_key(self) -> str:
        return f"{self._prefix}:runs"

    @staticmethod
    def _decode(raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def append(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        seq = int(seq)
        self._redis.set(self._event_key(run_id, seq), json.dumps(_safe_event(event)))
        # Order this run's events by seq, and index the run by last-write time.
        self._redis.zadd(self._seq_index_key(run_id), {str(seq): float(seq)})
        self._redis.zadd(self._runs_index_key, {run_id: time.time()})

    async def get(self, run_id: str) -> _List[dict[str, Any]]:
        seqs = self._redis.zrange(self._seq_index_key(run_id), 0, -1)
        events: list[dict[str, Any]] = []
        for s in seqs:
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            decoded = self._decode(self._redis.get(self._event_key(run_id, int(s))))
            if decoded is not None:
                events.append(decoded)
        return events

    async def list_runs(self, limit: int = 100) -> _List[str]:
        ids = self._redis.zrevrange(self._runs_index_key, 0, limit - 1)
        return [i.decode("utf-8") if isinstance(i, bytes) else i for i in ids]

    async def delete(self, run_id: str) -> None:
        seqs = self._redis.zrange(self._seq_index_key(run_id), 0, -1)
        for s in seqs:
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            self._redis.delete(self._event_key(run_id, int(s)))
        self._redis.delete(self._seq_index_key(run_id))
        self._redis.zrem(self._runs_index_key, run_id)


# ---------------------------------------------------------------------------
# Component registration — selectable by name under the ``trace`` kind.
# ---------------------------------------------------------------------------
def _register_backends() -> None:
    """Register trace-store backends as ``trace`` components (discoverable by name)."""
    from ..extensions import register

    register("trace", "memory", lambda **kw: InMemoryTraceStore())
    register("trace", "sqlite", lambda **kw: SQLiteTraceStore(**kw))

    def _pg(**kw: Any) -> Any:
        return PostgresTraceStore(**kw)

    def _redis(**kw: Any) -> Any:
        return RedisTraceStore(**kw)

    register("trace", "postgres", _pg)
    register("trace", "aurora", _pg)  # Aurora PostgreSQL via the same driver
    register("trace", "redis", _redis)


_register_backends()


__all__ = [
    "TraceStore",
    "InMemoryTraceStore",
    "SQLiteTraceStore",
    "PostgresTraceStore",
    "RedisTraceStore",
]
