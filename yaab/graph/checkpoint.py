"""Checkpointer protocol and backends for durable graph execution.

A checkpointer persists graph state at every superstep so a run can survive a
crash and resume, and so human-in-the-loop interrupts can be parked and picked
up later. Serialization goes through the Rust-accelerated framed encoder in
:mod:`yaab._core` (with a pure-Python fallback).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol, runtime_checkable

from .. import _core


@runtime_checkable
class Checkpointer(Protocol):
    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None: ...

    def get(self, thread_id: str) -> tuple[int, dict[str, Any]] | None: ...

    def history(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]: ...


class MemorySaver:
    """In-memory checkpointer; keeps full history for time-travel debugging."""

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[int, bytes]]] = {}

    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None:
        blob = _core.encode_checkpoint(state)
        self._store.setdefault(thread_id, []).append((step, blob))

    def get(self, thread_id: str) -> tuple[int, dict[str, Any]] | None:
        history = self._store.get(thread_id)
        if not history:
            return None
        step, blob = history[-1]
        return step, _core.decode_checkpoint(blob)

    def history(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]:
        return [(s, _core.decode_checkpoint(b)) for s, b in self._store.get(thread_id, [])]


class SQLiteSaver:
    """Durable checkpointer backed by SQLite."""

    def __init__(self, path: str = "yaab_checkpoints.db") -> None:
        # check_same_thread=False: a served background run checkpoints from the
        # worker thread while a resume reads from a request handler; sqlite3
        # serialized mode (threadsafety 3) makes the shared connection safe.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            "thread_id TEXT, step INTEGER, blob BLOB, PRIMARY KEY (thread_id, step))"
        )
        self._conn.commit()

    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None:
        blob = _core.encode_checkpoint(state)
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)", (thread_id, step, blob)
        )
        self._conn.commit()

    def get(self, thread_id: str) -> tuple[int, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT step, blob FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0], _core.decode_checkpoint(row[1])

    def history(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT step, blob FROM checkpoints WHERE thread_id = ? ORDER BY step",
            (thread_id,),
        ).fetchall()
        return [(s, _core.decode_checkpoint(b)) for s, b in rows]


class PostgresSaver:
    """Durable checkpointer backed by Postgres / Aurora PostgreSQL.

    Uses ``psycopg`` (``pip install 'yaab-sdk[postgres]'``), imported lazily. Stores
    the framed checkpoint blob per ``(thread_id, step)`` — point the DSN at an
    Aurora endpoint for a managed, HA checkpoint store.
    """

    def __init__(self, dsn: str, *, table: str = "yaab_checkpoints") -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "psycopg is required for PostgresSaver. `pip install 'yaab-sdk[postgres]'`."
            ) from exc
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"thread_id TEXT, step INTEGER, blob BYTEA, PRIMARY KEY (thread_id, step))"
        )

    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None:
        blob = _core.encode_checkpoint(state)
        self._conn.execute(
            f"INSERT INTO {self._table} (thread_id, step, blob) VALUES (%s, %s, %s) "
            f"ON CONFLICT (thread_id, step) DO UPDATE SET blob = EXCLUDED.blob",
            (thread_id, step, blob),
        )

    def get(self, thread_id: str) -> tuple[int, dict[str, Any]] | None:
        row = self._conn.execute(
            f"SELECT step, blob FROM {self._table} WHERE thread_id = %s ORDER BY step DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0], _core.decode_checkpoint(bytes(row[1]))

    def history(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]:
        rows = self._conn.execute(
            f"SELECT step, blob FROM {self._table} WHERE thread_id = %s ORDER BY step",
            (thread_id,),
        ).fetchall()
        return [(s, _core.decode_checkpoint(bytes(b))) for s, b in rows]


class RedisSaver:
    """Durable checkpointer backed by Redis / ElastiCache / MemoryDB.

    Uses ``redis`` (``pip install 'yaab-sdk[redis]'``), imported lazily. Each step is
    a base64 blob in a per-thread Redis hash, so the full history (and
    time-travel) survives across processes.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "yaab:checkpoint",
        ttl_seconds: int | None = None,
        client: Any = None,
    ) -> None:
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "redis is required for RedisSaver. `pip install 'yaab-sdk[redis]'`."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, thread_id: str) -> str:
        return f"{self._prefix}:{thread_id}"

    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None:
        import base64

        blob = base64.b64encode(_core.encode_checkpoint(state)).decode("ascii")
        key = self._key(thread_id)
        self._redis.hset(key, str(step), blob)
        if self._ttl is not None:
            self._redis.expire(key, self._ttl)

    def get(self, thread_id: str) -> tuple[int, dict[str, Any]] | None:
        entries = self._redis.hgetall(self._key(thread_id))
        if not entries:
            return None
        step = max(int(s) for s in entries)
        return step, self._decode(entries[str(step)])

    def history(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]:
        entries = self._redis.hgetall(self._key(thread_id))
        return [
            (int(s), self._decode(b)) for s, b in sorted(entries.items(), key=lambda kv: int(kv[0]))
        ]

    @staticmethod
    def _decode(blob: str) -> dict[str, Any]:
        import base64

        return _core.decode_checkpoint(base64.b64decode(blob))


__all__ = ["Checkpointer", "MemorySaver", "SQLiteSaver", "PostgresSaver", "RedisSaver"]
