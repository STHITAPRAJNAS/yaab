"""Checkpointer protocol and backends for durable graph execution.

A checkpointer persists graph state at every superstep so a run can survive a
crash and resume, and so human-in-the-loop interrupts can be parked and picked
up later. Serialization goes through the Rust-accelerated framed encoder in
:mod:`yaab._core` (with a pure-Python fallback).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Protocol, runtime_checkable

from .. import _core


@runtime_checkable
class Checkpointer(Protocol):
    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None:
        ...

    def get(self, thread_id: str) -> Optional[tuple[int, dict[str, Any]]]:
        ...

    def history(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]:
        ...


class MemorySaver:
    """In-memory checkpointer; keeps full history for time-travel debugging."""

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[int, bytes]]] = {}

    def put(self, thread_id: str, step: int, state: dict[str, Any]) -> None:
        blob = _core.encode_checkpoint(state)
        self._store.setdefault(thread_id, []).append((step, blob))

    def get(self, thread_id: str) -> Optional[tuple[int, dict[str, Any]]]:
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
        self._conn = sqlite3.connect(path)
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

    def get(self, thread_id: str) -> Optional[tuple[int, dict[str, Any]]]:
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


__all__ = ["Checkpointer", "MemorySaver", "SQLiteSaver"]
