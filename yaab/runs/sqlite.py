"""SQLite run store — durable runs for a single node.

Runs survive a restart and are visible to every store view over the same file,
so two processes on one host behave as two replicas sharing the source of
truth. The claim primitive uses ``BEGIN IMMEDIATE`` to take a write lock before
selecting the next queued row, so concurrent workers never claim the same run.

The full record is stored as a JSON ``data`` column; ``status`` and
``lease_expires_at`` are mirrored into indexed columns so the queue scan and
lease reaper stay cheap.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from .base import TERMINAL_STATUSES, RunRecord, RunStatus

# Alias for ``list[str]`` used after ``def list`` shadows the builtin.
_RunIds = list

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    lease_expires_at REAL,
    data TEXT NOT NULL
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_runs_status_lease ON runs (status, lease_expires_at)"


class SQLiteRunStore:
    """Persist run records in a SQLite ``runs`` table keyed by run id."""

    def __init__(self, path: str = "yaab_runs.db") -> None:
        # ``isolation_level=None`` gives us explicit transaction control so the
        # claim can use BEGIN IMMEDIATE for an atomic read-modify-write.
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)

    # --- (de)serialization ------------------------------------------------
    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> RunRecord:
        return RunRecord.model_validate_json(row[0])

    def _write(self, record: RunRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, status, created_at, lease_expires_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.run_id,
                record.status.value,
                record.created_at,
                record.lease_expires_at,
                record.model_dump_json(),
            ),
        )

    def _read(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_record(row) if row is not None else None

    # --- lifecycle --------------------------------------------------------
    async def create(self, record: RunRecord) -> None:
        self._write(record)

    async def get(self, run_id: str) -> RunRecord | None:
        return self._read(run_id)

    async def update(self, run_id: str, **fields: Any) -> RunRecord | None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rec = self._read(run_id)
            if rec is None:
                self._conn.execute("COMMIT")
                return None
            fields.setdefault("updated_at", time.time())
            updated = rec.model_copy(update=fields)
            self._write(updated)
            self._conn.execute("COMMIT")
            return updated
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    async def list(self, *, limit: int = 100, status: RunStatus | None = None) -> list[RunRecord]:
        if status is not None:
            rows = self._conn.execute(
                "SELECT data FROM runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    async def request_cancel(self, run_id: str) -> bool:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rec = self._read(run_id)
            if rec is None:
                self._conn.execute("COMMIT")
                return False
            self._write(
                rec.model_copy(update={"cancel_requested": True, "updated_at": time.time()})
            )
            self._conn.execute("COMMIT")
            return True
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # --- worker queue primitives -----------------------------------------
    async def claim_next(self, *, pod_id: str, lease_seconds: float) -> RunRecord | None:
        now = time.time()
        # BEGIN IMMEDIATE takes the database write lock up front, so a racing
        # claimer blocks here (busy_timeout) rather than reading the same row.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT data FROM runs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                (RunStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            rec = self._row_to_record(row)
            claimed = rec.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "owner_pod": pod_id,
                    "lease_expires_at": now + lease_seconds,
                    "started_at": rec.started_at or now,
                    "updated_at": now,
                }
            )
            self._write(claimed)
            self._conn.execute("COMMIT")
            return claimed
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    async def heartbeat(self, run_id: str, *, pod_id: str, lease_seconds: float) -> None:
        await self.update(
            run_id,
            owner_pod=pod_id,
            lease_expires_at=time.time() + lease_seconds,
        )

    async def reap_expired_leases(self) -> _RunIds[str]:
        now = time.time()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self._conn.execute(
                "SELECT data FROM runs WHERE status = ? "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
                (RunStatus.RUNNING.value, now),
            ).fetchall()
            reaped: list[str] = []
            for row in rows:
                rec = self._row_to_record(row)
                if rec.status in TERMINAL_STATUSES:
                    continue
                self._write(
                    rec.model_copy(
                        update={
                            "status": RunStatus.QUEUED,
                            "owner_pod": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                        }
                    )
                )
                reaped.append(rec.run_id)
            self._conn.execute("COMMIT")
            return reaped
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
