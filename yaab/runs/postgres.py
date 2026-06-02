"""Postgres run store — the true multi-replica backend.

Runs live in a single Postgres table that every replica shares, so any replica
can poll, cancel, claim, or resume any run. The claim primitive uses
``SELECT ... FOR UPDATE SKIP LOCKED`` — the standard durable-queue claim — so a
fleet of workers drains the queue without ever handing the same row to two
workers and without blocking on each other. The lease reaper re-queues rows
abandoned by a crashed replica.

Uses ``psycopg`` (v3), imported lazily so it is only required when this backend
is actually constructed. The full record is stored as a JSONB ``data`` column
with mirrored, indexed ``status`` / ``lease_expires_at`` columns.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import RunRecord, RunStatus

# Alias for ``list[str]`` used after ``def list`` shadows the builtin.
_RunIds = list


def _require_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "psycopg is required for the Postgres backends. "
            "Install with `pip install 'yaab-sdk[postgres]'`."
        ) from exc
    return psycopg


class PostgresRunStore:
    """Persist run records in a Postgres ``jsonb`` table keyed by run id."""

    def __init__(self, dsn: str, *, table: str = "yaab_runs") -> None:
        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"run_id TEXT PRIMARY KEY, "
            f"status TEXT NOT NULL, "
            f"created_at DOUBLE PRECISION NOT NULL, "
            f"lease_expires_at DOUBLE PRECISION, "
            f"data JSONB NOT NULL)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_status_lease "
            f"ON {table} (status, lease_expires_at)"
        )

    # --- (de)serialization ------------------------------------------------
    @staticmethod
    def _to_record(data: Any) -> RunRecord:
        return RunRecord.model_validate(data)

    def _write(self, conn: Any, record: RunRecord) -> None:
        conn.execute(
            f"INSERT INTO {self._table} "
            f"(run_id, status, created_at, lease_expires_at, data) "
            f"VALUES (%s, %s, %s, %s, %s) "
            f"ON CONFLICT (run_id) DO UPDATE SET "
            f"status = EXCLUDED.status, "
            f"created_at = EXCLUDED.created_at, "
            f"lease_expires_at = EXCLUDED.lease_expires_at, "
            f"data = EXCLUDED.data",
            (
                record.run_id,
                record.status.value,
                record.created_at,
                record.lease_expires_at,
                json.dumps(record.model_dump()),
            ),
        )

    def _read(self, conn: Any, run_id: str) -> RunRecord | None:
        row = conn.execute(
            f"SELECT data FROM {self._table} WHERE run_id = %s", (run_id,)
        ).fetchone()
        return self._to_record(row[0]) if row is not None else None

    # --- lifecycle --------------------------------------------------------
    async def create(self, record: RunRecord) -> None:
        self._write(self._conn, record)

    async def get(self, run_id: str) -> RunRecord | None:
        return self._read(self._conn, run_id)

    async def update(
        self, run_id: str, *, expect_status: RunStatus | None = None, **fields: Any
    ) -> RunRecord | None:
        with self._conn.transaction():
            row = self._conn.execute(
                f"SELECT data FROM {self._table} WHERE run_id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            rec = self._to_record(row[0])
            if expect_status is not None and rec.status is not expect_status:
                return None
            fields.setdefault("updated_at", time.time())
            updated = rec.model_copy(update=fields)
            self._write(self._conn, updated)
            return updated

    async def list(self, *, limit: int = 100, status: RunStatus | None = None) -> list[RunRecord]:
        if status is not None:
            rows = self._conn.execute(
                f"SELECT data FROM {self._table} WHERE status = %s "
                f"ORDER BY created_at DESC LIMIT %s",
                (status.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT data FROM {self._table} ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [self._to_record(r[0]) for r in rows]

    async def request_cancel(self, run_id: str) -> bool:
        updated = await self.update(run_id, cancel_requested=True)
        return updated is not None

    # --- worker queue primitives -----------------------------------------
    async def claim_next(self, *, pod_id: str, lease_seconds: float) -> RunRecord | None:
        now = time.time()
        with self._conn.transaction():
            row = self._conn.execute(
                f"SELECT data FROM {self._table} WHERE status = %s "
                f"ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED",
                (RunStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            rec = self._to_record(row[0])
            claimed = rec.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "owner_pod": pod_id,
                    "lease_expires_at": now + lease_seconds,
                    "started_at": rec.started_at or now,
                    "updated_at": now,
                    "lease_generation": rec.lease_generation + 1,
                }
            )
            self._write(self._conn, claimed)
            return claimed

    async def heartbeat(self, run_id: str, *, pod_id: str, lease_seconds: float) -> None:
        await self.update(
            run_id,
            owner_pod=pod_id,
            lease_expires_at=time.time() + lease_seconds,
        )

    async def reap_expired_leases(self) -> _RunIds[str]:
        now = time.time()
        with self._conn.transaction():
            rows = self._conn.execute(
                f"SELECT data FROM {self._table} WHERE status = %s "
                f"AND lease_expires_at IS NOT NULL AND lease_expires_at < %s "
                f"FOR UPDATE SKIP LOCKED",
                (RunStatus.RUNNING.value, now),
            ).fetchall()
            reaped: list[str] = []
            for row in rows:
                rec = self._to_record(row[0])
                self._write(
                    self._conn,
                    rec.model_copy(
                        update={
                            "status": RunStatus.QUEUED,
                            "owner_pod": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                            # Fence: a reaped run gets a new generation so the
                            # stale worker can no longer finalize over it.
                            "lease_generation": rec.lease_generation + 1,
                        }
                    ),
                )
                reaped.append(rec.run_id)
            return reaped


__all__ = ["PostgresRunStore"]
