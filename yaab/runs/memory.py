"""In-memory run store (default for dev and tests).

Keeps run records in a process-local dict. Not durable across restarts and not
shared across replicas — single-process only — but preserves today's behavior
bit-for-bit and serves as the reference implementation for the protocol.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .base import TERMINAL_STATUSES, RunRecord, RunStatus

# Alias for ``list[str]`` used after ``def list`` shadows the builtin.
_RunIds = list


class InMemoryRunStore:
    """Hold run records in a process-local dict."""

    def __init__(self) -> None:
        self._store: dict[str, RunRecord] = {}
        # Guards the read-filter-modify-write of the queue/lease state so that
        # ``claim_next`` (and the other mutating primitives) are atomic against
        # concurrent claimers within a single event loop. SQLite uses
        # ``BEGIN IMMEDIATE`` and Postgres uses ``FOR UPDATE`` for the same
        # contract; this lock is the in-memory equivalent.
        self._lock = asyncio.Lock()

    async def create(self, record: RunRecord) -> None:
        # Store a copy so external mutation of the caller's object can't leak in.
        self._store[record.run_id] = record.model_copy(deep=True)

    async def get(self, run_id: str) -> RunRecord | None:
        rec = self._store.get(run_id)
        return rec.model_copy(deep=True) if rec is not None else None

    async def update(
        self, run_id: str, *, expect_status: RunStatus | None = None, **fields: Any
    ) -> RunRecord | None:
        async with self._lock:
            rec = self._store.get(run_id)
            if rec is None:
                return None
            if expect_status is not None and rec.status is not expect_status:
                return None
            fields.setdefault("updated_at", time.time())
            updated = rec.model_copy(update=fields)
            self._store[run_id] = updated
            return updated.model_copy(deep=True)

    async def list(self, *, limit: int = 100, status: RunStatus | None = None) -> list[RunRecord]:
        records = list(self._store.values())
        if status is not None:
            records = [r for r in records if r.status is status]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return [r.model_copy(deep=True) for r in records[:limit]]

    async def request_cancel(self, run_id: str) -> bool:
        async with self._lock:
            rec = self._store.get(run_id)
            if rec is None:
                return False
            self._store[run_id] = rec.model_copy(
                update={"cancel_requested": True, "updated_at": time.time()}
            )
            return True

    async def claim_next(self, *, pod_id: str, lease_seconds: float) -> RunRecord | None:
        # Hold the lock across the whole read-filter-modify-write so two
        # concurrent claimers can never select and claim the same QUEUED row.
        async with self._lock:
            now = time.time()
            queued = [r for r in self._store.values() if r.status is RunStatus.QUEUED]
            if not queued:
                return None
            queued.sort(key=lambda r: r.created_at)
            target = queued[0]
            claimed = target.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "owner_pod": pod_id,
                    "lease_expires_at": now + lease_seconds,
                    "started_at": target.started_at or now,
                    "updated_at": now,
                    "lease_generation": target.lease_generation + 1,
                }
            )
            self._store[target.run_id] = claimed
            return claimed.model_copy(deep=True)

    async def heartbeat(self, run_id: str, *, pod_id: str, lease_seconds: float) -> None:
        async with self._lock:
            rec = self._store.get(run_id)
            if rec is None:
                return
            self._store[run_id] = rec.model_copy(
                update={
                    "owner_pod": pod_id,
                    "lease_expires_at": time.time() + lease_seconds,
                    "updated_at": time.time(),
                }
            )

    async def reap_expired_leases(self) -> _RunIds[str]:
        async with self._lock:
            now = time.time()
            reaped: list[str] = []
            for run_id, rec in list(self._store.items()):
                if (
                    rec.status is RunStatus.RUNNING
                    and rec.status not in TERMINAL_STATUSES
                    and rec.lease_expires_at is not None
                    and rec.lease_expires_at < now
                ):
                    self._store[run_id] = rec.model_copy(
                        update={
                            "status": RunStatus.QUEUED,
                            "owner_pod": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                            # Bump the fence so a stale worker whose lease we just
                            # reaped cannot finalize over a newer claimant.
                            "lease_generation": rec.lease_generation + 1,
                        }
                    )
                    reaped.append(run_id)
            return reaped
