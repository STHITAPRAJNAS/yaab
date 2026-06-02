"""Scheduled runs — durable cron rows that materialize queued runs when due.

A schedule is just another durable row: ``(cron_id, schedule, prompt, agent,
enabled, next_run_at)``. The worker ticks the store; every schedule whose
next-run time has arrived becomes exactly one queued run and rolls its next-run
time forward, so a tick is idempotent for a given moment, a disabled schedule is
inert, and a missed tick simply fires on the next one.

Schedule parsing is deliberately small so it carries no extra dependency:

* ``"every N seconds"`` / ``"every N minutes"`` / ``"every N hours"`` (singular
  forms accepted),
* ``"@every Ns"`` / ``"@every Nm"`` / ``"@every Nh"`` shorthand,
* a bare number, read as a fixed interval in seconds.

Richer calendar expressions (day-of-week, specific clock times) can be layered
on later without changing this protocol.

Backends mirror the run-store trio: an in-memory dict for dev and tests, and a
SQLite table so schedules survive a restart and are shared across replicas.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# Alias for ``list`` used after ``def list`` shadows the builtin in annotations.
_List = list


class CronRecord(BaseModel):
    """A durable schedule: when to run which agent with which prompt.

    ``next_run_at`` is the wall-clock second at which the schedule next fires;
    :meth:`CronStore.mark_run` rolls it forward by the parsed interval. A
    disabled schedule is retained but never materializes a run.
    """

    cron_id: str
    schedule: str
    prompt: str
    agent: str
    enabled: bool = True
    next_run_at: float
    created_at: float
    last_run_at: float | None = None
    session_id: str | None = None
    identity: str | None = None
    timezone: str | None = None
    webhook: str | None = None


# --- schedule parsing -------------------------------------------------
_EVERY_RE = re.compile(r"^\s*every\s+(\d+(?:\.\d+)?)\s+(second|minute|hour)s?\s*$", re.IGNORECASE)
_AT_EVERY_RE = re.compile(r"^\s*@every\s+(\d+(?:\.\d+)?)\s*([smh])\s*$", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")

_UNIT_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
_SHORT_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_schedule(schedule: str) -> float:
    """Return the interval in seconds for a supported schedule string.

    Supports ``"every N seconds|minutes|hours"``, the ``"@every Ns|Nm|Nh"``
    shorthand, and a bare number (seconds). Raises :class:`ValueError` for any
    other form — including full cron expressions, which are intentionally not
    yet supported so no scheduling dependency is pulled in.
    """
    match = _EVERY_RE.match(schedule)
    if match is not None:
        value, unit = match.group(1), match.group(2).lower()
        return float(value) * _UNIT_SECONDS[unit]

    match = _AT_EVERY_RE.match(schedule)
    if match is not None:
        value, unit = match.group(1), match.group(2).lower()
        return float(value) * _SHORT_UNIT_SECONDS[unit]

    match = _BARE_NUMBER_RE.match(schedule)
    if match is not None:
        return float(match.group(1))

    raise ValueError(
        f"unsupported schedule {schedule!r}: use 'every N seconds|minutes|hours', "
        "'@every Ns|Nm|Nh', or a number of seconds"
    )


@runtime_checkable
class CronStore(Protocol):
    """Pluggable, durable backend for schedules.

    Implementations are interchangeable: an in-memory dict for dev, SQLite for a
    durable shared schedule table. The worker reads :meth:`due` each tick and
    rolls each fired schedule forward with :meth:`mark_run`.
    """

    async def create(self, record: CronRecord) -> None:
        """Persist (or replace) a schedule."""
        ...

    async def get(self, cron_id: str) -> CronRecord | None:
        """Return the schedule, or ``None`` if it does not exist."""
        ...

    async def list(self) -> _List[CronRecord]:
        """Return all schedules."""
        ...

    async def delete(self, cron_id: str) -> bool:
        """Remove a schedule. Returns ``True`` if it existed."""
        ...

    async def due(self, *, now: float | None = None) -> _List[CronRecord]:
        """Return enabled schedules whose ``next_run_at`` has arrived."""
        ...

    async def mark_run(self, cron_id: str, *, now: float | None = None) -> CronRecord | None:
        """Record that a schedule just fired and roll ``next_run_at`` forward.

        ``last_run_at`` is set to ``now`` and ``next_run_at`` advances by the
        schedule's parsed interval. Returns the updated record, or ``None`` if
        the schedule no longer exists.
        """
        ...


def _advance(record: CronRecord, now: float) -> dict[str, Any]:
    """Compute the rolled-forward fields after a schedule fires at ``now``."""
    interval = parse_schedule(record.schedule)
    # Advance from the scheduled time so a slow tick does not drift the cadence;
    # but never schedule into the past, so a long-stalled schedule fires once and
    # then resumes its normal cadence rather than firing repeatedly to catch up.
    next_at = record.next_run_at + interval
    if next_at <= now:
        next_at = now + interval
    return {"last_run_at": now, "next_run_at": next_at}


class InMemoryCronStore:
    """Hold schedules in a process-local dict (default for dev and tests)."""

    def __init__(self) -> None:
        self._store: dict[str, CronRecord] = {}

    async def create(self, record: CronRecord) -> None:
        self._store[record.cron_id] = record.model_copy(deep=True)

    async def get(self, cron_id: str) -> CronRecord | None:
        rec = self._store.get(cron_id)
        return rec.model_copy(deep=True) if rec is not None else None

    async def list(self) -> _List[CronRecord]:
        records = sorted(self._store.values(), key=lambda c: c.created_at)
        return [r.model_copy(deep=True) for r in records]

    async def delete(self, cron_id: str) -> bool:
        return self._store.pop(cron_id, None) is not None

    async def due(self, *, now: float | None = None) -> _List[CronRecord]:
        moment = time.time() if now is None else now
        due = [r for r in self._store.values() if r.enabled and r.next_run_at <= moment]
        due.sort(key=lambda c: c.next_run_at)
        return [r.model_copy(deep=True) for r in due]

    async def mark_run(self, cron_id: str, *, now: float | None = None) -> CronRecord | None:
        rec = self._store.get(cron_id)
        if rec is None:
            return None
        moment = time.time() if now is None else now
        updated = rec.model_copy(update=_advance(rec, moment))
        self._store[cron_id] = updated
        return updated.model_copy(deep=True)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS crons (
    cron_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL,
    next_run_at REAL NOT NULL,
    created_at REAL NOT NULL,
    data TEXT NOT NULL
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_crons_enabled_next ON crons (enabled, next_run_at)"


class SQLiteCronStore:
    """Persist schedules in a SQLite ``crons`` table keyed by schedule id.

    Schedules survive a restart and are visible to every store view over the
    same file, so two processes on one host see one shared schedule set. The
    full record is stored as a JSON ``data`` column; ``enabled`` and
    ``next_run_at`` are mirrored into indexed columns so the due-scan stays
    cheap.
    """

    def __init__(self, path: str = "yaab_crons.db") -> None:
        # ``isolation_level=None`` gives explicit transaction control so a tick's
        # claim-and-advance is an atomic read-modify-write. check_same_thread=False
        # lets the served worker thread and request handlers share the store
        # (sqlite3 serialized mode makes that safe).
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> CronRecord:
        return CronRecord.model_validate_json(row[0])

    def _write(self, record: CronRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO crons "
            "(cron_id, enabled, next_run_at, created_at, data) VALUES (?, ?, ?, ?, ?)",
            (
                record.cron_id,
                1 if record.enabled else 0,
                record.next_run_at,
                record.created_at,
                record.model_dump_json(),
            ),
        )

    def _read(self, cron_id: str) -> CronRecord | None:
        row = self._conn.execute("SELECT data FROM crons WHERE cron_id = ?", (cron_id,)).fetchone()
        return self._row_to_record(row) if row is not None else None

    async def create(self, record: CronRecord) -> None:
        self._write(record)

    async def get(self, cron_id: str) -> CronRecord | None:
        return self._read(cron_id)

    async def list(self) -> _List[CronRecord]:
        rows = self._conn.execute("SELECT data FROM crons ORDER BY created_at ASC").fetchall()
        return [self._row_to_record(r) for r in rows]

    async def delete(self, cron_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM crons WHERE cron_id = ?", (cron_id,))
        return cur.rowcount > 0

    async def due(self, *, now: float | None = None) -> _List[CronRecord]:
        moment = time.time() if now is None else now
        rows = self._conn.execute(
            "SELECT data FROM crons WHERE enabled = 1 AND next_run_at <= ? "
            "ORDER BY next_run_at ASC",
            (moment,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    async def mark_run(self, cron_id: str, *, now: float | None = None) -> CronRecord | None:
        moment = time.time() if now is None else now
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rec = self._read(cron_id)
            if rec is None:
                self._conn.execute("COMMIT")
                return None
            updated = rec.model_copy(update=_advance(rec, moment))
            self._write(updated)
            self._conn.execute("COMMIT")
            return updated
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


__all__ = [
    "CronRecord",
    "CronStore",
    "InMemoryCronStore",
    "SQLiteCronStore",
    "parse_schedule",
]
