"""Durable background runs that survive restarts and span replicas.

The run store is the cross-process system-of-record for every run's lifecycle:
a run becomes a durable row that survives a restart, is visible from any replica
behind a load balancer, and carries everything needed to poll, cancel, lease,
and resume it. Backends mirror the session/checkpoint trio exactly:

* :class:`InMemoryRunStore` — process-local default for dev and tests.
* :class:`SQLiteRunStore` — durable on a single node (atomic worker claim).
* :class:`PostgresRunStore` — true multi-replica HA (``FOR UPDATE SKIP LOCKED``).
* :class:`RedisRunStore` — a distributed, durable run queue.

:class:`StoreCancellationToken` bridges the runner's cooperative cancel to the
store's durable flag so a cancel on any replica stops the run everywhere.

Each backend is registered under the ``run`` component kind, so it can be
selected by name: ``yaab.extensions.get("run", "sqlite", path=...)``.
"""

from __future__ import annotations

from typing import Any

from .base import RunRecord, RunStatus, RunStore
from .cancel import StoreCancellationToken
from .cron import (
    CronRecord,
    CronStore,
    InMemoryCronStore,
    SQLiteCronStore,
    parse_schedule,
)
from .memory import InMemoryRunStore
from .safety import warn_if_ephemeral
from .sqlite import SQLiteRunStore
from .trace import (
    InMemoryTraceStore,
    SQLiteTraceStore,
    TraceStore,
)
from .worker import RunWorker

__all__ = [
    # run store
    "RunStore",
    "RunRecord",
    "RunStatus",
    "InMemoryRunStore",
    "SQLiteRunStore",
    "PostgresRunStore",
    "RedisRunStore",
    "StoreCancellationToken",
    # background worker + schedules
    "RunWorker",
    "CronStore",
    "CronRecord",
    "InMemoryCronStore",
    "SQLiteCronStore",
    "parse_schedule",
    # per-run trace store
    "TraceStore",
    "InMemoryTraceStore",
    "SQLiteTraceStore",
    "PostgresTraceStore",
    "RedisTraceStore",
    # startup durability guardrail
    "warn_if_ephemeral",
]


def __getattr__(name: str) -> Any:
    # Lazy imports so psycopg / redis are only needed when their backend is used.
    if name == "PostgresRunStore":
        from .postgres import PostgresRunStore

        return PostgresRunStore
    if name == "RedisRunStore":
        from .redis import RedisRunStore

        return RedisRunStore
    if name == "PostgresTraceStore":
        from .trace import PostgresTraceStore

        return PostgresTraceStore
    if name == "RedisTraceStore":
        from .trace import RedisTraceStore

        return RedisTraceStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_backends() -> None:
    """Register run-store backends as ``run`` components (discoverable by name)."""
    from ..extensions import register

    register("run", "memory", lambda **kw: InMemoryRunStore())
    register("run", "sqlite", lambda **kw: SQLiteRunStore(**kw))

    def _pg(**kw: Any) -> Any:
        from .postgres import PostgresRunStore

        return PostgresRunStore(**kw)

    def _redis(**kw: Any) -> Any:
        from .redis import RedisRunStore

        return RedisRunStore(**kw)

    register("run", "postgres", _pg)
    register("run", "aurora", _pg)  # Aurora PostgreSQL via the same driver
    register("run", "redis", _redis)

    # Schedules are a durable store kind too.
    register("cron", "memory", lambda **kw: InMemoryCronStore())
    register("cron", "sqlite", lambda **kw: SQLiteCronStore(**kw))


# Importing ``.trace`` above already registered the ``trace`` backends; this adds
# the ``run`` and ``cron`` backends so a fresh ``import yaab.runs`` wires them all.
_register_backends()
