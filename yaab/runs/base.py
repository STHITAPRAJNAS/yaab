"""Durable run store — the cross-process system-of-record for every run.

A run is no longer a fleeting in-process task: it is a durable row that
survives a restart, is visible from any replica behind a load balancer, and
carries everything needed to poll, cancel, lease, and resume it. This module
defines the record shape and the pluggable backend protocol; the backends
(in-memory, SQLite, Postgres, Redis) live alongside it and mirror the session
backends exactly.

The protocol has three concerns:

* **Lifecycle** — :meth:`RunStore.create` / :meth:`get` / :meth:`update` /
  :meth:`list` track a run from queued to a terminal state.
* **Cross-replica cancel** — :meth:`request_cancel` flips a durable flag any
  replica can observe, so a cancel issued on one replica stops the run on the
  replica executing it.
* **Worker queue primitives** — :meth:`claim_next` / :meth:`heartbeat` /
  :meth:`reap_expired_leases` let a fleet of workers drain the queue with
  bounded concurrency and recover runs abandoned by a crashed replica.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# Alias for ``list[str]`` used by methods declared after ``def list`` (whose
# name would otherwise shadow the builtin in those return annotations).
_RunIds = list


class RunStatus(str, Enum):
    """Where a run is in its lifecycle.

    ``PAUSED`` marks a run sleeping in the store awaiting an out-of-band human
    decision; it consumes no compute and can resume on any replica.
    """

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Statuses a run can no longer move out of (cancel/claim are no-ops on these).
TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)


class RunRecord(BaseModel):
    """The durable record of a single run.

    Everything a remote caller or a worker on another replica needs to poll,
    cancel, lease, resume, or report on the run lives here. ``output`` and
    ``usage`` are kept JSON-safe so the record serializes to any backend.
    """

    run_id: str
    agent: str
    status: RunStatus = RunStatus.QUEUED
    prompt: str = ""
    session_id: str | None = None
    identity: str | None = None
    background: bool = False
    resume_id: str | None = None
    output: Any | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    owner_pod: str | None = None
    lease_expires_at: float | None = None
    #: Monotonic fencing token bumped on every (re)claim and every reap. A worker
    #: only finalizes/heartbeats a run while the record's generation still matches
    #: the one it claimed — so a stale worker whose lease was reaped and re-queued
    #: cannot overwrite the result of the newer worker that re-claimed the run.
    #: This is the standard fencing-token solution (cf. Chubby/Zookeeper).
    lease_generation: int = 0


@runtime_checkable
class RunStore(Protocol):
    """Pluggable, durable backend for run lifecycle, cancel, and the worker queue.

    Implementations are interchangeable: an in-memory dict for single-process
    dev, SQLite for a single durable node, Postgres for true multi-replica HA,
    Redis for a distributed queue. Swapping one in is a one-line change.
    """

    async def create(self, record: RunRecord) -> None:
        """Persist a new run record."""
        ...

    async def get(self, run_id: str) -> RunRecord | None:
        """Return the record, or ``None`` if no such run exists."""
        ...

    async def update(
        self, run_id: str, *, expect_status: RunStatus | None = None, **fields: Any
    ) -> RunRecord | None:
        """Atomically patch the given fields and return the updated record.

        Returns ``None`` if the run does not exist. ``updated_at`` is refreshed
        automatically.

        When ``expect_status`` is given, the patch is a guarded compare-and-set:
        it only applies (and returns the record) if the run is *currently* in
        that status, and returns ``None`` otherwise. This lets a caller win a
        race exactly once — e.g. only one replica flips a ``PAUSED`` run to
        ``RUNNING`` to resume it after an approval decision.
        """
        ...

    async def list(self, *, limit: int = 100, status: RunStatus | None = None) -> list[RunRecord]:
        """Return recent runs newest-first, optionally filtered by status."""
        ...

    async def request_cancel(self, run_id: str) -> bool:
        """Flag the run for cancellation. Returns ``True`` if the run existed.

        This is the cross-replica cancel signal: any replica can call it; the
        replica executing the run observes the flag and stops cooperatively.
        """
        ...

    async def claim_next(self, *, pod_id: str, lease_seconds: float) -> RunRecord | None:
        """Atomically claim the oldest queued run for ``pod_id``.

        Marks it running, records the owner and a lease deadline, and returns
        it — or ``None`` if the queue is empty. Exactly one claimer ever wins a
        given row, even under concurrency.
        """
        ...

    async def heartbeat(self, run_id: str, *, pod_id: str, lease_seconds: float) -> None:
        """Extend the lease on a run this pod is executing."""
        ...

    async def reap_expired_leases(self) -> _RunIds[str]:
        """Re-queue running runs whose lease has expired (crash recovery).

        Returns the ids re-queued. A run abandoned by a crashed replica becomes
        claimable again by another replica.
        """
        ...
