"""Durable store for out-of-band human sign-off on sensitive tool calls.

When a guarded tool (a wire transfer, an account deletion) is reached and no
inline approver resolves it, the run is parked as a durable *pending approval
record* instead of blocking a thread. A reviewer later approves or denies it
from any replica, and the run resumes from its last checkpoint — so a paused
approval consumes zero compute and survives restarts.

The store is the system-of-record for those records. It mirrors the storage
pattern used elsewhere in the SDK: one protocol plus four interchangeable
backends in a single module — an in-memory default for tests and single-process
dev, plus SQLite, Postgres, and Redis for durable, multi-replica deployments.
Each backend is registered under the ``approval`` component kind so it can be
selected by name.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ApprovalDecision(str, Enum):
    """The lifecycle state of a pending approval."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class ApprovalRequest(BaseModel):
    """A durable record of a sensitive tool call awaiting human sign-off.

    The correlation ids tie the record back to the parked run: ``run_id`` is the
    run it belongs to and ``resume_id`` is the checkpoint key the loop resumes
    from once a reviewer decides. ``tool`` and ``arguments`` are surfaced to the
    reviewer so they can judge the request.
    """

    approval_id: str = Field(default_factory=lambda: f"ap_{uuid.uuid4().hex[:12]}")
    run_id: str
    resume_id: str
    agent: str
    identity: str | None = None
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision: ApprovalDecision = ApprovalDecision.PENDING
    reviewer: str | None = None
    reason: str | None = None
    created_at: float = Field(default_factory=time.time)
    decided_at: float | None = None
    expires_at: float | None = None


@runtime_checkable
class ApprovalStore(Protocol):
    """Pluggable storage for pending and decided approval records.

    Implementations are durable and safe to share across replicas: a request
    persisted on one process is visible — and decidable — from any other.
    """

    async def create(self, req: ApprovalRequest) -> None:
        """Persist a new pending approval request."""
        ...

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        """Fetch one request by id, or ``None`` if unknown."""
        ...

    async def list_pending(self, *, agent: str | None = None) -> list[ApprovalRequest]:
        """List still-pending requests, optionally scoped to one agent."""
        ...

    async def decide(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reviewer: str,
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        """Record a reviewer's decision; returns the updated record or ``None``."""
        ...

    async def for_run(self, run_id: str) -> list[ApprovalRequest]:
        """All approval records (pending or decided) belonging to a run."""
        ...


def _apply_decision(
    req: ApprovalRequest,
    *,
    decision: ApprovalDecision,
    reviewer: str,
    reason: str | None,
) -> ApprovalRequest:
    """Return a copy of ``req`` with a reviewer's decision applied."""
    return req.model_copy(
        update={
            "decision": decision,
            "reviewer": reviewer,
            "reason": reason,
            "decided_at": time.time(),
        }
    )


class InMemoryApprovalStore:
    """Process-local approval store — the default for tests and single-process dev.

    Holds records in a dict; nothing survives the process, so swap in a durable
    backend before running more than one replica.
    """

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRequest] = {}

    async def create(self, req: ApprovalRequest) -> None:
        self._records[req.approval_id] = req

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._records.get(approval_id)

    async def list_pending(self, *, agent: str | None = None) -> list[ApprovalRequest]:
        return [
            r
            for r in self._records.values()
            if r.decision == ApprovalDecision.PENDING and (agent is None or r.agent == agent)
        ]

    async def decide(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reviewer: str,
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        existing = self._records.get(approval_id)
        if existing is None:
            return None
        updated = _apply_decision(existing, decision=decision, reviewer=reviewer, reason=reason)
        self._records[approval_id] = updated
        return updated

    async def for_run(self, run_id: str) -> list[ApprovalRequest]:
        return [r for r in self._records.values() if r.run_id == run_id]


class SQLiteApprovalStore:
    """Durable approval store backed by SQLite for single-node deployments.

    Records are stored as JSON keyed by ``approval_id`` with indexed ``run_id``,
    ``agent``, and ``decision`` columns so two views over one database file see
    each other's pending records — the floor for resuming a parked run on a
    different worker than the one that paused it.
    """

    def __init__(self, path: str = "yaab_approvals.db") -> None:
        import sqlite3

        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS approvals ("
            "approval_id TEXT PRIMARY KEY, run_id TEXT, agent TEXT, "
            "decision TEXT, created_at REAL, data TEXT NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals (run_id)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_approvals_pending ON approvals (decision, agent)"
        )
        self._conn.commit()

    def _store(self, req: ApprovalRequest) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO approvals "
            "(approval_id, run_id, agent, decision, created_at, data) VALUES (?, ?, ?, ?, ?, ?)",
            (
                req.approval_id,
                req.run_id,
                req.agent,
                req.decision.value,
                req.created_at,
                req.model_dump_json(),
            ),
        )
        self._conn.commit()

    def _load(self, approval_id: str) -> ApprovalRequest | None:
        row = self._conn.execute(
            "SELECT data FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return ApprovalRequest.model_validate_json(row[0]) if row else None

    async def create(self, req: ApprovalRequest) -> None:
        self._store(req)

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._load(approval_id)

    async def list_pending(self, *, agent: str | None = None) -> list[ApprovalRequest]:
        if agent is None:
            rows = self._conn.execute(
                "SELECT data FROM approvals WHERE decision = ? ORDER BY created_at",
                (ApprovalDecision.PENDING.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM approvals WHERE decision = ? AND agent = ? ORDER BY created_at",
                (ApprovalDecision.PENDING.value, agent),
            ).fetchall()
        return [ApprovalRequest.model_validate_json(r[0]) for r in rows]

    async def decide(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reviewer: str,
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        existing = self._load(approval_id)
        if existing is None:
            return None
        updated = _apply_decision(existing, decision=decision, reviewer=reviewer, reason=reason)
        self._store(updated)
        return updated

    async def for_run(self, run_id: str) -> list[ApprovalRequest]:
        rows = self._conn.execute(
            "SELECT data FROM approvals WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()
        return [ApprovalRequest.model_validate_json(r[0]) for r in rows]


def _require_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "psycopg is required for PostgresApprovalStore. "
            "Install with `pip install 'yaab-sdk[postgres]'`."
        ) from exc
    return psycopg


class PostgresApprovalStore:
    """Durable approval store backed by Postgres / Aurora PostgreSQL.

    Uses ``psycopg`` (``pip install 'yaab-sdk[postgres]'``), imported lazily, so
    the dependency is only needed when this backend is actually constructed. The
    true multi-replica backend: any pod can list and decide a pending request.
    """

    def __init__(self, dsn: str, *, table: str = "yaab_approvals") -> None:
        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"approval_id TEXT PRIMARY KEY, run_id TEXT, agent TEXT, "
            f"decision TEXT, created_at DOUBLE PRECISION, data JSONB NOT NULL)"
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_run ON {table} (run_id)")
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_pending ON {table} (decision, agent)"
        )

    def _store(self, req: ApprovalRequest) -> None:
        self._conn.execute(
            f"INSERT INTO {self._table} "
            f"(approval_id, run_id, agent, decision, created_at, data) "
            f"VALUES (%s, %s, %s, %s, %s, %s) "
            f"ON CONFLICT (approval_id) DO UPDATE SET decision = EXCLUDED.decision, "
            f"data = EXCLUDED.data",
            (
                req.approval_id,
                req.run_id,
                req.agent,
                req.decision.value,
                req.created_at,
                json.dumps(req.model_dump()),
            ),
        )

    def _load(self, approval_id: str) -> ApprovalRequest | None:
        row = self._conn.execute(
            f"SELECT data FROM {self._table} WHERE approval_id = %s", (approval_id,)
        ).fetchone()
        return ApprovalRequest.model_validate(row[0]) if row else None

    async def create(self, req: ApprovalRequest) -> None:
        self._store(req)

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._load(approval_id)

    async def list_pending(self, *, agent: str | None = None) -> list[ApprovalRequest]:
        if agent is None:
            rows = self._conn.execute(
                f"SELECT data FROM {self._table} WHERE decision = %s ORDER BY created_at",
                (ApprovalDecision.PENDING.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT data FROM {self._table} WHERE decision = %s AND agent = %s "
                f"ORDER BY created_at",
                (ApprovalDecision.PENDING.value, agent),
            ).fetchall()
        return [ApprovalRequest.model_validate(r[0]) for r in rows]

    async def decide(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reviewer: str,
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        existing = self._load(approval_id)
        if existing is None:
            return None
        updated = _apply_decision(existing, decision=decision, reviewer=reviewer, reason=reason)
        self._store(updated)
        return updated

    async def for_run(self, run_id: str) -> list[ApprovalRequest]:
        rows = self._conn.execute(
            f"SELECT data FROM {self._table} WHERE run_id = %s ORDER BY created_at", (run_id,)
        ).fetchall()
        return [ApprovalRequest.model_validate(r[0]) for r in rows]


class RedisApprovalStore:
    """Durable approval store backed by Redis / ElastiCache / MemoryDB.

    Uses ``redis`` (``pip install 'yaab-sdk[redis]'``), imported lazily; a
    pre-built client may be injected for tests. Each request is a JSON value in a
    per-id hash field, with a pending-id set and per-run id set for fast listing,
    so any replica sees and decides the same records.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "yaab:approval",
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
                    "redis is required for RedisApprovalStore. "
                    "Install with `pip install 'yaab-sdk[redis]'`."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _data_key(self) -> str:
        return f"{self._prefix}:data"

    def _pending_key(self) -> str:
        return f"{self._prefix}:pending"

    def _run_key(self, run_id: str) -> str:
        return f"{self._prefix}:run:{run_id}"

    def _store(self, req: ApprovalRequest) -> None:
        self._redis.hset(self._data_key(), req.approval_id, req.model_dump_json())
        self._redis.sadd(self._run_key(req.run_id), req.approval_id)
        if req.decision == ApprovalDecision.PENDING:
            self._redis.sadd(self._pending_key(), req.approval_id)

    def _load(self, approval_id: str) -> ApprovalRequest | None:
        raw = self._redis.hget(self._data_key(), approval_id)
        return ApprovalRequest.model_validate_json(raw) if raw else None

    async def create(self, req: ApprovalRequest) -> None:
        self._store(req)

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._load(approval_id)

    async def list_pending(self, *, agent: str | None = None) -> list[ApprovalRequest]:
        ids = self._redis.smembers(self._pending_key())
        out: list[ApprovalRequest] = []
        for approval_id in ids:
            req = self._load(approval_id)
            if req is None or req.decision != ApprovalDecision.PENDING:
                continue
            if agent is not None and req.agent != agent:
                continue
            out.append(req)
        out.sort(key=lambda r: r.created_at)
        return out

    async def decide(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reviewer: str,
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        existing = self._load(approval_id)
        if existing is None:
            return None
        updated = _apply_decision(existing, decision=decision, reviewer=reviewer, reason=reason)
        self._redis.hset(self._data_key(), updated.approval_id, updated.model_dump_json())
        if updated.decision != ApprovalDecision.PENDING:
            self._redis.srem(self._pending_key(), updated.approval_id)
        return updated

    async def for_run(self, run_id: str) -> list[ApprovalRequest]:
        ids = self._redis.smembers(self._run_key(run_id))
        out: list[ApprovalRequest] = []
        for approval_id in ids:
            req = self._load(approval_id)
            if req is not None:
                out.append(req)
        out.sort(key=lambda r: r.created_at)
        return out


def _register_backends() -> None:
    """Register approval backends as ``approval`` components (selectable by name)."""
    from ..extensions import register

    register("approval", "memory", lambda **kw: InMemoryApprovalStore())
    register("approval", "sqlite", lambda **kw: SQLiteApprovalStore(**kw))

    def _pg(**kw: Any) -> Any:
        return PostgresApprovalStore(**kw)

    def _redis(**kw: Any) -> Any:
        return RedisApprovalStore(**kw)

    register("approval", "postgres", _pg)
    register("approval", "aurora", _pg)  # Aurora PostgreSQL via the same driver
    register("approval", "redis", _redis)


_register_backends()


__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "SQLiteApprovalStore",
    "PostgresApprovalStore",
    "RedisApprovalStore",
]
