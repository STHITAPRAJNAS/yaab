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

import asyncio
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
    #: Optional intent-based access control on who may decide this request. When
    #: ``allowed_reviewers`` is non-empty, only an authenticated identity in the
    #: list may approve/deny it; ``required_role`` (when set) names a capability
    #: the caller must hold. Empty/None means any authenticated reviewer may act
    #: (today's behavior), so this is additive and opt-in.
    allowed_reviewers: list[str] = Field(default_factory=list)
    required_role: str | None = None
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
        """Persist a new pending approval request (idempotent on ``approval_id``).

        A re-create with an existing id is a no-op — it must never clobber a
        record a reviewer already decided — so a crash-window re-pause that
        re-derives the same deterministic id self-heals instead of duplicating.
        """
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
        # Serializes decide() so a first decision is permanent within the process.
        self._lock = asyncio.Lock()

    async def create(self, req: ApprovalRequest) -> None:
        # Idempotent: a re-pause after a crash (same deterministic id) must not
        # clobber an existing record — especially not one a reviewer already
        # decided. First write wins; later ones are no-ops.
        self._records.setdefault(req.approval_id, req)

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
        async with self._lock:
            existing = self._records.get(approval_id)
            if existing is None:
                return None
            if existing.decision is not ApprovalDecision.PENDING:
                # First decision is permanent; a later decide is a no-op read.
                return existing
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

        # ``isolation_level=None`` gives explicit transaction control so a decide
        # can take a write lock with BEGIN IMMEDIATE for an atomic read-apply-write.
        # ``check_same_thread=False`` lets the served app's worker thread and
        # request handlers share one store safely (sqlite3 serialized mode).
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS approvals ("
            "approval_id TEXT PRIMARY KEY, run_id TEXT, agent TEXT, "
            "decision TEXT, created_at REAL, data TEXT NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals (run_id)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_approvals_pending ON approvals (decision, agent)"
        )

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

    def _load(self, approval_id: str) -> ApprovalRequest | None:
        row = self._conn.execute(
            "SELECT data FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return ApprovalRequest.model_validate_json(row[0]) if row else None

    async def create(self, req: ApprovalRequest) -> None:
        # Idempotent insert: a crash-window re-pause with the same deterministic
        # id must not clobber an existing (possibly already-decided) record.
        self._conn.execute(
            "INSERT OR IGNORE INTO approvals "
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
        # BEGIN IMMEDIATE takes the write lock up front, so a racing reviewer
        # blocks here (busy_timeout) rather than reading the same PENDING record.
        # The first decision is permanent: a second concurrent decide observes the
        # already-decided record under the lock and returns it unchanged, so no
        # decision is silently overwritten.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self._load(approval_id)
            if existing is None:
                self._conn.execute("COMMIT")
                return None
            if existing.decision is not ApprovalDecision.PENDING:
                self._conn.execute("COMMIT")
                return existing
            updated = _apply_decision(existing, decision=decision, reviewer=reviewer, reason=reason)
            self._store(updated)
            self._conn.execute("COMMIT")
            return updated
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

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

    def _load_locked(self, approval_id: str) -> ApprovalRequest | None:
        """Read the row taking a ``FOR UPDATE`` row lock (within a transaction)."""
        row = self._conn.execute(
            f"SELECT data FROM {self._table} WHERE approval_id = %s FOR UPDATE",
            (approval_id,),
        ).fetchone()
        return ApprovalRequest.model_validate(row[0]) if row else None

    async def create(self, req: ApprovalRequest) -> None:
        # Idempotent insert: a crash-window re-pause with the same deterministic
        # id must not clobber an existing (possibly already-decided) record.
        self._conn.execute(
            f"INSERT INTO {self._table} "
            f"(approval_id, run_id, agent, decision, created_at, data) "
            f"VALUES (%s, %s, %s, %s, %s, %s) "
            f"ON CONFLICT (approval_id) DO NOTHING",
            (
                req.approval_id,
                req.run_id,
                req.agent,
                req.decision.value,
                req.created_at,
                json.dumps(req.model_dump()),
            ),
        )

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
        # Take a row-level ``FOR UPDATE`` lock inside one transaction so two
        # reviewers can't both read PENDING and both write a decision. The first
        # decision is permanent: a second concurrent decide blocks on the lock,
        # then observes the already-decided row and returns it unchanged.
        with self._conn.transaction():
            existing = self._load_locked(approval_id)
            if existing is None:
                return None
            if existing.decision is not ApprovalDecision.PENDING:
                return existing
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
        # Idempotent: HSETNX only writes the record when its field is absent, so a
        # crash-window re-pause with the same deterministic id never clobbers an
        # existing (possibly already-decided) record. The index sets are additive.
        created = self._redis.hsetnx(self._data_key(), req.approval_id, req.model_dump_json())
        self._redis.sadd(self._run_key(req.run_id), req.approval_id)
        if created and req.decision == ApprovalDecision.PENDING:
            self._redis.sadd(self._pending_key(), req.approval_id)

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

    #: Atomic compare-and-set: only flip the record (and drop it from the pending
    #: set) when it is still PENDING. A racing second reviewer's HSET is rejected
    #: server-side, so the first decision is permanent. Returns the stored JSON.
    _DECIDE_LUA = """
    local raw = redis.call('HGET', KEYS[1], ARGV[1])
    if not raw then return nil end
    local rec = cjson.decode(raw)
    if rec['decision'] == 'pending' then
        redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
        redis.call('SREM', KEYS[2], ARGV[1])
        return ARGV[2]
    end
    return raw
    """

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
        candidate = _apply_decision(existing, decision=decision, reviewer=reviewer, reason=reason)
        stored = self._redis.eval(
            self._DECIDE_LUA,
            2,
            self._data_key(),
            self._pending_key(),
            approval_id,
            candidate.model_dump_json(),
        )
        if stored is None:  # pragma: no cover - load already proved it exists
            return None
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        # The script returns the now-authoritative record: our decision if we won
        # the race, or the pre-existing decision if another reviewer got there first.
        return ApprovalRequest.model_validate_json(stored)

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
