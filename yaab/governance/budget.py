"""Multi-tenant spend governance: a durable spend ledger + an enforcing plugin.

A :class:`SpendStore` is an append-only ledger of ``(key, usd, at)`` rows; a
:class:`SpendGovernancePlugin` records each model call's cost against a run's
identity (and optional tenant) and blocks a run that is already over budget.

Enforcement is **post-hoc**: a model call's exact cost is only known after it
returns, so a key over budget blocks the *next* run rather than a mid-call one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from ..models.base import ModelResponse
from ..plugins import Plugin
from ..plugins.builtins import BudgetExceeded
from ..types import RunContext

_WINDOW_SECONDS = {"day": 86_400.0, "month": 2_592_000.0}


@dataclass
class Budget:
    """A spend cap for one key over a rolling window."""

    limit_usd: float
    window: Literal["lifetime", "day", "month"] = "lifetime"


#: Budgets are app-owned policy: a mapping keyed by the same opaque key the
#: plugin derives (``id:alice`` / ``tenant:acme``), or a resolver callable.
BudgetPolicy = Mapping[str, Budget] | Callable[[str], "Budget | None"]


def _window_start(budget: Budget, now: float) -> float | None:
    """The earliest timestamp counted toward ``budget`` at ``now`` (None=lifetime)."""
    span = _WINDOW_SECONDS.get(budget.window)
    return None if span is None else now - span


class SpendStore(Protocol):
    """An append-only spend ledger keyed by an opaque string."""

    async def record(self, key: str, usd: float, *, at: float) -> None: ...

    async def total(self, key: str, *, since: float | None = None) -> float: ...


class InMemorySpendStore:
    """Process-local spend ledger — the default for tests and single-process dev."""

    def __init__(self) -> None:
        self._entries: list[tuple[str, float, float]] = []

    async def record(self, key: str, usd: float, *, at: float) -> None:
        self._entries.append((key, usd, at))

    async def total(self, key: str, *, since: float | None = None) -> float:
        return sum(
            usd for k, usd, at in self._entries if k == key and (since is None or at >= since)
        )


class SQLiteSpendStore:
    """Durable spend ledger backed by SQLite — durable on a single node.

    Two views over one database file see each other's spend, so a paused/over-budget
    decision is consistent across worker threads and processes on the same host.
    """

    def __init__(self, path: str = "yaab_spend.db") -> None:
        import sqlite3

        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS spend "
            "(key TEXT NOT NULL, usd REAL NOT NULL, at REAL NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_spend_key_at ON spend (key, at)")

    async def record(self, key: str, usd: float, *, at: float) -> None:
        self._conn.execute("INSERT INTO spend (key, usd, at) VALUES (?, ?, ?)", (key, usd, at))

    async def total(self, key: str, *, since: float | None = None) -> float:
        if since is None:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE key = ?", (key,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE key = ? AND at >= ?",
                (key, since),
            ).fetchone()
        return float(row[0])


class PostgresSpendStore:
    """Durable spend ledger backed by Postgres / Aurora — the multi-pod backend.

    Uses ``psycopg`` (``pip install 'yaab-sdk[postgres]'``), imported lazily, so a
    spend cap is enforced against one shared ledger every pod reads and writes —
    a ``rate``/budget that is global across replicas, not per-pod.
    """

    def __init__(self, dsn: str, *, table: str = "yaab_spend") -> None:
        from ..artifacts.postgres import _require_psycopg

        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            f"(key TEXT NOT NULL, usd DOUBLE PRECISION NOT NULL, at DOUBLE PRECISION NOT NULL)"
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_key_at ON {table} (key, at)")

    async def record(self, key: str, usd: float, *, at: float) -> None:
        self._conn.execute(
            f"INSERT INTO {self._table} (key, usd, at) VALUES (%s, %s, %s)", (key, usd, at)
        )

    async def total(self, key: str, *, since: float | None = None) -> float:
        if since is None:
            row = self._conn.execute(
                f"SELECT COALESCE(SUM(usd), 0) FROM {self._table} WHERE key = %s", (key,)
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT COALESCE(SUM(usd), 0) FROM {self._table} WHERE key = %s AND at >= %s",
                (key, since),
            ).fetchone()
        return float(row[0])


def _budget_for(policy: BudgetPolicy, key: str) -> Budget | None:
    if callable(policy):
        return policy(key)
    return policy.get(key)


class SpendGovernancePlugin(Plugin):
    """Enforce per-identity / per-tenant spend caps across runs.

    ``before_run`` blocks a run whose identity or tenant key is already at/over
    its budget; ``after_model`` records each call's ``cost_usd`` against those
    keys. ``tenant_of`` maps an identity to a tenant key (``None`` = no tenant
    tier). ``clock`` is injectable for tests.
    """

    name = "spend_governance"

    def __init__(
        self,
        store: SpendStore,
        budgets: BudgetPolicy,
        *,
        tenant_of: Callable[[str | None], str | None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.budgets = budgets
        self.tenant_of = tenant_of
        self.clock = clock

    def _keys(self, ctx: RunContext) -> list[str]:
        keys: list[str] = []
        if ctx.identity:
            keys.append(f"id:{ctx.identity}")
        if self.tenant_of is not None:
            tenant = self.tenant_of(ctx.identity)
            if tenant:
                keys.append(f"tenant:{tenant}")
        return keys

    async def before_run(self, ctx: RunContext, agent: str, prompt: str) -> None:
        now = self.clock()
        for key in self._keys(ctx):
            budget = _budget_for(self.budgets, key)
            if budget is None:
                continue
            spent = await self.store.total(key, since=_window_start(budget, now))
            if spent >= budget.limit_usd:
                raise BudgetExceeded(
                    f"{key} is over budget: ${spent:.4f} >= ${budget.limit_usd:.4f} "
                    f"({budget.window})"
                )

    async def after_model(
        self, ctx: RunContext, agent: str, response: ModelResponse
    ) -> ModelResponse | None:
        cost = response.usage.cost_usd
        if cost <= 0:
            return None
        now = self.clock()
        for key in self._keys(ctx):
            await self.store.record(key, cost, at=now)
        return None
