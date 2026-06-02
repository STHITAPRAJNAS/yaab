"""Durable approval store: roundtrip, decide flow, and cross-pod visibility.

The store is the system-of-record for out-of-band human sign-off on sensitive
tool calls. These tests cover the in-memory and SQLite backends offline, prove
that two SQLite views over one file see each other's pending records (the
"two replicas" scenario), and confirm the new exception/event types carry their
correlation fields while staying catchable by existing handlers.
"""

from __future__ import annotations

import time

import pytest

from yaab.exceptions import ApprovalPending, ApprovalRequired
from yaab.governance.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStore,
    InMemoryApprovalStore,
    SQLiteApprovalStore,
)
from yaab.types import EventType


def _req(approval_id: str = "ap1", run_id: str = "run1", **over: object) -> ApprovalRequest:
    base: dict[str, object] = {
        "approval_id": approval_id,
        "run_id": run_id,
        "resume_id": over.pop("resume_id", run_id),
        "agent": "svc",
        "identity": "alice",
        "tool": "wire_transfer",
        "arguments": {"amount": 100},
    }
    base.update(over)
    return ApprovalRequest(**base)  # type: ignore[arg-type]


def _stores(tmp_path) -> list[ApprovalStore]:
    return [
        InMemoryApprovalStore(),
        SQLiteApprovalStore(path=str(tmp_path / "approvals.db")),
    ]


# --------------------------------------------------------------------------- #
# Model / enum basics
# --------------------------------------------------------------------------- #


def test_request_defaults_to_pending():
    req = _req()
    assert req.decision == ApprovalDecision.PENDING
    assert req.decided_at is None
    assert req.reviewer is None
    assert req.created_at > 0


def test_decision_enum_values():
    assert ApprovalDecision.PENDING.value == "pending"
    assert ApprovalDecision.APPROVED.value == "approved"
    assert ApprovalDecision.DENIED.value == "denied"
    assert ApprovalDecision.EXPIRED.value == "expired"


# --------------------------------------------------------------------------- #
# Roundtrip + decide on memory and sqlite
# --------------------------------------------------------------------------- #


async def test_create_get_roundtrip(tmp_path):
    for store in _stores(tmp_path):
        req = _req()
        await store.create(req)
        got = await store.get("ap1")
        assert got is not None
        assert got.approval_id == "ap1"
        assert got.run_id == "run1"
        assert got.resume_id == "run1"
        assert got.tool == "wire_transfer"
        assert got.arguments == {"amount": 100}
        assert got.decision == ApprovalDecision.PENDING


async def test_get_missing_returns_none(tmp_path):
    for store in _stores(tmp_path):
        assert await store.get("nope") is None


async def test_list_pending_only(tmp_path):
    for store in _stores(tmp_path):
        await store.create(_req("a", "r1"))
        await store.create(_req("b", "r2"))
        await store.decide("b", decision=ApprovalDecision.APPROVED, reviewer="bob")
        pending = await store.list_pending()
        ids = {p.approval_id for p in pending}
        assert ids == {"a"}


async def test_list_pending_filtered_by_agent(tmp_path):
    for store in _stores(tmp_path):
        await store.create(_req("a", "r1", agent="alpha"))
        await store.create(_req("b", "r2", agent="beta"))
        pending = await store.list_pending(agent="alpha")
        assert {p.approval_id for p in pending} == {"a"}


async def test_decide_approve_marks_record(tmp_path):
    for store in _stores(tmp_path):
        await store.create(_req())
        before = time.time()
        updated = await store.decide(
            "ap1", decision=ApprovalDecision.APPROVED, reviewer="bob", reason="ok"
        )
        assert updated is not None
        assert updated.decision == ApprovalDecision.APPROVED
        assert updated.reviewer == "bob"
        assert updated.reason == "ok"
        assert updated.decided_at is not None and updated.decided_at >= before
        # persisted
        got = await store.get("ap1")
        assert got is not None and got.decision == ApprovalDecision.APPROVED


async def test_decide_deny(tmp_path):
    for store in _stores(tmp_path):
        await store.create(_req())
        updated = await store.decide(
            "ap1", decision=ApprovalDecision.DENIED, reviewer="bob", reason="too risky"
        )
        assert updated is not None
        assert updated.decision == ApprovalDecision.DENIED
        assert updated.reason == "too risky"


async def test_decide_missing_returns_none(tmp_path):
    for store in _stores(tmp_path):
        assert (
            await store.decide("nope", decision=ApprovalDecision.APPROVED, reviewer="bob")
        ) is None


async def test_for_run_returns_all_records_for_run(tmp_path):
    for store in _stores(tmp_path):
        await store.create(_req("a", "shared"))
        await store.create(_req("b", "shared"))
        await store.create(_req("c", "other"))
        recs = await store.for_run("shared")
        assert {r.approval_id for r in recs} == {"a", "b"}


# --------------------------------------------------------------------------- #
# Two-pod simulation: two SQLite views over one file
# --------------------------------------------------------------------------- #


async def test_sqlite_pending_visible_across_two_instances(tmp_path):
    path = str(tmp_path / "shared.db")
    pod_a = SQLiteApprovalStore(path=path)
    pod_b = SQLiteApprovalStore(path=path)

    await pod_a.create(_req("ap1", "run1"))

    # The "other pod" sees the pending record it never created.
    seen = await pod_b.list_pending()
    assert [p.approval_id for p in seen] == ["ap1"]

    # And a decision made on pod B is visible to pod A.
    await pod_b.decide("ap1", decision=ApprovalDecision.APPROVED, reviewer="bob")
    got = await pod_a.get("ap1")
    assert got is not None and got.decision == ApprovalDecision.APPROVED
    assert await pod_a.list_pending() == []


# --------------------------------------------------------------------------- #
# Redis backend via an injected fake client
# --------------------------------------------------------------------------- #


class _FakeRedis:
    """Minimal hash/set client covering what RedisApprovalStore uses."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}

    def hset(self, key, field=None, value=None, mapping=None):
        h = self.hashes.setdefault(key, {})
        if mapping:
            h.update({k: str(v) for k, v in mapping.items()})
        if field is not None:
            h[field] = value

    def hsetnx(self, key, field, value):
        h = self.hashes.setdefault(key, {})
        if field in h:
            return 0
        h[field] = value
        return 1

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def sadd(self, key, *members):
        self.sets.setdefault(key, set()).update(members)

    def srem(self, key, *members):
        s = self.sets.get(key)
        if s is not None:
            s.difference_update(members)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def eval(self, script, numkeys, *args):
        """Interpret the store's compare-and-set decide script (PENDING-guard).

        Only the small Lua used by ``RedisApprovalStore.decide`` is supported:
        HGET the record, and if it is still pending, HSET the candidate and SREM
        it from the pending set; otherwise return the existing raw record.
        """
        import json

        keys = args[:numkeys]
        argv = args[numkeys:]
        data_key, pending_key = keys[0], keys[1]
        approval_id, candidate = argv[0], argv[1]
        raw = self.hashes.get(data_key, {}).get(approval_id)
        if raw is None:
            return None
        if json.loads(raw).get("decision") == "pending":
            self.hashes.setdefault(data_key, {})[approval_id] = candidate
            self.srem(pending_key, approval_id)
            return candidate
        return raw


async def test_redis_backend_with_fake_client_roundtrips():
    from yaab.governance.approvals import RedisApprovalStore

    store = RedisApprovalStore(client=_FakeRedis())
    await store.create(_req("ap1", "run1"))
    await store.create(_req("ap2", "run1"))

    got = await store.get("ap1")
    assert got is not None and got.run_id == "run1"

    pending = await store.list_pending()
    assert {p.approval_id for p in pending} == {"ap1", "ap2"}

    await store.decide("ap1", decision=ApprovalDecision.DENIED, reviewer="bob")
    pending = await store.list_pending()
    assert {p.approval_id for p in pending} == {"ap2"}

    for_run = await store.for_run("run1")
    assert {r.approval_id for r in for_run} == {"ap1", "ap2"}


def test_redis_backend_requires_driver_without_client():
    try:
        import redis  # noqa: F401

        pytest.skip("redis is installed")
    except ImportError:
        pass
    from yaab.governance.approvals import RedisApprovalStore

    with pytest.raises(RuntimeError, match="redis"):
        RedisApprovalStore()


# --------------------------------------------------------------------------- #
# Component registration under the "approval" kind
# --------------------------------------------------------------------------- #


def test_backends_registered_under_approval_kind(tmp_path):
    from yaab.extensions import available, get

    names = set(available("approval"))
    assert {"memory", "sqlite", "postgres", "redis"} <= names

    mem = get("approval", "memory")
    assert isinstance(mem, InMemoryApprovalStore)

    sql = get("approval", "sqlite", path=str(tmp_path / "reg.db"))
    assert isinstance(sql, SQLiteApprovalStore)


def test_postgres_backend_requires_driver():
    try:
        import psycopg  # noqa: F401

        pytest.skip("psycopg is installed")
    except ImportError:
        pass
    from yaab.governance.approvals import PostgresApprovalStore

    with pytest.raises(RuntimeError, match="psycopg"):
        PostgresApprovalStore("postgresql://x")


# --------------------------------------------------------------------------- #
# Exception + event additions (Item 2 contract for runner/plugin agents)
# --------------------------------------------------------------------------- #


def test_approval_pending_carries_correlation_fields():
    exc = ApprovalPending(
        tool="wire_transfer",
        arguments={"amount": 100},
        approval_id="ap1",
        run_id="run1",
        resume_id="res1",
    )
    assert exc.approval_id == "ap1"
    assert exc.run_id == "run1"
    assert exc.resume_id == "res1"
    assert exc.tool == "wire_transfer"
    assert exc.arguments == {"amount": 100}


def test_approval_pending_is_caught_as_approval_required():
    """Back-compat: existing ``except ApprovalRequired`` handlers still catch it."""
    raised: ApprovalRequired | None = None
    try:
        raise ApprovalPending(
            tool="t",
            arguments={},
            approval_id="ap1",
            run_id="run1",
            resume_id="res1",
        )
    except ApprovalRequired as exc:  # the existing handler shape
        raised = exc
    assert isinstance(raised, ApprovalPending)
    assert raised.tool == "t"


def test_event_type_has_approval_required():
    assert EventType.APPROVAL_REQUIRED.value == "approval_required"


def test_event_duration_ms_defaults_to_none():
    from yaab.types import Event

    ev = Event(type=EventType.RUN_END, agent="svc", run_id="run1")
    assert ev.duration_ms is None
    ev2 = Event(type=EventType.MODEL_RESPONSE, agent="svc", run_id="run1", duration_ms=12.5)
    assert ev2.duration_ms == 12.5


async def test_decide_accepts_string_decision(tmp_path):
    """Reviewers shouldn't need the enum import: 'approved' (str) must coerce."""
    store = SQLiteApprovalStore(str(tmp_path / "a.db"))
    req = ApprovalRequest(
        approval_id="ap-str",
        run_id="r1",
        resume_id="r1",
        agent="banker",
        tool="wire_transfer",
        arguments={"amount": 1},
    )
    await store.create(req)
    decided = await store.decide("ap-str", decision="approved", reviewer="alice")
    assert decided is not None
    assert decided.decision is ApprovalDecision.APPROVED
    # Idempotent second decide with a string is also fine.
    again = await store.decide("ap-str", decision="denied", reviewer="bob")
    assert again is not None and again.decision is ApprovalDecision.APPROVED
