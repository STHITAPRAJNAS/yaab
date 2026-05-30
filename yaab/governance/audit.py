"""Audit log & lineage — the evidence backbone.

An append-only, tamper-evident (hash-chained) record of every run, model call,
tool call, guard decision, lifecycle transition, and human approval. Each entry
folds the previous entry's hash into its own (via the Rust core), so any
retroactive edit breaks the chain and :meth:`AuditLog.verify` detects it.

This is what feeds SR 11-7 ongoing-monitoring evidence and EU AI Act Art. 12
lifetime event logging.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .. import _core

GENESIS = "0" * 64


class AuditKind(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    GUARDRAIL = "guardrail"
    LIFECYCLE = "lifecycle"
    APPROVAL = "approval"
    REGISTRY = "registry"
    ERROR = "error"


class AuditEvent(BaseModel):
    """A single tamper-evident audit entry."""

    id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: float = Field(default_factory=time.time)
    kind: AuditKind
    agent_id: str | None = None
    version: str | None = None
    identity: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def signing_payload(self) -> str:
        """The canonical string that gets hashed into the chain."""
        return json.dumps(
            {
                "id": self.id,
                "timestamp": self.timestamp,
                "kind": self.kind.value,
                "agent_id": self.agent_id,
                "version": self.version,
                "identity": self.identity,
                "payload": self.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@runtime_checkable
class AuditSink(Protocol):
    """A destination for audit events (OTel collector, Logfire, SQL, ...)."""

    def write(self, event: AuditEvent) -> None: ...


class InMemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


class SQLiteAuditSink:
    """Durable audit sink backed by SQLite."""

    def __init__(self, path: str = "yaab_audit.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS audit ("
            "id TEXT PRIMARY KEY, ts REAL, kind TEXT, agent_id TEXT, "
            "prev_hash TEXT, hash TEXT, data TEXT)"
        )
        self._conn.commit()

    def write(self, event: AuditEvent) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO audit VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.timestamp,
                event.kind.value,
                event.agent_id,
                event.prev_hash,
                event.hash,
                event.model_dump_json(),
            ),
        )
        self._conn.commit()


class AuditLog:
    """The hash-chained audit ledger."""

    def __init__(self, sinks: list[AuditSink] | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._last_hash = GENESIS
        self.sinks: list[AuditSink] = sinks if sinks is not None else [InMemoryAuditSink()]

    def record(
        self,
        kind: AuditKind,
        *,
        agent_id: str | None = None,
        version: str | None = None,
        identity: str | None = None,
        **payload: Any,
    ) -> AuditEvent:
        event = AuditEvent(
            kind=kind,
            agent_id=agent_id,
            version=version,
            identity=identity,
            payload=payload,
            prev_hash=self._last_hash,
        )
        event.hash = _core.hash_event(self._last_hash, event.signing_payload())
        self._last_hash = event.hash
        self._events.append(event)
        for sink in self.sinks:
            sink.write(event)
        return event

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def verify(self) -> bool:
        """Return ``True`` iff the hash chain is intact."""
        entries = [(e.signing_payload(), e.hash) for e in self._events]
        return _core.verify_chain(GENESIS, entries) is None

    def for_agent(self, agent_id: str) -> list[AuditEvent]:
        return [e for e in self._events if e.agent_id == agent_id]
