"""Agent Registry — the canonical system-of-record for every agent.

Each entry is an :class:`AgentCard` that extends the A2A Agent Card with the
governance fields a model-risk / AI-governance team needs (the Atlan "enterprise
AI registry" 12-field minimum, plus agent-specific decision-authority and
permission scope). In ``enforcing`` mode the runner refuses to run an agent
unless it is registered and ``model_approval_status == approved``.

The registry can export A2A-compatible discovery cards at
``/.well-known/agent.json``.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RiskTier(str, Enum):
    """Internal risk tiering, orthogonal to any single regulatory regime."""

    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    CRITICAL = "critical"


class EUActCategory(str, Enum):
    """EU AI Act risk categories (Reg. 2024/1689)."""

    PROHIBITED = "prohibited"
    HIGH_RISK = "high_risk"
    LIMITED = "limited"
    MINIMAL = "minimal"
    GPAI = "gpai"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class DecisionAuthority(str, Enum):
    """What the agent is allowed to *do* with its output."""

    ADVISORY = "advisory"  # recommends; a human acts
    AUTOMATED = "automated"  # acts autonomously, reversible
    BINDING = "binding"  # acts autonomously, materially binding


class AgentCard(BaseModel):
    """The registry record for one agent version (A2A-compatible superset).

    ``extra="allow"`` so a central/enterprise registry can attach its own fields
    (e.g. ``usecase_id``, ``blueprint``, cost-center) and have them round-trip
    losslessly through ``model_dump``/JSON. Prefer the typed ``metadata`` dict for
    organization-specific attributes you want to query consistently.
    """

    model_config = ConfigDict(extra="allow")

    # Identity & ownership
    agent_id: str
    name: str
    version: str = "0.1.0"
    business_owner: str | None = None
    technical_owner: str | None = None
    team: str | None = None

    # Purpose & scope
    intended_use_case: str = ""
    output_actions: DecisionAuthority = DecisionAuthority.ADVISORY
    decision_authority: DecisionAuthority = DecisionAuthority.ADVISORY
    action_scope: list[str] = Field(default_factory=list)
    data_inputs: list[str] = Field(default_factory=list)
    training_data_sources: list[str] = Field(default_factory=list)

    # Risk & compliance
    risk_tier: RiskTier = RiskTier.LIMITED
    eu_act_category: EUActCategory = EUActCategory.MINIMAL
    regulatory_exemptions: list[str] = Field(default_factory=list)
    model_approval_status: ApprovalStatus = ApprovalStatus.PENDING
    last_audit_date: float | None = None
    deployment_environment: str = "dev"

    # Operational
    model_versions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    incident_history: list[dict[str, Any]] = Field(default_factory=list)
    model_card_url: str | None = None
    lineage: dict[str, list[str]] = Field(default_factory=dict)

    # Organization-specific attributes (free-form, central-registry friendly):
    # e.g. {"usecase_id": "UC-123", "blueprint": "rag-support-v2", "cost_center": "..."}
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Bookkeeping
    lifecycle_state: str = "DRAFT"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    skills: list[dict[str, Any]] = Field(default_factory=list)

    def to_a2a_card(self, url: str = "") -> dict[str, Any]:
        """Render an A2A-style discovery card for ``/.well-known/agent.json``."""
        return {
            "name": self.name,
            "description": self.intended_use_case,
            "url": url,
            "version": self.version,
            "capabilities": {"streaming": True},
            "skills": self.skills or [{"id": t, "name": t} for t in self.tools],
            "x-yaab-governance": {
                "agent_id": self.agent_id,
                "risk_tier": self.risk_tier.value,
                "eu_act_category": self.eu_act_category.value,
                "approval_status": self.model_approval_status.value,
                "decision_authority": self.decision_authority.value,
                "lifecycle_state": self.lifecycle_state,
            },
        }


@runtime_checkable
class RegistryBackend(Protocol):
    """Pluggable storage for agent cards."""

    def upsert(self, card: AgentCard) -> None: ...

    def fetch(self, agent_id: str) -> AgentCard | None: ...

    def all(self) -> list[AgentCard]: ...


class InMemoryRegistryBackend:
    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def upsert(self, card: AgentCard) -> None:
        self._cards[card.agent_id] = card

    def fetch(self, agent_id: str) -> AgentCard | None:
        return self._cards.get(agent_id)

    def all(self) -> list[AgentCard]:
        return list(self._cards.values())


class SQLiteRegistryBackend:
    """Durable registry backend backed by SQLite."""

    def __init__(self, path: str = "yaab_registry.db") -> None:
        import sqlite3

        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS registry (agent_id TEXT PRIMARY KEY, data TEXT)"
        )
        self._conn.commit()

    def upsert(self, card: AgentCard) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO registry VALUES (?, ?)",
            (card.agent_id, card.model_dump_json()),
        )
        self._conn.commit()

    def fetch(self, agent_id: str) -> AgentCard | None:
        row = self._conn.execute(
            "SELECT data FROM registry WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        return AgentCard.model_validate_json(row[0]) if row else None

    def all(self) -> list[AgentCard]:
        rows = self._conn.execute("SELECT data FROM registry").fetchall()
        return [AgentCard.model_validate_json(r[0]) for r in rows]


class RemoteRegistryBackend:
    """RegistryBackend backed by a central/enterprise HTTP registry service.

    Lets governance enforce against an org-wide system-of-record instead of a
    local store: ``register()`` writes through to the remote service, and the
    enforcing run-gate reads approval status from it on every run.

    Expected REST contract (override ``*_path`` to adapt to your service):

        PUT  {base_url}/agents/{agent_id}   body: AgentCard JSON  -> 2xx
        GET  {base_url}/agents/{agent_id}   -> AgentCard JSON (404 if absent)
        GET  {base_url}/agents             -> [AgentCard, ...] or {"agents": [...]}

    Because ``AgentCard`` allows extra fields, any custom attributes your central
    registry returns (``usecase_id``, ``blueprint``, ...) round-trip intact.

    A pre-built ``httpx.Client`` may be injected (handy for tests via
    ``httpx.MockTransport``); otherwise one is created from ``base_url`` +
    ``headers`` + ``timeout``. Requires the ``http`` extra (``pip install
    'yaab[http]'``).
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            import httpx

            self._client = httpx.Client(
                base_url=base_url.rstrip("/"), headers=headers or {}, timeout=timeout
            )

    def upsert(self, card: AgentCard) -> None:
        resp = self._client.put(
            f"/agents/{card.agent_id}",
            content=card.model_dump_json(),
            headers={"content-type": "application/json"},
        )
        resp.raise_for_status()

    def fetch(self, agent_id: str) -> AgentCard | None:
        resp = self._client.get(f"/agents/{agent_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return AgentCard.model_validate(resp.json())

    def all(self) -> list[AgentCard]:
        resp = self._client.get("/agents")
        resp.raise_for_status()
        body = resp.json()
        items = body.get("agents", []) if isinstance(body, dict) else body
        return [AgentCard.model_validate(x) for x in items]


class AgentRegistry:
    """The registry facade over a pluggable backend."""

    def __init__(self, backend: RegistryBackend | None = None) -> None:
        self.backend = backend or InMemoryRegistryBackend()

    def register(self, card: AgentCard) -> AgentCard:
        card.updated_at = time.time()
        self.backend.upsert(card)
        return card

    def get(self, agent_id: str) -> AgentCard | None:
        return self.backend.fetch(agent_id)

    def list(self) -> list[AgentCard]:
        return self.backend.all()

    def is_approved(self, agent_id: str) -> bool:
        card = self.get(agent_id)
        return bool(card and card.model_approval_status == ApprovalStatus.APPROVED)

    def inventory(self) -> list[dict[str, Any]]:
        """Produce the SR 11-7 / EU AI Act model-inventory view."""
        rows: list[dict[str, Any]] = []
        for c in self.list():
            rows.append(
                {
                    "agent_id": c.agent_id,
                    "name": c.name,
                    "version": c.version,
                    "owner": c.business_owner,
                    "risk_tier": c.risk_tier.value,
                    "eu_act_category": c.eu_act_category.value,
                    "approval_status": c.model_approval_status.value,
                    "lifecycle_state": c.lifecycle_state,
                    "last_audit_date": c.last_audit_date,
                    "deployment_environment": c.deployment_environment,
                    "metadata": c.metadata,
                }
            )
        return rows
