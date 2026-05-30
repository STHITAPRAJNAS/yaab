"""Lifecycle Manager — the model-risk lifecycle as a finite-state machine.

States and transitions are SR 11-7 aligned. Each transition requires evidence
artifacts (development docs, validation report, effective-challenge sign-off,
change-control logs, monitoring reports, retirement record) and is fully
audited. Illegal transitions raise :class:`LifecycleError`.
"""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field

from ..exceptions import LifecycleError
from .audit import AuditKind, AuditLog
from .registry import AgentRegistry, ApprovalStatus


class LifecycleState(str, Enum):
    DRAFT = "DRAFT"
    IN_DEVELOPMENT = "IN_DEVELOPMENT"
    IN_VALIDATION = "IN_VALIDATION"
    APPROVED = "APPROVED"
    DEPLOYED = "DEPLOYED"
    MONITORED = "MONITORED"
    SUSPENDED = "SUSPENDED"
    REJECTED = "REJECTED"
    DECOMMISSIONED = "DECOMMISSIONED"


# Allowed transitions and the evidence each one requires.
_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
    LifecycleState.DRAFT: {LifecycleState.IN_DEVELOPMENT, LifecycleState.REJECTED},
    LifecycleState.IN_DEVELOPMENT: {LifecycleState.IN_VALIDATION, LifecycleState.REJECTED},
    LifecycleState.IN_VALIDATION: {
        LifecycleState.APPROVED,
        LifecycleState.REJECTED,
        LifecycleState.IN_DEVELOPMENT,
    },
    LifecycleState.APPROVED: {LifecycleState.DEPLOYED, LifecycleState.SUSPENDED},
    LifecycleState.DEPLOYED: {
        LifecycleState.MONITORED,
        LifecycleState.SUSPENDED,
        LifecycleState.DECOMMISSIONED,
    },
    LifecycleState.MONITORED: {
        LifecycleState.SUSPENDED,
        LifecycleState.DECOMMISSIONED,
        LifecycleState.IN_VALIDATION,  # periodic re-validation
    },
    LifecycleState.SUSPENDED: {
        LifecycleState.DEPLOYED,
        LifecycleState.DECOMMISSIONED,
        LifecycleState.IN_VALIDATION,
    },
    LifecycleState.REJECTED: {LifecycleState.IN_DEVELOPMENT},
    LifecycleState.DECOMMISSIONED: set(),
}

# Evidence required to *enter* a state (key) — SR 11-7 mapped.
REQUIRED_EVIDENCE: dict[LifecycleState, list[str]] = {
    LifecycleState.IN_DEVELOPMENT: ["development_docs", "conceptual_soundness"],
    LifecycleState.IN_VALIDATION: ["validation_plan"],
    LifecycleState.APPROVED: ["validation_report", "effective_challenge_signoff"],
    LifecycleState.DEPLOYED: ["change_control_record"],
    LifecycleState.MONITORED: ["monitoring_plan"],
    LifecycleState.DECOMMISSIONED: ["retirement_record"],
}


class EvidenceArtifact(BaseModel):
    """A piece of evidence attached at a lifecycle transition."""

    kind: str
    summary: str = ""
    author: str | None = None
    reference: str | None = None
    timestamp: float = Field(default_factory=time.time)


class LifecycleManager:
    """Drives agents through the model-risk lifecycle with audited transitions."""

    def __init__(self, registry: AgentRegistry, audit: AuditLog) -> None:
        self.registry = registry
        self.audit = audit
        self._evidence: dict[str, list[EvidenceArtifact]] = {}

    def evidence_for(self, agent_id: str) -> list[EvidenceArtifact]:
        return list(self._evidence.get(agent_id, []))

    def transition(
        self,
        agent_id: str,
        to: LifecycleState,
        *,
        actor: str | None = None,
        evidence: list[EvidenceArtifact] | None = None,
    ) -> LifecycleState:
        card = self.registry.get(agent_id)
        if card is None:
            raise LifecycleError(f"agent '{agent_id}' is not registered")

        current = LifecycleState(card.lifecycle_state)
        if to not in _TRANSITIONS.get(current, set()):
            raise LifecycleError(f"illegal transition {current.value} -> {to.value}")

        # Validate required evidence is present (accumulated or supplied now).
        supplied = {e.kind for e in (evidence or [])}
        have = {e.kind for e in self._evidence.get(agent_id, [])} | supplied
        missing = [k for k in REQUIRED_EVIDENCE.get(to, []) if k not in have]
        if missing:
            raise LifecycleError(f"cannot enter {to.value}: missing evidence {missing}")

        if evidence:
            self._evidence.setdefault(agent_id, []).extend(evidence)

        card.lifecycle_state = to.value
        if to is LifecycleState.APPROVED:
            card.model_approval_status = ApprovalStatus.APPROVED
            card.last_audit_date = time.time()
        elif to is LifecycleState.REJECTED:
            card.model_approval_status = ApprovalStatus.REJECTED
        elif to in (LifecycleState.SUSPENDED, LifecycleState.DECOMMISSIONED):
            card.model_approval_status = ApprovalStatus.REVOKED
        self.registry.register(card)

        self.audit.record(
            AuditKind.LIFECYCLE,
            agent_id=agent_id,
            version=card.version,
            identity=actor,
            from_state=current.value,
            to_state=to.value,
            evidence=sorted(supplied),
        )
        return to

    def add_evidence(self, agent_id: str, artifact: EvidenceArtifact) -> None:
        self._evidence.setdefault(agent_id, []).append(artifact)
