"""SR 11-7 / OCC 2011-12 compliance mapper.

Projects governance data onto the three pillars (development, validation,
governance) and validation's three core elements (conceptual soundness, ongoing
monitoring, outcomes analysis), plus the model-inventory and effective-challenge
requirements.
"""

from __future__ import annotations

from typing import Optional

from ..audit import AuditLog
from ..registry import AgentRegistry, ApprovalStatus
from .base import (
    ComplianceReport,
    ControlResult,
    ControlStatus,
    _audit_kinds,
)


class SR117Mapper:
    regime = "sr_11_7"

    def map(
        self, registry: AgentRegistry, audit: AuditLog, agent_id: Optional[str] = None
    ) -> ComplianceReport:
        kinds = _audit_kinds(audit, agent_id)
        card = registry.get(agent_id) if agent_id else None
        controls: list[ControlResult] = []

        # Pillar 1 — development & implementation.
        has_inventory = bool(registry.list())
        controls.append(
            ControlResult(
                id="SR11-7.1",
                title="Model inventory maintained (incl. in-development and retired)",
                status=ControlStatus.SATISFIED if has_inventory else ControlStatus.GAP,
                evidence=[f"{len(registry.list())} registry entries"],
            )
        )

        # Pillar 2 — validation (three core elements).
        approved = card and card.model_approval_status == ApprovalStatus.APPROVED
        controls.append(
            ControlResult(
                id="SR11-7.2a",
                title="Conceptual soundness evaluated",
                status=ControlStatus.SATISFIED
                if "lifecycle" in kinds
                else ControlStatus.PARTIAL,
                evidence=["lifecycle transitions recorded"] if "lifecycle" in kinds else [],
            )
        )
        controls.append(
            ControlResult(
                id="SR11-7.2b",
                title="Ongoing monitoring (process verification + benchmarking)",
                status=ControlStatus.SATISFIED
                if {"model_call", "run_start"} & kinds
                else ControlStatus.GAP,
                evidence=["audit log captures runs/model calls"]
                if {"model_call", "run_start"} & kinds
                else [],
            )
        )
        controls.append(
            ControlResult(
                id="SR11-7.2c",
                title="Outcomes analysis / back-testing",
                status=ControlStatus.PARTIAL,
                notes="attach Evaluator ExperimentResults to the registry entry",
            )
        )

        # Pillar 3 — governance, policies, controls.
        controls.append(
            ControlResult(
                id="SR11-7.3a",
                title="Effective challenge / independent approval sign-off",
                status=ControlStatus.SATISFIED if approved else ControlStatus.GAP,
                evidence=["approval recorded in lifecycle"] if approved else [],
            )
        )
        controls.append(
            ControlResult(
                id="SR11-7.3b",
                title="Tamper-evident audit trail",
                status=ControlStatus.SATISFIED if audit.verify() else ControlStatus.GAP,
                evidence=["hash chain verified"] if audit.verify() else ["hash chain BROKEN"],
            )
        )
        return ComplianceReport(regime=self.regime, agent_id=agent_id, controls=controls)
