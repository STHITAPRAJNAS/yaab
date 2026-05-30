"""NIST AI RMF 1.0 (AI 100-1) compliance mapper.

Maps governance data onto the four functions: GOVERN (cross-cutting), MAP,
MEASURE, MANAGE.
"""

from __future__ import annotations

from ..audit import AuditLog
from ..registry import AgentRegistry
from .base import ComplianceReport, ControlResult, ControlStatus, _audit_kinds


class NISTAIRMFMapper:
    regime = "nist_ai_rmf"

    def map(
        self, registry: AgentRegistry, audit: AuditLog, agent_id: str | None = None
    ) -> ComplianceReport:
        kinds = _audit_kinds(audit, agent_id)
        card = registry.get(agent_id) if agent_id else None
        controls: list[ControlResult] = []

        controls.append(
            ControlResult(
                id="GOVERN",
                title="Risk culture, roles, and policies in place",
                status=ControlStatus.SATISFIED if registry.list() else ControlStatus.GAP,
                evidence=["registry + lifecycle + policy engine active"],
            )
        )
        controls.append(
            ControlResult(
                id="MAP",
                title="Context & intended purpose documented; AI inventory exists",
                status=ControlStatus.SATISFIED
                if (card and card.intended_use_case)
                else ControlStatus.PARTIAL,
                evidence=["intended_use_case set"] if (card and card.intended_use_case) else [],
            )
        )
        controls.append(
            ControlResult(
                id="MEASURE",
                title="TEVV — testing, evaluation, monitoring",
                status=ControlStatus.SATISFIED
                if "model_call" in kinds or "run_start" in kinds
                else ControlStatus.PARTIAL,
                evidence=["audit-log monitoring; attach Evaluator results"],
            )
        )
        controls.append(
            ControlResult(
                id="MANAGE",
                title="Risk treatment & incident response",
                status=ControlStatus.SATISFIED
                if (card and not card.incident_history) or "guardrail" in kinds
                else ControlStatus.PARTIAL,
                evidence=["guardrail decisions audited"] if "guardrail" in kinds else [],
            )
        )
        return ComplianceReport(regime=self.regime, agent_id=agent_id, controls=controls)
