"""ISO/IEC 42001 (AI management system) compliance mapper.

A lightweight projection onto the AIMS clauses most directly evidenced by the
governance data model (lifecycle, operational controls, performance evaluation).
"""

from __future__ import annotations

from typing import Optional

from ..audit import AuditLog
from ..registry import AgentRegistry
from .base import ComplianceReport, ControlResult, ControlStatus


class ISO42001Mapper:
    regime = "iso_42001"

    def map(
        self, registry: AgentRegistry, audit: AuditLog, agent_id: Optional[str] = None
    ) -> ComplianceReport:
        card = registry.get(agent_id) if agent_id else None
        controls = [
            ControlResult(
                id="6.1",
                title="Actions to address AI risks & opportunities",
                status=ControlStatus.SATISFIED
                if (card and card.risk_tier)
                else ControlStatus.PARTIAL,
                evidence=[f"risk_tier={card.risk_tier.value}"] if card else [],
            ),
            ControlResult(
                id="8.1",
                title="Operational planning & control (lifecycle)",
                status=ControlStatus.SATISFIED if registry.list() else ControlStatus.GAP,
                evidence=["lifecycle FSM enforced"],
            ),
            ControlResult(
                id="9.1",
                title="Monitoring, measurement, analysis & evaluation",
                status=ControlStatus.SATISFIED if audit.verify() else ControlStatus.GAP,
                evidence=["tamper-evident audit log"],
            ),
            ControlResult(
                id="A.6.2.6",
                title="AI system operation & monitoring records",
                status=ControlStatus.SATISFIED if audit.events else ControlStatus.PARTIAL,
                evidence=[f"{len(audit.events)} audit events"],
            ),
        ]
        return ComplianceReport(regime=self.regime, agent_id=agent_id, controls=controls)
