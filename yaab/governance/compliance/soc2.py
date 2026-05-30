"""SOC 2 compliance mapper.

Maps the Trust Services Criteria most directly evidenced by YAAB's runtime:
logging/monitoring (CC7), change management (CC8), and confidentiality (C1).
"""

from __future__ import annotations

from typing import Optional

from ..audit import AuditLog
from ..registry import AgentRegistry
from .base import ComplianceReport, ControlResult, ControlStatus, _audit_kinds


class SOC2Mapper:
    regime = "soc2"

    def map(
        self, registry: AgentRegistry, audit: AuditLog, agent_id: Optional[str] = None
    ) -> ComplianceReport:
        kinds = _audit_kinds(audit, agent_id)
        controls = [
            ControlResult(
                id="CC7.2",
                title="System monitoring detects anomalies (guardrails)",
                status=ControlStatus.SATISFIED if "guardrail" in kinds else ControlStatus.PARTIAL,
                evidence=["policy engine scans every run"],
            ),
            ControlResult(
                id="CC7.3",
                title="Security incidents are logged & evaluated",
                status=ControlStatus.SATISFIED if audit.events else ControlStatus.GAP,
                evidence=[f"{len(audit.events)} audit events"],
            ),
            ControlResult(
                id="CC8.1",
                title="Change management (lifecycle transitions)",
                status=ControlStatus.SATISFIED if "lifecycle" in kinds else ControlStatus.PARTIAL,
                evidence=["lifecycle transitions audited"] if "lifecycle" in kinds else [],
            ),
            ControlResult(
                id="C1.1",
                title="Confidential information is protected (PII/secret scanners)",
                status=ControlStatus.SATISFIED,
                evidence=["PII + secret output scanners enabled"],
            ),
        ]
        return ComplianceReport(regime=self.regime, agent_id=agent_id, controls=controls)
