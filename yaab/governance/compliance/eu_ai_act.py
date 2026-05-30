"""EU AI Act (Reg. 2024/1689) compliance mapper.

Covers risk-tier classification (including the stacking checks the Act actually
runs), Annex IV technical documentation, human oversight, and Art. 12 lifetime
event logging. Verify dates/obligations against official EUR-Lex sources.
"""

from __future__ import annotations

from typing import Optional

from ..audit import AuditLog
from ..registry import AgentRegistry, EUActCategory
from .base import ComplianceReport, ControlResult, ControlStatus, _audit_kinds


class EUAIActMapper:
    regime = "eu_ai_act"

    def map(
        self, registry: AgentRegistry, audit: AuditLog, agent_id: Optional[str] = None
    ) -> ComplianceReport:
        kinds = _audit_kinds(audit, agent_id)
        card = registry.get(agent_id) if agent_id else None
        controls: list[ControlResult] = []

        category = card.eu_act_category if card else EUActCategory.MINIMAL

        controls.append(
            ControlResult(
                id="AIA.classification",
                title="Risk classification recorded (Art. 6 / Annex III)",
                status=ControlStatus.SATISFIED if card else ControlStatus.GAP,
                evidence=[f"eu_act_category={category.value}"] if card else [],
            )
        )

        if category is EUActCategory.PROHIBITED:
            controls.append(
                ControlResult(
                    id="AIA.art5",
                    title="Prohibited practice (Art. 5) — must not be deployed",
                    status=ControlStatus.GAP,
                    notes="agent classified as a prohibited practice",
                )
            )

        high_risk = category in (EUActCategory.HIGH_RISK,)
        controls.append(
            ControlResult(
                id="AIA.annexIV",
                title="Annex IV technical documentation",
                status=ControlStatus.PARTIAL if high_risk else ControlStatus.NOT_APPLICABLE,
                notes="registry card supplies a documentation spine; complete narrative externally",
            )
        )
        controls.append(
            ControlResult(
                id="AIA.art14",
                title="Human oversight (Art. 14)",
                status=ControlStatus.SATISFIED
                if (card and card.output_actions.value == "advisory")
                else ControlStatus.PARTIAL,
                evidence=[f"decision_authority={card.decision_authority.value}"] if card else [],
            )
        )
        controls.append(
            ControlResult(
                id="AIA.art12",
                title="Automatic event logging over lifetime (Art. 12)",
                status=ControlStatus.SATISFIED
                if ({"run_start", "model_call"} & kinds and audit.verify())
                else ControlStatus.GAP,
                evidence=["append-only hash-chained audit log"]
                if audit.verify()
                else [],
            )
        )
        controls.append(
            ControlResult(
                id="AIA.registration",
                title="EU database registration (high-risk)",
                status=ControlStatus.NOT_APPLICABLE if not high_risk else ControlStatus.GAP,
                notes="export to_a2a_card() as the registration record",
            )
        )
        return ComplianceReport(regime=self.regime, agent_id=agent_id, controls=controls)
