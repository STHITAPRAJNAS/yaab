"""Compliance mapper framework.

A :class:`ComplianceMapper` projects the governance data model (registry +
lifecycle + audit + evals) onto a specific regime's controls and emits an
audit-ready :class:`ComplianceReport`. Mappers are additive plugins — adding a
regime never touches the core.

Mappers produce *evidence*, not legal sign-off: a human reviewer still attests.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..audit import AuditLog
from ..registry import AgentRegistry


class ControlStatus(str, Enum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    GAP = "gap"
    NOT_APPLICABLE = "not_applicable"


class ControlResult(BaseModel):
    id: str
    title: str
    status: ControlStatus
    evidence: list[str] = Field(default_factory=list)
    notes: str = ""


class ComplianceReport(BaseModel):
    regime: str
    agent_id: str | None = None
    generated_at: float = Field(default_factory=time.time)
    controls: list[ControlResult] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of applicable controls that are satisfied."""
        applicable = [c for c in self.controls if c.status is not ControlStatus.NOT_APPLICABLE]
        if not applicable:
            return 0.0
        satisfied = sum(1 for c in applicable if c.status is ControlStatus.SATISFIED)
        return satisfied / len(applicable)

    @property
    def gaps(self) -> list[ControlResult]:
        return [c for c in self.controls if c.status is ControlStatus.GAP]

    def to_markdown(self) -> str:
        lines = [
            f"# Compliance Report — {self.regime}",
            "",
            f"- Agent: `{self.agent_id or 'ALL'}`",
            f"- Coverage: **{self.coverage:.0%}**",
            f"- Gaps: **{len(self.gaps)}**",
            "",
            "| Control | Title | Status | Evidence |",
            "| --- | --- | --- | --- |",
        ]
        for c in self.controls:
            ev = "; ".join(c.evidence) if c.evidence else "—"
            lines.append(f"| {c.id} | {c.title} | {c.status.value} | {ev} |")
        return "\n".join(lines)


@runtime_checkable
class ComplianceMapper(Protocol):
    regime: str

    def map(
        self,
        registry: AgentRegistry,
        audit: AuditLog,
        agent_id: str | None = None,
    ) -> ComplianceReport: ...


def _audit_kinds(audit: AuditLog, agent_id: str | None) -> set[str]:
    events = audit.for_agent(agent_id) if agent_id else audit.events
    return {e.kind.value for e in events}
