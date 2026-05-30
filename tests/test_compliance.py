"""Tests for compliance mappers and reports."""

from __future__ import annotations

import pytest

from yaab.governance import AgentCard, AuditLog, EUActCategory, RiskTier
from yaab.governance.audit import AuditKind
from yaab.governance.compliance import available_mappers, get_mapper
from yaab.governance.registry import AgentRegistry


@pytest.fixture
def populated():
    registry = AgentRegistry()
    registry.register(
        AgentCard(
            agent_id="a1",
            name="A1",
            risk_tier=RiskTier.HIGH,
            eu_act_category=EUActCategory.HIGH_RISK,
            intended_use_case="credit scoring",
        )
    )
    audit = AuditLog()
    audit.record(AuditKind.RUN_START, agent_id="a1")
    audit.record(AuditKind.MODEL_CALL, agent_id="a1")
    audit.record(AuditKind.LIFECYCLE, agent_id="a1", to_state="APPROVED")
    return registry, audit


def test_all_regimes_available():
    names = set(available_mappers())
    assert {"sr_11_7", "eu_ai_act", "nist_ai_rmf", "iso_42001", "soc2"} <= names


@pytest.mark.parametrize(
    "regime", ["sr_11_7", "eu_ai_act", "nist_ai_rmf", "iso_42001", "soc2"]
)
def test_mapper_produces_report(populated, regime):
    registry, audit = populated
    mapper = get_mapper(regime)
    report = mapper.map(registry, audit, "a1")
    assert report.regime == regime
    assert len(report.controls) > 0
    assert 0.0 <= report.coverage <= 1.0
    md = report.to_markdown()
    assert "Compliance Report" in md


def test_sr117_flags_audit_integrity(populated):
    registry, audit = populated
    report = get_mapper("sr_11_7").map(registry, audit, "a1")
    audit_control = next(c for c in report.controls if c.id == "SR11-7.3b")
    assert audit_control.status.value == "satisfied"  # chain is intact
