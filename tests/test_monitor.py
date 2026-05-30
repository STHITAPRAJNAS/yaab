"""Tests for drift detection and trust scoring (Tier 3c)."""

from __future__ import annotations

from yaab.governance import AuditLog, DriftMonitor, TrustScorer
from yaab.governance.audit import AuditKind


def test_drift_monitor_flags_regression():
    mon = DriftMonitor(baseline_window=3, recent_window=3, threshold=0.1)
    for s in [0.9, 0.92, 0.88]:  # baseline ~0.9
        mon.record_score("a", s)
    for s in [0.6, 0.58, 0.62]:  # recent ~0.6
        mon.record_score("a", s)
    report = mon.report("a")
    assert report.drifted is True
    assert report.delta < 0
    assert report.samples == 6


def test_drift_monitor_stable_not_flagged():
    mon = DriftMonitor(baseline_window=2, recent_window=2, threshold=0.1)
    for s in [0.8, 0.82, 0.79, 0.81]:
        mon.record_score("a", s)
    assert mon.report("a").drifted is False


def test_trust_scorer_perfect_when_clean():
    audit = AuditLog()
    audit.record(AuditKind.RUN_START, agent_id="a")
    audit.record(AuditKind.RUN_END, agent_id="a")
    report = TrustScorer().score("a", audit, eval_score=1.0)
    assert report.score == 1.0
    assert report.runs == 1
    assert report.guardrail_blocks == 0


def test_trust_scorer_penalizes_blocks_and_errors():
    audit = AuditLog()
    audit.record(AuditKind.RUN_START, agent_id="a")
    audit.record(AuditKind.GUARDRAIL, agent_id="a", action="block")
    audit.record(AuditKind.ERROR, agent_id="a", error="boom")
    report = TrustScorer().score("a", audit, eval_score=1.0)
    assert report.score < 1.0
    assert report.guardrail_blocks == 1
    assert report.errors == 1
    assert report.components["safety"] < 1.0
    assert report.components["reliability"] < 1.0


def test_trust_scorer_weights_sum_normalized():
    audit = AuditLog()
    audit.record(AuditKind.RUN_START, agent_id="a")
    # Custom weights still yield a score in [0, 1].
    report = TrustScorer(weights={"performance": 1.0}).score("a", audit, eval_score=0.5)
    assert 0.0 <= report.score <= 1.0
    assert report.score == 0.5
