"""Behavioral drift detection & trust scoring (CrewAI #5155/#5789).

Production agents degrade quietly: an eval score that was 0.9 in validation
drifts to 0.6, guardrails start firing more often, error rates climb. This
module turns the existing eval + audit substrate into an ongoing monitor.

* :class:`DriftMonitor` — record a rolling series of eval scores per agent and
  flag when the recent window drops materially below a baseline.
* :class:`TrustScorer` — fold eval performance, guardrail blocks, and errors
  from the audit log into a single 0–1 trust score for an agent, with a
  human-readable breakdown.

Both read from data YAAB already produces, so they need no new instrumentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .audit import AuditKind, AuditLog


class DriftReport(BaseModel):
    agent_id: str
    baseline: float
    recent: float
    delta: float
    drifted: bool
    samples: int


class DriftMonitor:
    """Track eval scores over time and detect material regressions.

    ``baseline`` is the first ``baseline_window`` scores (e.g. validation-time);
    drift is flagged when the mean of the last ``recent_window`` scores falls
    more than ``threshold`` below the baseline mean.
    """

    def __init__(
        self,
        *,
        baseline_window: int = 5,
        recent_window: int = 5,
        threshold: float = 0.1,
    ) -> None:
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.threshold = threshold
        self._scores: dict[str, list[float]] = {}

    def record_score(self, agent_id: str, score: float) -> None:
        self._scores.setdefault(agent_id, []).append(score)

    def report(self, agent_id: str) -> DriftReport:
        scores = self._scores.get(agent_id, [])
        baseline_vals = scores[: self.baseline_window]
        recent_vals = scores[-self.recent_window :]
        baseline = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0
        recent = sum(recent_vals) / len(recent_vals) if recent_vals else 0.0
        delta = recent - baseline
        return DriftReport(
            agent_id=agent_id,
            baseline=baseline,
            recent=recent,
            delta=delta,
            drifted=bool(scores) and delta < -self.threshold,
            samples=len(scores),
        )


class TrustReport(BaseModel):
    agent_id: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    runs: int = 0
    guardrail_blocks: int = 0
    errors: int = 0


class TrustScorer:
    """Compute a 0–1 trust score for an agent from eval + audit signals.

    The score blends three components (each in [0, 1], higher is better):

    * ``performance`` — mean eval score for the agent (defaults to 1.0 if none);
    * ``safety`` — 1 minus the rate of guardrail blocks per run;
    * ``reliability`` — 1 minus the rate of errors per run.

    Weights are configurable; the breakdown is returned for transparency.
    """

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or {"performance": 0.5, "safety": 0.3, "reliability": 0.2}

    def score(
        self,
        agent_id: str,
        audit: AuditLog,
        *,
        eval_score: float | None = None,
    ) -> TrustReport:
        events = audit.for_agent(agent_id)
        runs = sum(1 for e in events if e.kind is AuditKind.RUN_START)
        blocks = sum(
            1
            for e in events
            if e.kind is AuditKind.GUARDRAIL and e.payload.get("action") in ("block", "deny")
        )
        errors = sum(1 for e in events if e.kind is AuditKind.ERROR)

        denom = max(runs, 1)
        performance = eval_score if eval_score is not None else 1.0
        safety = max(0.0, 1.0 - blocks / denom)
        reliability = max(0.0, 1.0 - errors / denom)

        components = {
            "performance": performance,
            "safety": safety,
            "reliability": reliability,
        }
        total = sum(self.weights.get(k, 0.0) * v for k, v in components.items())
        weight_sum = sum(self.weights.values()) or 1.0
        return TrustReport(
            agent_id=agent_id,
            score=total / weight_sum,
            components=components,
            runs=runs,
            guardrail_blocks=blocks,
            errors=errors,
        )


__all__ = ["DriftMonitor", "DriftReport", "TrustScorer", "TrustReport"]
