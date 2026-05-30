"""Governance end-to-end: register → validate → approve → run → audit → report.

Shows enforcing mode refusing an unapproved agent, the model-risk lifecycle,
guardrails, the tamper-evident audit log, and an audit-ready SR 11-7 report.
"""

from yaab import Agent, Runner
from yaab.governance import (
    AgentCard,
    DecisionAuthority,
    EvidenceArtifact,
    GovernanceMode,
    GovernanceService,
    LifecycleState,
    RiskTier,
)
from yaab.governance.compliance import get_mapper
from yaab.testing import TestModel

gov = GovernanceService(mode=GovernanceMode.ENFORCING)

# 1) Register the agent with its governance metadata.
gov.registry.register(
    AgentCard(
        agent_id="kyc-bot",
        name="KYC Bot",
        business_owner="risk@bank.example",
        intended_use_case="Customer due-diligence triage",
        risk_tier=RiskTier.HIGH,
        decision_authority=DecisionAuthority.ADVISORY,
    )
)

# 2) Walk it through the model-risk lifecycle (each step is evidence-gated).
steps = [
    (LifecycleState.IN_DEVELOPMENT, ["development_docs", "conceptual_soundness"]),
    (LifecycleState.IN_VALIDATION, ["validation_plan"]),
    (LifecycleState.APPROVED, ["validation_report", "effective_challenge_signoff"]),
    (LifecycleState.DEPLOYED, ["change_control_record"]),
]
for state, evidence in steps:
    gov.lifecycle.transition(
        "kyc-bot",
        state,
        actor="validator@bank",
        evidence=[EvidenceArtifact(kind=k) for k in evidence],
    )
print("approved:", gov.registry.is_approved("kyc-bot"))

# 3) Run under enforcing governance.
agent = Agent("KYC Bot", model=TestModel("Customer appears low-risk."), registry_id="kyc-bot")
runner = Runner(governance=gov)
result = runner.run_sync(agent, "Assess customer 12345", identity="analyst@bank")
print("output:", result.output)

# 4) The audit trail is tamper-evident.
print("audit events:", len(gov.audit.events), "| chain intact:", gov.audit.verify())

# 5) Generate an audit-ready compliance report.
report = get_mapper("sr_11_7").map(gov.registry, gov.audit, "kyc-bot")
print(f"\nSR 11-7 coverage: {report.coverage:.0%}, gaps: {len(report.gaps)}")
print(report.to_markdown())
