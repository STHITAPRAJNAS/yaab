# Governance & compliance

Governance is YAAB's differentiator and a **runtime concern**, not a document.
It is opt-in by **mode**, so prototyping stays frictionless while production
enforces registration, approval, and guardrails.

```python
from yaab.governance import GovernanceService, GovernanceMode

gov = GovernanceService(mode=GovernanceMode.ENFORCING)   # off | observe | enforcing
runner = Runner(governance=gov)
```

| Mode | Behavior |
|---|---|
| `off` | governance disabled |
| `observe` | registry/policy/audit run and record, but never block |
| `enforcing` | unregistered/unapproved agents are refused; `BLOCK` guardrails stop the run |

## Agent registry & model inventory

Every agent is a versioned **Agent Card** capturing ownership, purpose, decision
authority, data lineage, risk tier, and approval status.

```python
from yaab.governance import AgentCard, RiskTier, DecisionAuthority

gov.registry.register(AgentCard(
    agent_id="kyc-bot",
    name="KYC Bot",
    business_owner="risk@bank.example",
    intended_use_case="Customer due-diligence triage",
    risk_tier=RiskTier.HIGH,
    decision_authority=DecisionAuthority.ADVISORY,
))

gov.registry.inventory()   # the SR 11-7 / EU AI Act model inventory
```

In enforcing mode, link the agent with `registry_id="kyc-bot"`; the runner
refuses to run it until it is registered **and** approved.

## Lifecycle (model-risk FSM)

```python
from yaab.governance import LifecycleState, EvidenceArtifact

gov.lifecycle.transition("kyc-bot", LifecycleState.IN_DEVELOPMENT,
    evidence=[EvidenceArtifact(kind="development_docs"),
              EvidenceArtifact(kind="conceptual_soundness")])
gov.lifecycle.transition("kyc-bot", LifecycleState.IN_VALIDATION,
    evidence=[EvidenceArtifact(kind="validation_plan")])
gov.lifecycle.transition("kyc-bot", LifecycleState.APPROVED,
    evidence=[EvidenceArtifact(kind="validation_report"),
              EvidenceArtifact(kind="effective_challenge_signoff")])
```

States: `DRAFT → IN_DEVELOPMENT → IN_VALIDATION → APPROVED → DEPLOYED →
MONITORED → DECOMMISSIONED` (+ `SUSPENDED`/`REJECTED`). Each transition is
evidence-gated and audited; illegal transitions raise `LifecycleError`.

## Guardrails (defense in depth)

The policy engine runs input scanners (prompt-injection, PII, secrets, banned
topics) and output scanners (secret/PII leakage, system-prompt leak). Decisions
are `allow` / `redact` / `flag` / `block`, and every one is audited.

```python
from yaab.governance import PolicyEngine, PIIScanner, PromptInjectionScanner, TopicScanner

gov.policy = PolicyEngine([
    PromptInjectionScanner(),
    PIIScanner(),                     # redacts emails/SSNs/cards/phones
    TopicScanner(banned=["insider trading"]),
])
```

Bring your own by implementing the `GuardrailScanner` protocol (adapters for
LLM Guard / NeMo Guardrails fit here).

## Audit log & lineage

Append-only, tamper-evident (hash-chained in Rust). Every run, model call, tool
call, guard decision, and lifecycle change is recorded.

```python
gov.audit.events          # the full ledger
gov.audit.verify()        # True iff the hash chain is intact
gov.audit.for_agent("kyc-bot")
```

Use a durable sink in production:

```python
from yaab.governance import AuditLog, SQLiteAuditSink
gov = GovernanceService(audit=AuditLog(sinks=[SQLiteAuditSink("audit.db")]))
```

## Evaluation

Code-first datasets + metrics that double as optimizer metrics and drift
monitoring:

```python
from yaab.governance import Dataset, Case, Experiment, ExactMatch

ds = Dataset(name="qa", cases=[Case(name="c1", inputs="2+2?", expected="4")])
exp = Experiment(ds, [ExactMatch()])
report = await exp.run(lambda x: str(eval(x.rstrip("?"))))
print(report.mean_score, report.aggregate)
```

## Compliance mappers

Project the governance data onto a regime's controls and emit an audit-ready
report. Built-in regimes: **SR 11-7, EU AI Act, NIST AI RMF, ISO/IEC 42001,
SOC 2**.

```python
from yaab.governance.compliance import get_mapper

report = get_mapper("eu_ai_act").map(gov.registry, gov.audit, "kyc-bot")
print(report.coverage, len(report.gaps))
print(report.to_markdown())
```

Or from the CLI:

```bash
yaab compliance report sr_11_7 --db registry.db
```

> Mappers produce **evidence, not legal sign-off**. Effective challenge and
> conformity assessment still require qualified human reviewers — YAAB produces
> the evidence; humans attest to it.

Add a regime by implementing the `ComplianceMapper` protocol and registering it
under the `yaab.compliance` entry point — no core change required.
