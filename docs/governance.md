# Governance & compliance

Governance in YAAB is a **runtime concern**, not a document.
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

Bring your own by implementing the `GuardrailScanner` protocol.

### Industry guardrail adapters (out of the box)

Adapters for the standard engines ship in `yaab.governance.guardrails`, each
behind the same `GuardrailScanner` protocol so they drop straight into the
`PolicyEngine` or are selectable from the component registry. The heavy deps are
optional extras, imported lazily.

```python
from yaab.governance import PresidioPIIScanner, LLMGuardScanner, NeMoGuardrailsScanner

gov.policy = PolicyEngine([
    PresidioPIIScanner(),                 # pip install 'yaab-sdk[presidio]' — NER-based PII
    LLMGuardScanner(),                    # pip install 'yaab-sdk[llm-guard]' — Protect AI scanners
    NeMoGuardrailsScanner(rails=my_rails) # pip install 'yaab-sdk[nemo]'      — NVIDIA NeMo rails
])

# …or by name through the component registry:
from yaab import get_component, available_components
available_components("guardrail")   # ['llm_guard', 'nemo', 'pii', 'presidio', 'prompt_injection', ...]
pii = get_component("guardrail", "presidio")
```

Each adapter also accepts an injected engine (`PresidioPIIScanner(analyzer=…)`,
`LLMGuardScanner(input_scanners=[…])`, `NeMoGuardrailsScanner(check=…)`) for
custom configuration and offline testing.

## Tool authorization & idempotency

Authorize a tool call *before* it runs, and dedupe side-effecting calls — the
two most-requested governance seams across the ecosystem. Both are Runner
plugins, so they compose with guardrails and audit.

```python
from yaab import Runner
from yaab.governance import (
    ToolAuthorizationPlugin, RBACAuthorizer, CallableAuthorizer, IdempotencyPlugin,
)

authz = ToolAuthorizationPlugin(
    [
        RBACAuthorizer(
            deny=["delete_account"],                       # never allowed
            require_capability={"update_inventory": "write"},  # needs ctx capability
        ),
        CallableAuthorizer(lambda tool, args, ctx: args.get("amount", 0) <= 10_000),
    ],
    audit=gov.audit,
    hard=False,   # soft: deny is fed back to the model; hard=True raises PolicyViolation
)

# Don't charge/email/trade twice if the model repeats a call:
idem = IdempotencyPlugin(tools=["charge"], key_fn=lambda t, a: a["order_id"])

runner = Runner(plugins=[authz, idem])
```

A soft denial returns an error string to the model (so the agent can adapt); a
hard denial raises `PolicyViolation`. Every non-allow decision is audited. The
caller's capabilities come from `ctx.state["capabilities"]`.

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

## Drift detection & trust scoring

Production agents degrade quietly. The monitor turns the eval + audit substrate
into an ongoing health signal — no new instrumentation.

```python
from yaab.governance import DriftMonitor, TrustScorer

# Feed periodic eval scores; flag when recent performance drops below baseline.
drift = DriftMonitor(baseline_window=5, recent_window=5, threshold=0.1)
for score in nightly_eval_scores:
    drift.record_score("kyc-bot", score)
report = drift.report("kyc-bot")
if report.drifted:
    alert(f"{report.agent_id} drifted: {report.baseline:.2f} -> {report.recent:.2f}")

# Blend eval performance, guardrail blocks, and errors into one 0-1 trust score.
trust = TrustScorer().score("kyc-bot", gov.audit, eval_score=report.recent)
print(trust.score, trust.components)   # {'performance':…, 'safety':…, 'reliability':…}
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


## Central registry & custom fields

The registry is a facade over a pluggable `RegistryBackend` (`upsert` / `fetch`
/ `all`). Built-ins: `InMemoryRegistryBackend`, `SQLiteRegistryBackend`, and
`RemoteRegistryBackend` for a central/enterprise HTTP system-of-record. Point
governance at your central registry and the enforcing run-gate reads approval
status from it on every run:

```python
import httpx
from yaab import Runner
from yaab.governance import (
    AgentRegistry, RemoteRegistryBackend, GovernanceService, GovernanceMode,
)

registry = AgentRegistry(
    RemoteRegistryBackend(
        base_url="https://registry.internal/api",
        headers={"authorization": "Bearer <token>"},
    )
)
gov = GovernanceService(mode=GovernanceMode.ENFORCING, registry=registry)
runner = Runner(governance=gov)
```

The expected REST contract (override paths to fit your service):

```
PUT  {base_url}/agents/{agent_id}   body: AgentCard JSON  -> 2xx
GET  {base_url}/agents/{agent_id}   -> AgentCard JSON (404 if absent)
GET  {base_url}/agents             -> [AgentCard, ...] or {"agents": [...]}
```

### Org-specific attributes (usecase_id, blueprint, ...)

`AgentCard` carries a typed `metadata` dict for organization-specific attributes,
and sets `extra="allow"` so any additional fields your central registry uses
round-trip losslessly through JSON:

```python
from yaab.governance import AgentCard

card = AgentCard(
    agent_id="support-bot",
    name="Support Bot",
    intended_use_case="Customer support triage",
    metadata={"usecase_id": "UC-123", "blueprint": "rag-support-v2"},
    # or as top-level extra fields — both are preserved:
    cost_center="CX-7",
)
registry.register(card)

got = registry.get("support-bot")
got.metadata["usecase_id"]   # "UC-123"
got.cost_center              # "CX-7"  (extra field, preserved)
```

`metadata` also surfaces in `registry.inventory()` (the SR 11-7 / EU AI Act
model-inventory view), so your custom keys appear alongside risk tier, approval
status, and lifecycle state.
