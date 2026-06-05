"""Governance, registry & compliance: registry, lifecycle, policy, audit, and compliance."""

from __future__ import annotations

from . import approvals_decide as approvals
from .approval import ToolApprovalPlugin
from .approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStore,
    InMemoryApprovalStore,
    PostgresApprovalStore,
    RedisApprovalStore,
    SQLiteApprovalStore,
)
from .approvals_decide import Decision as ReviewDecision
from .approvals_decide import (
    DecisionValidationError,
    ResumeBundle,
)
from .audit import AuditEvent, AuditKind, AuditLog, AuditSink, SQLiteAuditSink
from .authorization import (
    CallableAuthorizer,
    Decision,
    IdempotencyPlugin,
    RBACAuthorizer,
    ToolAuthorizationPlugin,
    ToolAuthorizer,
)
from .eval import (
    Case,
    Contains,
    Dataset,
    ExactMatch,
    Experiment,
    ExperimentResult,
    FunctionEvaluator,
    JSONMatch,
    Levenshtein,
    LLMJudge,
    NumericTolerance,
    Regex,
    ToolTrajectoryMatch,
)
from .evalset import EvalCase, EvalSet

# Side-effect import: registers the built-in scanners + industry adapters
# (Presidio / LLM-Guard / NeMo) in the component registry under "guardrail".
# Cheap — the heavy third-party deps are imported lazily inside each adapter.
from .guardrails import (  # noqa: E402
    LLMGuardScanner,
    NeMoGuardrailsScanner,
    PresidioPIIScanner,
)
from .lifecycle import EvidenceArtifact, LifecycleManager, LifecycleState
from .monitor import DriftMonitor, DriftReport, TrustReport, TrustScorer
from .policy import (
    Action,
    GuardrailResult,
    GuardrailScanner,
    PIIScanner,
    PolicyEngine,
    PromptInjectionScanner,
    SecretScanner,
    Stage,
    SystemPromptLeakScanner,
    TopicScanner,
)
from .registry import (
    AgentCard,
    AgentRegistry,
    ApprovalStatus,
    DecisionAuthority,
    EUActCategory,
    RemoteRegistryBackend,
    RiskTier,
    SQLiteRegistryBackend,
)
from .service import GovernanceMode, GovernanceService
from .simulation import (
    SimulationEvaluator,
    SimulationResult,
    UserSimulator,
    simulate,
    simulate_evalset,
)

__all__ = [
    # service
    "GovernanceService",
    "GovernanceMode",
    # registry
    "AgentRegistry",
    "AgentCard",
    "RiskTier",
    "EUActCategory",
    "ApprovalStatus",
    "DecisionAuthority",
    "SQLiteRegistryBackend",
    "RemoteRegistryBackend",
    # lifecycle
    "LifecycleManager",
    "LifecycleState",
    "EvidenceArtifact",
    # policy
    "PolicyEngine",
    "GuardrailScanner",
    "GuardrailResult",
    "Action",
    "Stage",
    "PromptInjectionScanner",
    "PIIScanner",
    "SecretScanner",
    "TopicScanner",
    "SystemPromptLeakScanner",
    # guardrail adapters (industry standards)
    "PresidioPIIScanner",
    "LLMGuardScanner",
    "NeMoGuardrailsScanner",
    # authorization & idempotency
    "ToolAuthorizationPlugin",
    "ToolAuthorizer",
    "RBACAuthorizer",
    "CallableAuthorizer",
    "Decision",
    "IdempotencyPlugin",
    "ToolApprovalPlugin",
    # out-of-band human sign-off (durable approval store + backends)
    "ApprovalStore",
    "ApprovalRequest",
    "ApprovalDecision",
    "InMemoryApprovalStore",
    "SQLiteApprovalStore",
    "PostgresApprovalStore",
    "RedisApprovalStore",
    # the unified human decision surface (pause -> decide -> resume)
    "approvals",
    "ReviewDecision",
    "ResumeBundle",
    "DecisionValidationError",
    # audit
    "AuditLog",
    "AuditEvent",
    "AuditKind",
    "AuditSink",
    "SQLiteAuditSink",
    # eval
    "Dataset",
    "Case",
    "Experiment",
    "ExperimentResult",
    "ExactMatch",
    "Contains",
    "FunctionEvaluator",
    "Regex",
    "JSONMatch",
    "NumericTolerance",
    "Levenshtein",
    "LLMJudge",
    "ToolTrajectoryMatch",
    # evalset (portable .evalset.json format)
    "EvalSet",
    "EvalCase",
    # user simulation (multi-turn agent eval)
    "UserSimulator",
    "SimulationResult",
    "SimulationEvaluator",
    "simulate",
    "simulate_evalset",
    # monitoring
    "DriftMonitor",
    "DriftReport",
    "TrustScorer",
    "TrustReport",
]
