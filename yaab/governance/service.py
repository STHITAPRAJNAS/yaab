"""GovernanceService — the single entry point that wires the layer together.

Bundles the registry, lifecycle manager, policy engine, audit log, and
evaluator, and exposes a :class:`GovernanceMode` that the runner honors:

* ``OFF`` — governance disabled (frictionless prototyping);
* ``OBSERVE`` — registry/policy/audit run and record, but never block;
* ``ENFORCING`` — unregistered/unapproved agents are refused and ``BLOCK``
  guardrail decisions stop the run.

This three-mode switch is what lets one SDK serve both the developer and the
regulator.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from ..exceptions import NotRegisteredError, PolicyViolation
from .audit import AuditKind, AuditLog
from .lifecycle import LifecycleManager
from .policy import Action, GuardrailResult, PolicyEngine, Stage
from .registry import AgentRegistry


class GovernanceMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    ENFORCING = "enforcing"


class GovernanceService:
    """Facade over the governance components, parameterized by mode."""

    def __init__(
        self,
        *,
        mode: GovernanceMode = GovernanceMode.OBSERVE,
        registry: Optional[AgentRegistry] = None,
        policy: Optional[PolicyEngine] = None,
        audit: Optional[AuditLog] = None,
    ) -> None:
        self.mode = mode
        self.registry = registry or AgentRegistry()
        self.policy = policy or PolicyEngine()
        self.audit = audit or AuditLog()
        self.lifecycle = LifecycleManager(self.registry, self.audit)

    # --- registry gate -------------------------------------------------
    def check_registered(self, agent_id: Optional[str], identity: Optional[str]) -> None:
        """Enforce registration + approval before a run (enforcing mode only)."""
        if self.mode is not GovernanceMode.ENFORCING:
            return
        if not agent_id:
            raise NotRegisteredError(
                "enforcing mode requires the agent to have a registry_id"
            )
        if not self.registry.is_approved(agent_id):
            self.audit.record(
                AuditKind.REGISTRY,
                agent_id=agent_id,
                identity=identity,
                decision="refused",
                reason="not registered or not approved",
            )
            raise NotRegisteredError(
                f"agent '{agent_id}' is not registered+approved; refusing to run "
                "in enforcing mode"
            )

    # --- guardrails ----------------------------------------------------
    def scan(
        self,
        text: str,
        stage: Stage,
        *,
        agent_id: Optional[str] = None,
        identity: Optional[str] = None,
    ) -> str:
        """Run guardrails. Returns possibly-redacted text; may raise on BLOCK.

        In OBSERVE mode a BLOCK is recorded but downgraded to a flag (text
        passes through); in ENFORCING mode a BLOCK raises ``PolicyViolation``.
        """
        if self.mode is GovernanceMode.OFF:
            return text
        results = self.policy.evaluate(text, stage)
        action, new_text = PolicyEngine.decide(results)
        for r in results:
            if r.action is not Action.ALLOW:
                self.audit.record(
                    AuditKind.GUARDRAIL,
                    agent_id=agent_id,
                    identity=identity,
                    stage=stage.value,
                    scanner=r.scanner,
                    action=r.action.value,
                    reason=r.reason,
                )
        if action is Action.BLOCK and self.mode is GovernanceMode.ENFORCING:
            blocker = next((r for r in results if r.action is Action.BLOCK), None)
            raise PolicyViolation(
                blocker.reason if blocker else "blocked by policy",
                scanner=blocker.scanner if blocker else "policy",
                stage=stage.value,
            )
        return new_text if new_text else text

    def record_run_start(
        self, agent_id: Optional[str], identity: Optional[str], prompt: str
    ) -> None:
        if self.mode is GovernanceMode.OFF:
            return
        self.audit.record(
            AuditKind.RUN_START, agent_id=agent_id, identity=identity, prompt=prompt[:500]
        )

    def record_run_end(
        self, agent_id: Optional[str], identity: Optional[str], output_repr: str
    ) -> None:
        if self.mode is GovernanceMode.OFF:
            return
        self.audit.record(
            AuditKind.RUN_END, agent_id=agent_id, identity=identity, output=output_repr[:500]
        )
