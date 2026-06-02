"""Startup guardrail that flags non-durable backends before they lose data.

A backend that keeps its state in process memory is correct for a single
process, but the moment a second replica runs behind a load balancer, each
replica has its own private copy and they never see one another's data — runs,
sessions, artifacts, and approvals silently vanish from whichever replica didn't
handle the request.

:func:`warn_if_ephemeral` makes that failure mode loud at boot instead of
mysterious in production: when more than one replica is configured it emits a
``RuntimeWarning`` naming exactly which backends are in-memory and therefore
unsafe to run across replicas. A single replica (the default) stays silent, so
existing single-process setups are unchanged.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

# Class names that keep all state in process memory. Matching on the class name
# keeps this check dependency-free and tolerant of where a backend is defined.
_EPHEMERAL_CLASS_NAMES = frozenset(
    {
        "InMemorySessionService",
        "InMemoryArtifactService",
        "InMemoryRunStore",
        "InMemoryApprovalStore",
        "InMemoryTraceStore",
        "InMemoryAuditSink",
        "InMemoryRegistry",
        "MemorySaver",
        "RateLimiter",
    }
)


def _is_ephemeral(backend: Any) -> bool:
    """True if ``backend`` keeps its state only in process memory."""
    if backend is None:
        return False
    name = type(backend).__name__
    if name in _EPHEMERAL_CLASS_NAMES:
        return True
    # Defensive fallback: treat anything whose class name starts with "InMemory"
    # as ephemeral, so new in-memory backends are covered automatically.
    return name.startswith("InMemory")


def _resolve_replicas(replicas: int | None) -> int:
    """Effective replica count, falling back to the ``YAAB_REPLICAS`` env var."""
    if replicas is not None:
        return replicas
    raw = os.environ.get("YAAB_REPLICAS")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return 1
    return 1


def warn_if_ephemeral(*, replicas: int | None = None, **backends: Any) -> list[str]:
    """Warn loudly if any configured backend cannot run across replicas.

    Args:
        replicas: Number of replicas this deployment runs. When ``None`` it is
            read from the ``YAAB_REPLICAS`` env var (default 1).
        **backends: Named backend instances to inspect, e.g.
            ``session_service=...``, ``artifact_service=...``, ``run_store=...``.
            ``None`` values are ignored.

    Returns:
        The list of backend argument names found to be ephemeral (empty when the
        configuration is safe or the check did not run).

    A ``RuntimeWarning`` is emitted, naming each ephemeral backend and its class,
    when more than one replica is configured or strict-durability mode is on
    (env ``YAAB_STRICT_DURABILITY=1``). It is silent for a single replica unless
    strict mode is set.
    """
    strict = os.environ.get("YAAB_STRICT_DURABILITY", "").strip() not in ("", "0", "false")
    effective_replicas = _resolve_replicas(replicas)
    should_check = strict or effective_replicas > 1
    if not should_check:
        return []

    ephemeral: list[str] = []
    details: list[str] = []
    for arg_name, backend in backends.items():
        if _is_ephemeral(backend):
            ephemeral.append(arg_name)
            details.append(f"{arg_name}={type(backend).__name__}")

    if not ephemeral:
        return []

    reason = (
        "strict durability is enabled (YAAB_STRICT_DURABILITY=1)"
        if strict and effective_replicas <= 1
        else f"this deployment runs {effective_replicas} replicas"
    )
    message = (
        f"In-memory backends will lose data across replicas: {', '.join(details)}. "
        f"Because {reason}, swap each to a durable backend "
        f"(SQLite for single-node, Postgres/Redis for multi-replica) so state is "
        f"shared across processes. To intentionally silence this for a single "
        f"replica, unset YAAB_STRICT_DURABILITY and keep YAAB_REPLICAS at 1."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return ephemeral


__all__ = ["warn_if_ephemeral"]
