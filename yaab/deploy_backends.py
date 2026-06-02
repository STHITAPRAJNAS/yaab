"""One call that makes a deployment durable and replica-safe.

Every stateful concern in the SDK — sessions, artifacts, the run store, the
approval store, the per-run trace, the checkpointer, the audit sink, the agent
registry, and the rate-limit budget — defaults to an in-process backend. That is
perfect for a single process and silently wrong behind a load balancer: each
replica keeps its own copy, so runs vanish on restart, an approval queued on one
replica is invisible to another, and a ``rate=10`` budget becomes ``10 x
replicas``.

:func:`durable_backends` removes the footgun. Give it one database URL (and,
optionally, a Redis URL for the shared rate-limit budget) and it returns a
:class:`DurableBackends` struct holding a consistent set of backends all pointed
at the same place. Splat it into a :class:`~yaab.Runner` and the server and the
whole deployment becomes safe to run at any number of replicas::

    from yaab import Runner, durable_backends
    from yaab.serve import serve

    backends = durable_backends(dsn="postgresql://user:pw@db/app")
    runner = Runner(**backends.runner_kwargs())
    serve(agent, **backends.serve_kwargs())

With no ``dsn`` the struct is process-local — the same zero-config default you
already have for dev and tests, but reachable through one consistent entry point.

Supported DSN forms:

* ``None`` — in-memory backends (single process; dev/test default).
* ``sqlite://<path>`` or ``sqlite:///<path>`` — durable on a single node.
* ``postgresql://...`` / ``postgres://...`` — true multi-replica HA (lazy
  ``psycopg`` import; only needed when a Postgres DSN is passed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DurableBackends", "durable_backends"]


@dataclass
class DurableBackends:
    """A coherent set of shared backends built from one (or two) URLs.

    Every field is something a :class:`~yaab.Runner` or the server accepts, all
    pointed at the same store, so a deployment is made durable in one call rather
    than by wiring nine backends by hand. Hand the relevant slice to each
    consumer with :meth:`runner_kwargs` and :meth:`serve_kwargs`.
    """

    #: The database URL these backends were built from (``None`` for in-memory).
    dsn: str | None = None
    #: The Redis URL backing the shared rate-limit budget, if one was given.
    redis_url: str | None = None

    # Conversation history + structured KV state.
    session_service: Any = None
    # Binary/file artifacts produced or consumed by tools.
    artifact_service: Any = None
    # Cross-process run lifecycle (the queue + cancel + lease source of truth).
    run_store: Any = None
    # Out-of-band human sign-off records.
    approval_store: Any = None
    # Per-run trace/event history for replay and the debugger.
    trace_store: Any = None
    # Makes background runs fault-tolerant (resume from the last step).
    run_checkpointer: Any = None
    # Hash-chained compliance ledger.
    audit_sink: Any = None
    # Agent registry / governance metadata.
    registry_backend: Any = None
    # Shared rate-limit budget across replicas (only when a Redis URL is given).
    rate_limiter: Any = None

    def runner_kwargs(self) -> dict[str, Any]:
        """The subset of fields a :class:`~yaab.Runner` accepts, ready to splat.

        ``Runner(**backends.runner_kwargs())`` wires the session, artifact,
        checkpoint, and trace backends in one line.
        """
        return {
            "session_service": self.session_service,
            "artifact_service": self.artifact_service,
            "run_checkpointer": self.run_checkpointer,
            "trace_store": self.trace_store,
        }

    def serve_kwargs(self) -> dict[str, Any]:
        """The subset of fields the server app accepts, ready to splat.

        ``serve(agent, **backends.serve_kwargs())`` wires the durable run queue,
        approval endpoints, trace history, and fault-tolerant background runs.
        """
        return {
            "run_store": self.run_store,
            "approval_store": self.approval_store,
            "trace_store": self.trace_store,
            "run_checkpointer": self.run_checkpointer,
        }


def _sqlite_path(dsn: str) -> str:
    """Extract the file path from a ``sqlite://`` / ``sqlite:///`` DSN.

    ``sqlite://relative.db`` -> ``relative.db``; ``sqlite:///abs/path.db`` ->
    ``/abs/path.db`` (the standard three-slash absolute form, where the host is
    empty and the path follows). A Windows ``sqlite://C:\\...`` form is returned
    as-is.
    """
    rest = dsn[len("sqlite://") :]
    # ``sqlite:///abs`` -> the third slash starts an absolute POSIX path; the
    # ``sqlite://path`` form has no leading slash. Either way ``rest`` is already
    # the path the SQLite driver expects.
    return rest


def _is_sqlite(dsn: str) -> bool:
    return dsn.startswith("sqlite://")


def _is_postgres(dsn: str) -> bool:
    return dsn.startswith(("postgresql://", "postgres://"))


def _in_memory_backends() -> DurableBackends:
    """Process-local backends — the dev/test default behind one entry point."""
    from .artifacts import InMemoryArtifactService
    from .governance.approvals import InMemoryApprovalStore
    from .governance.audit import InMemoryAuditSink
    from .governance.registry import InMemoryRegistryBackend
    from .graph.checkpoint import MemorySaver
    from .runs.memory import InMemoryRunStore
    from .runs.trace import InMemoryTraceStore
    from .sessions.memory import InMemorySessionService

    return DurableBackends(
        dsn=None,
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        run_store=InMemoryRunStore(),
        approval_store=InMemoryApprovalStore(),
        trace_store=InMemoryTraceStore(),
        run_checkpointer=MemorySaver(),
        audit_sink=InMemoryAuditSink(),
        registry_backend=InMemoryRegistryBackend(),
        rate_limiter=None,
    )


def _sqlite_backends(dsn: str) -> DurableBackends:
    """Durable single-node backends, every one against the same SQLite file."""
    from .artifacts.sqlite import SQLiteArtifactService
    from .governance.approvals import SQLiteApprovalStore
    from .governance.audit import SQLiteAuditSink
    from .governance.registry import SQLiteRegistryBackend
    from .graph.checkpoint import SQLiteSaver
    from .runs.sqlite import SQLiteRunStore
    from .runs.trace import SQLiteTraceStore
    from .sessions.sqlite import SQLiteSessionService

    path = _sqlite_path(dsn)
    return DurableBackends(
        dsn=dsn,
        session_service=SQLiteSessionService(path),
        artifact_service=SQLiteArtifactService(path),
        run_store=SQLiteRunStore(path),
        approval_store=SQLiteApprovalStore(path),
        trace_store=SQLiteTraceStore(path),
        run_checkpointer=SQLiteSaver(path),
        audit_sink=SQLiteAuditSink(path),
        registry_backend=SQLiteRegistryBackend(path),
        rate_limiter=None,
    )


def _postgres_backends(dsn: str) -> DurableBackends:
    """True multi-replica backends, every one against the same Postgres DSN.

    The ``psycopg`` import lives inside each backend, so a Postgres DSN only
    needs the driver installed (``pip install 'yaab-sdk[postgres]'``).
    """
    from .artifacts.postgres import PostgresArtifactService
    from .governance.approvals import PostgresApprovalStore

    # The audit sink and registry currently ship SQLite/in-memory backends; the
    # audit sink stays in-memory here and the registry falls back to SQLite so
    # the struct is always complete. Callers needing a durable audit sink across
    # replicas can override the field after building.
    from .governance.audit import InMemoryAuditSink
    from .governance.registry import SQLiteRegistryBackend
    from .graph.checkpoint import PostgresSaver
    from .runs.postgres import PostgresRunStore
    from .runs.trace import PostgresTraceStore
    from .sessions.postgres import PostgresSessionService

    return DurableBackends(
        dsn=dsn,
        session_service=PostgresSessionService(dsn),
        artifact_service=PostgresArtifactService(dsn),
        run_store=PostgresRunStore(dsn),
        approval_store=PostgresApprovalStore(dsn),
        trace_store=PostgresTraceStore(dsn),
        run_checkpointer=PostgresSaver(dsn),
        audit_sink=InMemoryAuditSink(),
        registry_backend=SQLiteRegistryBackend(),
        rate_limiter=None,
    )


def durable_backends(
    *,
    dsn: str | None = None,
    redis_url: str | None = None,
    redis_client: Any = None,
    rate: int = 60,
    rate_per: float = 60.0,
    rate_bucket: str = "default",
) -> DurableBackends:
    """Build a coherent set of shared, durable backends from one or two URLs.

    Args:
        dsn: One database URL backing every stateful concern. ``None`` builds
            process-local backends (dev/test). ``sqlite://<path>`` is durable on
            a single node; ``postgresql://...`` is the multi-replica backend.
        redis_url: When given, the rate-limit budget is shared across replicas
            via Redis, so a configured ``rate`` is a single global budget rather
            than per-replica. Without it, the struct carries no rate limiter and
            callers keep the in-process default.
        redis_client: An optional pre-built Redis client (used in tests to inject
            a fake); when given, ``redis_url`` is not dialled.
        rate: Permits per window for the shared rate limiter.
        rate_per: Window length in seconds for the shared rate limiter.
        rate_bucket: Shared bucket name identifying the rate-limit budget.

    Returns:
        A :class:`DurableBackends` you splat into a :class:`~yaab.Runner`
        (:meth:`~DurableBackends.runner_kwargs`) and the server
        (:meth:`~DurableBackends.serve_kwargs`).

    Raises:
        ValueError: if ``dsn`` is a non-empty string in an unrecognized form.
    """
    if dsn is None:
        backends = _in_memory_backends()
    elif _is_sqlite(dsn):
        backends = _sqlite_backends(dsn)
    elif _is_postgres(dsn):
        backends = _postgres_backends(dsn)
    else:
        raise ValueError(
            f"unrecognized dsn {dsn!r}; expected None, 'sqlite://<path>', or 'postgresql://...'"
        )

    if redis_url is not None or redis_client is not None:
        from .models.distributed_ratelimit import RedisRateLimiter

        backends.redis_url = redis_url
        backends.rate_limiter = RedisRateLimiter(
            rate,
            rate_per,
            bucket=rate_bucket,
            url=redis_url or "redis://localhost:6379/0",
            client=redis_client,
        )

    return backends
