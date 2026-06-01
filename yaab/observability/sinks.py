"""External audit sinks: Langfuse, Logfire, and an OTel span sink.

The :class:`~yaab.governance.audit.AuditLog` writes every event to its sinks;
these forward those events to popular observability backends. All client SDKs
are imported lazily, so installing one is opt-in. Each sink implements the
``AuditSink`` protocol (``write(event)``), so they slot in alongside the
in-memory and SQLite sinks.

    from yaab.governance import AuditLog
    from yaab.observability.sinks import LangfuseSink
    audit = AuditLog(sinks=[LangfuseSink()])
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..governance.audit import AuditEvent


class CallbackSink:
    """Forward each audit event to a user callback (the simplest custom sink)."""

    def __init__(self, callback: Callable[[AuditEvent], None]) -> None:
        self._cb = callback

    def write(self, event: AuditEvent) -> None:
        self._cb(event)


class LangfuseSink:
    """Send audit events to Langfuse as events/spans (lazy ``langfuse``)."""

    def __init__(self, *, client: Any = None, **client_kwargs: Any) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                from langfuse import Langfuse  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError("langfuse is required. `pip install langfuse`.") from exc
            self._client = Langfuse(**client_kwargs)

    def write(self, event: AuditEvent) -> None:
        try:
            self._client.create_event(
                name=f"yaab.{event.kind.value}",
                metadata={
                    "agent_id": event.agent_id,
                    "identity": event.identity,
                    "hash": event.hash,
                    **event.payload,
                },
            )
        except Exception:  # noqa: BLE001 - observability must never break a run
            pass


class LogfireSink:
    """Send audit events to Pydantic Logfire (lazy ``logfire``)."""

    def __init__(self, *, configure: bool = False, **configure_kwargs: Any) -> None:
        try:
            import logfire  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("logfire is required. `pip install logfire`.") from exc
        self._logfire = logfire
        if configure:
            logfire.configure(**configure_kwargs)

    def write(self, event: AuditEvent) -> None:
        try:
            self._logfire.info(
                "yaab.{kind}",
                kind=event.kind.value,
                agent_id=event.agent_id,
                identity=event.identity,
                **event.payload,
            )
        except Exception:  # noqa: BLE001
            pass


class OTelSpanSink:
    """Emit each audit event as a short OpenTelemetry span (lazy OTel)."""

    def __init__(self, tracer_name: str = "yaab.audit") -> None:
        from ..observability import has_otel

        if not has_otel():
            raise RuntimeError("opentelemetry is required. `pip install 'yaab-sdk[otel]'`.")
        from opentelemetry import trace  # type: ignore

        self._tracer = trace.get_tracer(tracer_name)

    def write(self, event: AuditEvent) -> None:
        try:
            with self._tracer.start_as_current_span(f"audit.{event.kind.value}") as span:
                span.set_attribute("yaab.agent_id", event.agent_id or "")
                span.set_attribute("yaab.identity", event.identity or "")
                for k, v in event.payload.items():
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(f"yaab.{k}", v)
        except Exception:  # noqa: BLE001
            pass


__all__ = ["CallbackSink", "LangfuseSink", "LogfireSink", "OTelSpanSink"]
