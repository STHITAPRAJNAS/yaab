"""Observability helpers built on OpenTelemetry GenAI semantic conventions.

OTel is optional. When ``opentelemetry`` is not installed, :func:`genai_span`
yields ``None`` and instrumentation becomes a no-op, so nothing in the runtime
depends on a tracing backend being present.

Tracing is also globally controllable at runtime — the ecosystem repeatedly asks
to **disable instrumentation** (ADK #2792, Strands #1059) and to **redact PII in
traces** (Strands #1292, OpenAI #2393):

* :func:`set_tracing_enabled` / :func:`tracing_enabled` — a global on/off switch
  (also honored via the ``YAAB_DISABLE_TRACING=1`` env var);
* :func:`set_trace_redactor` — register a function that scrubs span attributes
  before they are recorded.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

try:  # pragma: no cover - depends on optional extra
    from opentelemetry import trace

    _tracer = trace.get_tracer("yaab")
    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _tracer = None
    _HAS_OTEL = False

__all__ = [
    "genai_span",
    "has_otel",
    "tracing_enabled",
    "set_tracing_enabled",
    "set_trace_redactor",
]

# Global tracing switch. Defaults on, unless disabled via env var.
_TRACING_ENABLED = os.environ.get("YAAB_DISABLE_TRACING") != "1"

# Optional redactor: (key, value) -> value, applied to every span attribute.
Redactor = Callable[[str, Any], Any]
_REDACTOR: Optional[Redactor] = None


def has_otel() -> bool:
    return _HAS_OTEL


def tracing_enabled() -> bool:
    """Whether spans are currently emitted (global switch AND OTel present)."""
    return _TRACING_ENABLED and _HAS_OTEL


def set_tracing_enabled(enabled: bool) -> None:
    """Globally enable/disable YAAB span emission at runtime."""
    global _TRACING_ENABLED
    _TRACING_ENABLED = enabled


def set_trace_redactor(redactor: Optional[Redactor]) -> None:
    """Register (or clear) a redactor applied to every span attribute value.

    The redactor receives ``(key, value)`` and returns the value to record —
    e.g. mask anything whose key looks sensitive, or run a PII scrubber over
    string values. Pass ``None`` to remove it.
    """
    global _REDACTOR
    _REDACTOR = redactor


def _apply_redactor(key: str, value: Any) -> Any:
    if _REDACTOR is None:
        return value
    try:
        return _REDACTOR(key, value)
    except Exception:  # noqa: BLE001 - a broken redactor must not break tracing
        return value


@contextmanager
def genai_span(name: str, attributes: dict[str, Any]) -> Iterator[Optional[Any]]:
    """Open a span named ``gen_ai.<name>`` with the given attributes.

    Yields the span (or ``None`` when tracing is disabled / OTel absent) so
    callers can attach response attributes after the wrapped call returns. When
    a span is live, a registered redactor scrubs every attribute on the way in,
    and :meth:`set_attribute` is wrapped so post-hoc attributes are scrubbed too.
    """
    if not tracing_enabled() or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(f"gen_ai.{name}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, _apply_redactor(key, value))
        yield _RedactingSpan(span) if _REDACTOR is not None else span


class _RedactingSpan:
    """Thin wrapper that scrubs attribute values via the active redactor."""

    __slots__ = ("_span",)

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, _apply_redactor(key, value))

    def __getattr__(self, item: str) -> Any:
        return getattr(self._span, item)
