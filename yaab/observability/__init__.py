"""Observability helpers built on OpenTelemetry GenAI semantic conventions.

OTel is optional. When ``opentelemetry`` is not installed, :func:`genai_span`
yields ``None`` and instrumentation becomes a no-op, so nothing in the runtime
depends on a tracing backend being present.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:  # pragma: no cover - depends on optional extra
    from opentelemetry import trace

    _tracer = trace.get_tracer("yaab")
    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _tracer = None
    _HAS_OTEL = False

__all__ = ["genai_span", "has_otel"]


def has_otel() -> bool:
    return _HAS_OTEL


@contextmanager
def genai_span(name: str, attributes: dict[str, Any]) -> Iterator[Optional[Any]]:
    """Open a span named ``gen_ai.<name>`` with the given attributes.

    Yields the span (or ``None`` if OTel is absent) so callers can attach
    response attributes after the wrapped call returns.
    """
    if not _HAS_OTEL or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(f"gen_ai.{name}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
