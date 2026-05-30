"""OpenTelemetry-instrumented model wrapper.

Wraps any :class:`ModelProvider` and emits a span per request following the
OpenTelemetry **GenAI semantic conventions** (``gen_ai.system``,
``gen_ai.operation.name``, ``gen_ai.request.model``, token + cost attributes).
OTel is optional; without it the wrapper is a transparent pass-through.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from ..observability import genai_span
from ..types import Message
from .base import ModelProvider, ModelResponse, StreamChunk


class InstrumentedModel:
    """Decorate a model with GenAI-convention tracing."""

    def __init__(self, inner: ModelProvider, *, system: str = "yaab") -> None:
        self.inner = inner
        self.name = inner.name
        self.system = system

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        attrs = {
            "gen_ai.system": self.system,
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": self.name,
        }
        with genai_span("chat", attrs) as span:
            resp = await self.inner.complete(
                messages,
                tools=tools,
                output_schema=output_schema,
                tool_choice=tool_choice,
                **kwargs,
            )
            if span is not None:
                span.set_attribute("gen_ai.response.model", resp.model)
                span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
                span.set_attribute("gen_ai.usage.cost_usd", resp.usage.cost_usd)
                span.set_attribute("gen_ai.response.finish_reasons", [resp.finish_reason])
            return resp

    def stream(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        return self.inner.stream(messages, tools=tools, **kwargs)
