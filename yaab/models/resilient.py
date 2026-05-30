"""Resilience wrappers for model providers: rate limiting + circuit breaker.

Retries and fallbacks live in :class:`LiteLLMModel`; this adds the two controls
the ecosystem asks for to protect a *failing or rate-limited* provider
(OpenAI Agents #782):

* :class:`RateLimiter` — an async token-bucket limiting requests per minute;
* :class:`CircuitBreaker` — opens after consecutive failures, fails fast for a
  cooldown, then half-opens to probe recovery.

:class:`ResilientModel` wraps any :class:`ModelProvider` with both, transparently
(same interface), so it composes with instrumentation and the runner.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Optional

from ..exceptions import ModelError
from ..types import Message
from .base import ModelProvider, ModelResponse, StreamChunk


class RateLimiter:
    """Async token-bucket: at most ``rate`` permits per ``per`` seconds."""

    def __init__(self, rate: int, per: float = 60.0) -> None:
        self.rate = rate
        self.per = per
        self._tokens = float(rate)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated
                self._tokens = min(self.rate, self._tokens + elapsed * (self.rate / self.per))
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                # Wait for the next token to accrue.
                deficit = 1 - self._tokens
                await asyncio.sleep(deficit * (self.per / self.rate))


class CircuitBreaker:
    """Open after ``threshold`` consecutive failures; cool down, then half-open."""

    def __init__(self, *, threshold: int = 5, cooldown: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def check(self) -> None:
        if self.state == "open":
            raise ModelError(
                f"circuit breaker open after {self._failures} failures; "
                f"cooling down for {self.cooldown}s"
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()


class ResilientModel:
    """Wrap a model with a rate limiter and/or circuit breaker."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if self.circuit_breaker is not None:
            self.circuit_breaker.check()
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        try:
            resp = await self.inner.complete(
                messages, tools=tools, output_schema=output_schema,
                tool_choice=tool_choice, **kwargs,
            )
        except Exception:
            if self.circuit_breaker is not None:
                self.circuit_breaker.record_failure()
            raise
        if self.circuit_breaker is not None:
            self.circuit_breaker.record_success()
        return resp

    def stream(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        return self.inner.stream(messages, tools=tools, **kwargs)


__all__ = ["ResilientModel", "RateLimiter", "CircuitBreaker"]
