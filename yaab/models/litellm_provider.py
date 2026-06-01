"""The universal model layer: a thin :class:`ModelProvider` over LiteLLM.

LiteLLM gives one OpenAI-compatible interface to 100+ providers and thousands
of models. This wrapper adds: structured-output schemas, ordered fallback
chains, retries with backoff, and per-call cost tracking — all surfaced
through the same :class:`ModelResponse` the rest of YAAB consumes.

``litellm`` is an optional dependency; it is imported lazily so the SDK (and
``TestModel``) work with no extra install.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import ModelError
from ..types import Message, ToolCall, Usage
from .base import ModelResponse, StreamChunk


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract a provider-suggested retry delay (seconds) from an error.

    Honors, in order: a ``retry_after`` attribute, a ``Retry-After`` response
    header, or a "try again in N seconds" phrase in the message. Returns ``None``
    when no hint is present (caller falls back to its backoff schedule).
    """
    val = getattr(exc, "retry_after", None)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    headers = getattr(exc, "response_headers", None) or getattr(exc, "headers", None)
    if isinstance(headers, dict):
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            if key in headers:
                try:
                    return float(headers[key])
                except (TypeError, ValueError):
                    pass
    import re

    m = re.search(r"(?:try again|retry) in (\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _require_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ModelError(
            "litellm is not installed. Install the model layer with "
            "`pip install 'yaab[litellm]'`, or use TestModel for offline runs."
        ) from exc
    return litellm


class LiteLLMModel:
    """Provider-agnostic model over LiteLLM's unified interface."""

    def __init__(
        self,
        model: str,
        *,
        fallbacks: list[str] | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        temperature: float | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        track_cost: bool = True,
        **default_params: Any,
    ) -> None:
        self.name = model
        self.model = model
        self.fallbacks = fallbacks or []
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.track_cost = track_cost
        self.default_params = default_params

    def _params(self, **overrides: Any) -> dict[str, Any]:
        params = dict(self.default_params)
        if self.temperature is not None:
            params.setdefault("temperature", self.temperature)
        if self.api_base:
            params.setdefault("api_base", self.api_base)
        if self.api_key:
            params.setdefault("api_key", self.api_key)
        params.update(overrides)
        return params

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        litellm = _require_litellm()
        payload = [m.to_provider_dict() for m in messages]
        extra: dict[str, Any] = self._params(**kwargs)
        if tools:
            extra["tools"] = tools
            if tool_choice is not None:
                extra["tool_choice"] = tool_choice
        if output_schema is not None:
            # LiteLLM normalizes structured outputs across providers.
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": output_schema, "strict": True},
            }

        # Try the primary model, then each fallback, with backoff retries.
        candidates = [self.model, *self.fallbacks]
        last_error: Exception | None = None
        for candidate in candidates:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await litellm.acompletion(model=candidate, messages=payload, **extra)
                    return self._normalize(resp, candidate, litellm)
                except Exception as exc:  # noqa: BLE001 - normalize provider errors
                    last_error = exc
                    if attempt < self.max_retries:
                        # Honor the provider's Retry-After hint when present, else
                        # fall back to exponential backoff.
                        retry_after = _retry_after_seconds(exc)
                        delay = (
                            retry_after
                            if retry_after is not None
                            else self.retry_base_delay * (2**attempt)
                        )
                        await asyncio.sleep(delay)
        raise ModelError(f"all model candidates failed: {last_error}") from last_error

    def _normalize(self, resp: Any, model: str, litellm: Any) -> ModelResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = Usage(requests=1)
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            usage.input_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
            usage.output_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
            usage.total_tokens = getattr(raw_usage, "total_tokens", 0) or 0
            # Cached prompt tokens: OpenAI exposes prompt_tokens_details.cached_tokens;
            # Anthropic uses cache_read_input_tokens. Capture whichever is present.
            details = getattr(raw_usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details is not None else None
            if cached is None:
                cached = getattr(raw_usage, "cache_read_input_tokens", None)
            usage.cached_input_tokens = int(cached or 0)
        if self.track_cost:
            try:
                usage.cost_usd = float(litellm.completion_cost(completion_response=resp) or 0.0)
            except Exception:  # noqa: BLE001 - cost is best-effort
                usage.cost_usd = 0.0

        # Capture a reasoning/thinking trace when the provider exposes one.
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

        return ModelResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=model,
            reasoning=reasoning,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        litellm = _require_litellm()
        payload = [m.to_provider_dict() for m in messages]
        extra = self._params(**kwargs)
        if tools:
            extra["tools"] = tools
            if tool_choice is not None:
                extra["tool_choice"] = tool_choice
        response = await litellm.acompletion(
            model=self.model, messages=payload, stream=True, **extra
        )
        # Accumulate streamed tool-call fragments (id/name/args arrive in pieces,
        # keyed by index) so we can emit assembled ToolCalls at the end.
        pending: dict[int, dict[str, Any]] = {}
        async for chunk in response:
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield StreamChunk(delta=text)
            for tcd in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tcd, "index", 0) or 0
                slot = pending.setdefault(idx, {"id": None, "name": "", "args": ""})
                if getattr(tcd, "id", None):
                    slot["id"] = tcd.id
                fn = getattr(tcd, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
        for idx in sorted(pending):
            slot = pending[idx]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["args"] or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tc_kwargs: dict[str, Any] = {"name": slot["name"], "arguments": args}
            if slot["id"]:
                tc_kwargs["id"] = slot["id"]
            yield StreamChunk(tool_call=ToolCall(**tc_kwargs))
        yield StreamChunk(done=True)
