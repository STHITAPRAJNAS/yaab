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


#: The ephemeral cache-control marker Anthropic (via litellm) understands. A
#: breakpoint caches the entire prefix up to and including the marked block.
_EPHEMERAL = {"type": "ephemeral"}


def _supports_anthropic_cache(model: str) -> bool:
    """Whether ``model`` accepts Anthropic-style ``cache_control`` breakpoints.

    Keyed off the model name so it works across the routes litellm exposes a
    Claude model through (``anthropic/``, ``bedrock/...claude...``,
    ``vertex_ai/claude-...``), without importing provider tables.
    """
    name = model.lower()
    return "anthropic" in name or "claude" in name


def _cache_last_system_block(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``payload`` with a cache breakpoint on the system prompt.

    The system message's string content is rewritten into Anthropic's
    list-of-blocks form and the LAST block carries ``cache_control`` so the
    whole (typically large, stable) system prompt is cached. Messages are copied
    shallowly so the caller's :class:`Message`-derived dicts are never mutated.
    If there is no system message the payload is returned unchanged.
    """
    out: list[dict[str, Any]] = []
    patched = False
    for msg in payload:
        if not patched and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Already block form (e.g. multimodal): copy so we can mark a block.
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
            else:
                out.append(msg)
                continue
            if blocks:
                blocks[-1] = {**blocks[-1], "cache_control": _EPHEMERAL}
                out.append({**msg, "content": blocks})
                patched = True
                continue
        out.append(msg)
    return out


def _cache_last_tool(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``tools`` with a cache breakpoint on the last definition.

    One breakpoint on the final tool caches the entire tool block, which is the
    stable, expensive-to-resend prefix. The input list and its dicts are not
    mutated.
    """
    if not tools:
        return tools
    out = list(tools)
    out[-1] = {**out[-1], "cache_control": _EPHEMERAL}
    return out


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
        cache_system_prompt: bool = False,
        cache_tools: bool = False,
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
        #: Write-side prompt/context caching. When set, cache breakpoints are
        #: injected for providers that support them (Anthropic ``cache_control``
        #: blocks today) so the heavy, stable prefix — the system prompt and/or
        #: tool definitions — is billed at the cheaper cached rate on reuse.
        #: Providers without explicit cache directives (OpenAI, Gemini implicit
        #: caching) are left untouched. This is the WRITE counterpart to the
        #: cached-token READ accounting in :meth:`_normalize`.
        self.cache_system_prompt = cache_system_prompt
        self.cache_tools = cache_tools
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

    def _apply_cache(
        self, candidate: str, payload: list[dict[str, Any]], extra: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Inject provider-specific cache directives for ``candidate``.

        Returns the (possibly rewritten) messages payload; tool-cache markers are
        mutated into ``extra["tools"]`` in place of the original list. Applied
        per-candidate because a fallback chain can span providers — only the
        Anthropic candidates get ``cache_control`` blocks.

        Gemini needs no marker here: a user-supplied ``cached_content`` already
        flows through ``extra`` (it is a normal litellm kwarg), and otherwise
        Gemini's implicit caching applies automatically. Unsupported providers
        (OpenAI, etc.) are left untouched.
        """
        if not _supports_anthropic_cache(candidate):
            return payload
        if self.cache_system_prompt:
            payload = _cache_last_system_block(payload)
        if self.cache_tools and extra.get("tools"):
            extra["tools"] = _cache_last_tool(extra["tools"])
        return payload

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
            # Cache directives are provider-specific, so build them per-candidate
            # from the base payload/extra (a fallback chain can cross providers).
            cand_extra = dict(extra)
            cand_payload = self._apply_cache(candidate, payload, cand_extra)
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await litellm.acompletion(
                        model=candidate, messages=cand_payload, **cand_extra
                    )
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
        # Same write-side cache injection as complete(), so both paths benefit.
        payload = self._apply_cache(self.model, payload, extra)
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
