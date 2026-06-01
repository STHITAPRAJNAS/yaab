"""Streaming structured output — partial typed objects as they generate.

Pydantic AI #1452 and others ask for partial structured results during
generation (e.g. render a form as fields arrive). :func:`stream_structured`
streams tokens from the model, repeatedly attempts a *tolerant* parse of the
JSON-so-far, and yields the latest partial object whenever it changes —
validated leniently against the output type so partial states are allowed.

The tolerant parser closes any open strings/objects/arrays in the buffer so an
incomplete JSON fragment still parses into the best-effort object available.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


def _strip_code_fence(s: str) -> str:
    """Strip a leading ```/```json fence (and any trailing ```), even when the
    closing fence hasn't streamed in yet.

    Many providers (Gemini, Claude, others) wrap JSON output in a Markdown code
    fence despite a JSON-only instruction. Tolerating it keeps structured-output
    streaming from yielding nothing.
    """
    if not s.startswith("```"):
        return s
    # Drop the opening fence line: ``` optionally followed by a language tag.
    newline = s.find("\n")
    s = s[newline + 1 :] if newline != -1 else s[3:]
    # Drop a trailing closing fence if present.
    fence = s.rfind("```")
    if fence != -1:
        s = s[:fence]
    return s.strip()


def parse_partial_json(buffer: str) -> Any | None:
    """Best-effort parse of a possibly-incomplete JSON string.

    Closes dangling strings/brackets and trims trailing commas so a prefix of a
    JSON document yields the object built so far. Returns ``None`` if nothing
    parseable is present yet.
    """
    s = _strip_code_fence(buffer.strip())
    if not s:
        return None
    # Fast path: already valid.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()

    repaired = s
    if in_string:
        repaired += '"'
    # Drop a trailing comma or dangling key before closing.
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    # Close any still-open containers, innermost first.
    repaired += "".join(reversed(stack))

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Try once more after stripping a dangling `"key":` with no value.
        import re

        trimmed = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", s)
        if trimmed != s:
            return parse_partial_json(trimmed)
        return None


async def stream_structured(
    agent: Any,
    prompt: str,
    *,
    output_type: type,
    deps: Any = None,
    identity: str | None = None,
) -> AsyncIterator[Any]:
    """Yield partial instances of ``output_type`` as the model streams JSON.

    Each yield is the latest parseable partial. The final yield is the complete,
    fully-validated object. Validation of partials is lenient (missing fields are
    allowed); the last value is validated strictly.
    """
    from pydantic import BaseModel, TypeAdapter

    from .types import Message, Role

    runner = agent._get_runner()
    # Build the same messages the runner would, asking for JSON.
    ctx_messages = await runner._build_messages(
        agent,
        _mk_ctx(deps, identity),
        prompt,
        original=prompt,
    )
    model_cls: type[BaseModel] | None = (
        output_type
        if isinstance(output_type, type) and issubclass(output_type, BaseModel)
        else None
    )
    schema_hint = ""
    if model_cls is not None:
        schema_hint = "\n\nRespond ONLY with a JSON object matching this schema:\n" + json.dumps(
            model_cls.model_json_schema()
        )
    ctx_messages.append(Message(role=Role.SYSTEM, content=f"Output JSON only.{schema_hint}"))

    adapter: TypeAdapter = TypeAdapter(output_type)
    buffer = ""
    last_emitted: Any = None
    async for chunk in agent.model.stream(ctx_messages):
        if not chunk.delta:
            continue
        buffer += chunk.delta
        partial = parse_partial_json(buffer)
        if partial is None or partial == last_emitted:
            continue
        last_emitted = partial
        # Lenient: build the model from whatever fields exist so far.
        try:
            if model_cls is not None and isinstance(partial, dict):
                yield model_cls.model_construct(**partial)
            else:
                yield partial
        except Exception:  # noqa: BLE001 - skip un-constructable partials
            continue

    # Final strict validation of the complete buffer.
    final = parse_partial_json(buffer)
    if final is not None:
        try:
            yield adapter.validate_python(final)
        except Exception:  # noqa: BLE001 - fall back to the last partial
            if last_emitted is not None and last_emitted != final:
                yield last_emitted


def _mk_ctx(deps: Any, identity: str | None):
    from .types import RunContext

    return RunContext(deps=deps, identity=identity)


__all__ = ["stream_structured", "parse_partial_json"]
