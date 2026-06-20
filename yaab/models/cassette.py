"""Record/replay model wrapper — VCR-style cassettes for deterministic tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..types import Message

# kwargs that must never be written to a cassette or affect the key.
_SECRET_KEYS = {"api_key", "authorization", "auth", "key", "token"}


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in params.items()
        if k.lower() not in _SECRET_KEYS and not k.lower().endswith("_key")
    }


def _canonical_request(
    messages: list[Message],
    *,
    tools: list[dict[str, Any]] | None,
    output_schema: dict[str, Any] | None,
    tool_choice: Any | None,
    model: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": [m.to_provider_dict() for m in messages],
        "tools": tools,
        "output_schema": output_schema,
        "tool_choice": tool_choice,
        "model": model,
        "params": _redact(params),
    }


def _request_key(canonical: dict[str, Any]) -> str:
    blob = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
