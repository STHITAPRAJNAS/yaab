"""Record/replay model wrapper — VCR-style cassettes for deterministic tests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..exceptions import CassetteMiss
from ..types import Message

_FORMAT_VERSION = 1

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


class _Cassette:
    """The on-disk JSON cassette: an ordered list of interactions plus a
    per-key replay cursor."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.interactions: list[dict[str, Any]] = []
        self._cursor: dict[str, int] = defaultdict(int)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if doc.get("version") != _FORMAT_VERSION:
                raise CassetteMiss(
                    f"cassette {self.path} has unsupported version {doc.get('version')!r}"
                )
            self.interactions = list(doc["interactions"])
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise CassetteMiss(f"cassette {self.path} is unreadable: {exc}") from exc

    def next(self, key: str) -> dict[str, Any] | None:
        """Return the next unused interaction for ``key`` (in order), or None."""
        idx = self._cursor[key]
        matches = [it for it in self.interactions if it["key"] == key]
        if idx >= len(matches):
            return None
        self._cursor[key] += 1
        return matches[idx]

    def append(
        self,
        key: str,
        request: dict[str, Any],
        *,
        response: dict[str, Any] | None,
        stream: list[dict[str, Any]] | None,
    ) -> None:
        self.interactions.append(
            {"key": key, "request": request, "response": response, "stream": stream}
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"version": _FORMAT_VERSION, "interactions": self.interactions}
        self.path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
