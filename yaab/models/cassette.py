"""Record/replay model wrapper — VCR-style cassettes for deterministic tests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from ..exceptions import CassetteMiss
from ..types import Message
from .base import ModelProvider, ModelResponse

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
        #: The recording model's name, stamped at save time so replay (which has
        #: no inner model) reproduces the same request key.
        self.model: str | None = None
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
            self.model = doc.get("model")
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
        doc = {
            "version": _FORMAT_VERSION,
            "model": self.model,
            "interactions": self.interactions,
        }
        self.path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")


class CassetteModel:
    """Record/replay wrapper around a ``ModelProvider``.

    modes: ``record`` (always call inner, append), ``once`` (replay if present
    else call+append), ``replay`` (replay only; miss -> CassetteMiss).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        inner: ModelProvider | None = None,
        mode: Literal["once", "record", "replay"] = "once",
    ) -> None:
        if mode in ("record", "once") and inner is None:
            raise ValueError(f"mode={mode!r} requires an inner model")
        self.path = Path(path)
        self.inner = inner
        self.mode = mode
        self._cassette = _Cassette(path)
        if inner is not None:
            # Stamp the recording model so replay reproduces the same request key.
            self._cassette.model = inner.name
        # The model name used in keys: the live inner when recording, else the
        # name stamped into the cassette at record time.
        self.name = inner.name if inner is not None else (self._cassette.model or "cassette")

    def _key(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        output_schema: dict[str, Any] | None,
        tool_choice: Any | None,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        model = self.name
        canon = _canonical_request(
            messages,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            model=model,
            params=dict(kwargs),
        )
        return _request_key(canon), canon

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        key, canon = self._key(messages, tools, output_schema, tool_choice, kwargs)

        if self.mode != "record":
            hit = self._cassette.next(key)
            if hit is not None and hit.get("response") is not None:
                return ModelResponse.model_validate(hit["response"])
            if self.mode == "replay":
                raise CassetteMiss(
                    f"no recorded completion for request {key[:12]} in "
                    f"{self.path} (mode=replay)"
                )

        assert self.inner is not None  # guaranteed by __init__ for record/once
        resp = await self.inner.complete(
            messages,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
        self._cassette.append(key, canon, response=resp.model_dump(), stream=None)
        self._cassette.save()
        return resp
