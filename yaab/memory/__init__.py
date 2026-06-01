"""Long-term, cross-session memory with vector retrieval.

Retrieval uses the Rust-accelerated cosine/top-k from :mod:`yaab._core` (with a
pure-Python fallback), so memory lookups stay cheap as the store grows. The
embedder is pluggable: pass any ``Callable[[str], list[float]]``. A tiny
deterministic hashing embedder ships for offline use and tests.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .. import _core

Embedder = Callable[[str], list[float]]


class MemoryRecord(BaseModel):
    """A stored memory with its embedding and arbitrary metadata."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    text: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class MemoryService(Protocol):
    """Pluggable long-term memory backend."""

    async def add(self, text: str, *, metadata: dict | None = None) -> MemoryRecord: ...

    async def search(self, query: str, *, k: int = 5) -> list[tuple[MemoryRecord, float]]: ...


def hashing_embedder(dim: int = 64) -> Embedder:
    """A deterministic, dependency-free embedder for offline use and tests.

    Hashes tokens into a fixed-width bag-of-words vector and L2-normalizes it.
    Good enough for wiring/tests; swap in a real embedding model for production.
    """

    def embed(text: str) -> list[float]:
        vec = [0.0] * dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    return embed


#: Provider key -> its standard small embedding model, for default auto-upgrade.
_EMBED_PROVIDERS = {
    "OPENAI_API_KEY": "openai/text-embedding-3-small",
    "GEMINI_API_KEY": "gemini/text-embedding-004",
    "COHERE_API_KEY": "cohere/embed-english-v3.0",
    "MISTRAL_API_KEY": "mistral/mistral-embed",
    "VOYAGE_API_KEY": "voyage/voyage-3",
}

_warned_hashing = False


def default_embedder() -> Embedder:
    """Pick a sensible default embedder.

    Auto-upgrades to a real :class:`LiteLLMEmbedder` when ``litellm`` is installed
    **and** an embedding-provider key is in the environment (explicit opt-in by
    configuration); otherwise falls back to the deterministic hashing stub and
    logs a one-time warning that semantic recall will be weak. Construction is
    cheap and offline — the real embedder only calls the provider when used.
    """
    import importlib.util
    import os

    for env_key, model in _EMBED_PROVIDERS.items():
        if os.environ.get(env_key) and importlib.util.find_spec("litellm") is not None:
            from .embedders import LiteLLMEmbedder

            return LiteLLMEmbedder(model)

    global _warned_hashing
    if not _warned_hashing:
        _warned_hashing = True
        import logging

        logging.getLogger("yaab").warning(
            "Using the deterministic hashing embedder — semantic recall will be weak. "
            "Set an embedding key (e.g. OPENAI_API_KEY) or pass "
            "embedder='openai/text-embedding-3-small' for real embeddings."
        )
    return hashing_embedder()


def resolve_embedder(embedder: Embedder | str | None = None) -> Embedder:
    """Normalize an embedder argument: ``None`` -> default, ``str`` -> a LiteLLM
    model-name shorthand, callable -> passthrough.
    """
    if embedder is None:
        return default_embedder()
    if isinstance(embedder, str):
        from .embedders import LiteLLMEmbedder

        return LiteLLMEmbedder(embedder)
    return embedder


class InMemoryVectorMemory:
    """A simple in-process vector store over :func:`yaab._core.top_k`."""

    def __init__(self, embedder: Embedder | str | None = None) -> None:
        self.embedder = resolve_embedder(embedder)
        self._records: list[MemoryRecord] = []

    async def add(self, text: str, *, metadata: dict | None = None) -> MemoryRecord:
        record = MemoryRecord(text=text, embedding=self.embedder(text), metadata=metadata or {})
        self._records.append(record)
        return record

    async def search(self, query: str, *, k: int = 5) -> list[tuple[MemoryRecord, float]]:
        if not self._records:
            return []
        q = self.embedder(query)
        matrix = [r.embedding for r in self._records]
        hits = _core.top_k(q, matrix, k)
        return [(self._records[i], score) for i, score in hits]


__all__ = [
    "MemoryRecord",
    "MemoryService",
    "Embedder",
    "hashing_embedder",
    "default_embedder",
    "resolve_embedder",
    "InMemoryVectorMemory",
]

# Register built-in embedders in the component registry (side-effect import).
from . import embedders as _embedders  # noqa: E402,F401
