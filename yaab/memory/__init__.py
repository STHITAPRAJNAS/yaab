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
from typing import Callable, Optional, Protocol, runtime_checkable

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

    async def add(self, text: str, *, metadata: Optional[dict] = None) -> MemoryRecord:
        ...

    async def search(self, query: str, *, k: int = 5) -> list[tuple[MemoryRecord, float]]:
        ...


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


class InMemoryVectorMemory:
    """A simple in-process vector store over :func:`yaab._core.top_k`."""

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or hashing_embedder()
        self._records: list[MemoryRecord] = []

    async def add(self, text: str, *, metadata: Optional[dict] = None) -> MemoryRecord:
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
    "InMemoryVectorMemory",
]
