"""Embedders for vector memory.

The default :func:`~yaab.memory.hashing_embedder` is deterministic and offline
(great for tests/wiring). :class:`LiteLLMEmbedder` produces real embeddings via
any LiteLLM-supported model (OpenAI, Cohere, Bedrock, Vertex, ...). Both are
registered in the component registry under the ``embedder`` kind so third
parties can add their own and select them by name.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import ModelError
from ..extensions import register
from . import Embedder, hashing_embedder


class LiteLLMEmbedder:
    """A callable embedder backed by ``litellm.embedding`` (any provider)."""

    def __init__(self, model: str = "openai/text-embedding-3-small", **params: Any) -> None:
        self.model = model
        self.params = params

    def __call__(self, text: str) -> list[float]:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ModelError(
                "litellm is required for LiteLLMEmbedder. `pip install 'yaab-sdk[litellm]'`."
            ) from exc
        resp = litellm.embedding(model=self.model, input=[text], **self.params)
        return list(resp["data"][0]["embedding"])


class CachingEmbedder:
    """Wrap any embedder with a content-keyed cache to avoid re-embedding.

    Re-embedding the same text (re-indexing, repeated queries) is a recurring
    cost sink across RAG stacks. This caches by exact text; pass a ``store``
    dict to share/persist the cache across instances.
    """

    def __init__(self, embedder: Embedder, *, store: dict | None = None) -> None:
        self.embedder = embedder
        self._cache: dict[str, list[float]] = store if store is not None else {}
        self.hits = 0
        self.misses = 0

    def __call__(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vec = self.embedder(text)
        self._cache[text] = vec
        return vec


# Register embedders for discovery via yaab.extensions.get("embedder", ...).
register("embedder", "hashing", lambda **kw: hashing_embedder(**kw))
register("embedder", "litellm", lambda **kw: LiteLLMEmbedder(**kw))


__all__ = ["LiteLLMEmbedder", "CachingEmbedder", "Embedder"]
