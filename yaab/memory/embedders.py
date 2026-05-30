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
                "litellm is required for LiteLLMEmbedder. `pip install 'yaab[litellm]'`."
            ) from exc
        resp = litellm.embedding(model=self.model, input=[text], **self.params)
        return list(resp["data"][0]["embedding"])


# Register both embedders for discovery via yaab.extensions.get("embedder", ...).
register("embedder", "hashing", lambda **kw: hashing_embedder(**kw))
register("embedder", "litellm", lambda **kw: LiteLLMEmbedder(**kw))


__all__ = ["LiteLLMEmbedder", "Embedder"]
