"""Rerankers — reorder retrieved chunks for precision (a common RAG ask).

Retrieval recalls a broad candidate set; a reranker reorders it by a sharper
relevance signal and keeps the top ``n``. A :class:`Reranker` is a protocol so
cross-encoder / LLM / hosted rerankers drop in behind it.

Two dependency-free rerankers ship:

* :class:`KeywordReranker` — boost chunks by query-term overlap (lexical signal
  layered on top of vector recall — a cheap hybrid).
* :class:`LLMReranker` — ask a model to score each chunk's relevance (0–1).
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from .types import RetrievedChunk

_WORD_RE = re.compile(r"\w+")


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, results: list[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        ...


class KeywordReranker:
    """Blend the vector score with query-term overlap (lexical hybrid).

    Final score = ``(1 - weight) * vector_score + weight * lexical_overlap``,
    where overlap is the fraction of distinct query terms present in the chunk.
    """

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def rerank(
        self, query: str, results: list[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        terms = {w.lower() for w in _WORD_RE.findall(query)}
        rescored: list[RetrievedChunk] = []
        for r in results:
            chunk_terms = {w.lower() for w in _WORD_RE.findall(r.chunk.text)}
            overlap = len(terms & chunk_terms) / len(terms) if terms else 0.0
            blended = (1 - self.weight) * r.score + self.weight * overlap
            rescored.append(RetrievedChunk(chunk=r.chunk, score=blended))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_n]


class LLMReranker:
    """Score each chunk's relevance with a model and keep the top ``n``.

    Best-effort and model-agnostic; parsing failures fall back to the original
    retrieval score so a flaky judge never drops valid context.
    """

    def __init__(self, model: Any) -> None:
        from ..models import resolve_model

        self.model = resolve_model(model)

    async def arerank(
        self, query: str, results: list[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        from ..types import Message, Role

        rescored: list[RetrievedChunk] = []
        for r in results:
            prompt = (
                "Rate how relevant the passage is to the query on a scale of 0 to 1. "
                "Reply with only the number.\n\n"
                f"Query: {query}\n\nPassage: {r.chunk.text}\n\nRelevance:"
            )
            try:
                resp = await self.model.complete([Message(role=Role.USER, content=prompt)])
                score = float(re.search(r"[01](?:\.\d+)?", resp.content).group())  # type: ignore[union-attr]
            except (AttributeError, ValueError, TypeError):
                score = r.score
            rescored.append(RetrievedChunk(chunk=r.chunk, score=score))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_n]


class CrossEncoderReranker:
    """Cross-encoder reranker (``pip install sentence-transformers``).

    Scores each (query, chunk) pair with a cross-encoder model — the precision
    standard for reranking. The model is loaded lazily on first use; pass a
    preloaded ``model`` to inject one (or for testing).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "`pip install sentence-transformers`."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, results: list[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if not results:
            return []
        model = self._ensure_model()
        scores = model.predict([(query, r.chunk.text) for r in results])
        rescored = [
            RetrievedChunk(chunk=r.chunk, score=float(s))
            for r, s in zip(results, scores, strict=False)
        ]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_n]


__all__ = ["Reranker", "KeywordReranker", "LLMReranker", "CrossEncoderReranker"]
