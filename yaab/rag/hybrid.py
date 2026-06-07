"""Sparse (BM25) retrieval and reciprocal-rank fusion for hybrid search.

A dense embedding recall and a sparse keyword (BM25) recall catch different
things: dense generalizes by meaning, sparse nails exact rare terms. Running
both and fusing their *rankings* with reciprocal-rank fusion (order-only, no
score calibration needed) gives retrieval that is robust to either signal being
weak — the default of true hybrid search.

This is a small, dependency-free in-memory BM25 (Okapi); for very large corpora
a real sparse index belongs in the vector store, but this covers the in-process
``KnowledgeBase`` path with no extra install.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """An in-memory Okapi BM25 index over short documents keyed by id."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:  # idempotent re-add: replace
            self.remove(doc_id)
        toks = _tokenize(text)
        self._docs[doc_id] = toks
        tf = Counter(toks)
        self._tf[doc_id] = tf
        for term in tf:
            self._df[term] += 1
        self._total_len += len(toks)

    def remove(self, doc_id: str) -> None:
        toks = self._docs.pop(doc_id, None)
        if toks is None:
            return
        for term in self._tf.pop(doc_id):
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._total_len -= len(toks)

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # Okapi IDF with the +0.5 smoothing, floored at 0 so common terms can't
        # subtract from a document's score.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        """Top-``k`` ``(doc_id, score)`` pairs for ``query``, highest first."""
        if not self._docs:
            return []
        avgdl = self._total_len / len(self._docs)
        q_terms = _tokenize(query)
        scored: list[tuple[str, float]] = []
        for doc_id, tf in self._tf.items():
            dl = len(self._docs[doc_id])
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0.0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id-lists into one by reciprocal-rank fusion.

    Each list contributes ``1 / (k + rank)`` to every id it ranks (rank starting
    at 1). Order-only, so the lists need no comparable scores. Returns
    ``(id, fused_score)`` sorted highest first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
