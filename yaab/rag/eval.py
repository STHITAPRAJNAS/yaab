"""RAG-specific evaluation: faithfulness, context relevance, answer relevance.

RAGAS-style groundedness metrics are external to every agent SDK today — yet for
a governance-first SDK, *"is the answer grounded in retrieved context?"* is core
evidence. These evaluators plug into :mod:`yaab.governance.eval` (they implement
the same ``evaluate(case, output) -> float`` shape where practical) and also work
standalone over a query + answer + retrieved chunks.

The deterministic variants are dependency-free (lexical overlap); the LLM-judge
variants take any model and ask it to score 0–1.
"""

from __future__ import annotations

import re
from typing import Any

from .types import RetrievedChunk

_WORD_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def context_relevance(query: str, chunks: list[RetrievedChunk]) -> float:
    """Fraction of query terms covered by the retrieved context (0–1).

    A cheap recall proxy: low values mean retrieval surfaced off-topic context.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    ctx = set().union(*[_tokens(c.text) for c in chunks]) if chunks else set()
    return len(q & ctx) / len(q)


def faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float:
    """Fraction of answer terms supported by the retrieved context (0–1).

    A deterministic groundedness proxy: low values flag answer content that is
    *not* present in any retrieved chunk (potential hallucination). Stopword-ish
    short tokens are ignored to reduce noise.
    """
    ans = {t for t in _tokens(answer) if len(t) > 3}
    if not ans:
        return 1.0  # nothing substantive to ground
    ctx = set().union(*[_tokens(c.text) for c in chunks]) if chunks else set()
    return len(ans & ctx) / len(ans)


class FaithfulnessEvaluator:
    """LLM-judge groundedness: does the answer follow from the context?

    Model-agnostic and best-effort: a parse failure returns 0.0 so an
    ungradeable answer is never silently treated as faithful.
    """

    name = "faithfulness"

    def __init__(self, model: Any) -> None:
        from ..models import resolve_model

        self.model = resolve_model(model)

    async def ascore(self, answer: str, chunks: list[RetrievedChunk]) -> float:
        from ..types import Message, Role

        context = "\n".join(f"- {c.text}" for c in chunks)
        prompt = (
            "You are grading whether an ANSWER is fully supported by the CONTEXT. "
            "Reply with only a number from 0 (unsupported / hallucinated) to 1 "
            "(fully grounded).\n\n"
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nGroundedness score:"
        )
        try:
            resp = await self.model.complete([Message(role=Role.USER, content=prompt)])
            return float(re.search(r"[01](?:\.\d+)?", resp.content).group())  # type: ignore[union-attr]
        except (AttributeError, ValueError, TypeError):
            return 0.0


__all__ = ["context_relevance", "faithfulness", "FaithfulnessEvaluator"]
