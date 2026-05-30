"""Tests for RAG governance: faithfulness eval + retrieval guardrails."""

from __future__ import annotations

import pytest

from yaab import Document, KnowledgeBase
from yaab.rag import context_relevance, faithfulness
from yaab.rag.types import Chunk, RetrievedChunk


def _rc(text: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(text=text), score=score)


def test_faithfulness_grounded_answer_scores_high():
    chunks = [_rc("The capital of France is Paris, a major European city.")]
    assert faithfulness("Paris is the capital", chunks) == 1.0


def test_faithfulness_hallucinated_answer_scores_low():
    chunks = [_rc("The capital of France is Paris.")]
    score = faithfulness("Tokyo Japan population skyscrapers earthquakes", chunks)
    assert score < 0.3


def test_context_relevance():
    chunks = [_rc("Eiffel Tower Paris France")]
    # All query terms appear in the context -> full relevance.
    assert context_relevance("Eiffel Tower Paris", chunks) == 1.0
    # No overlap -> zero relevance.
    assert context_relevance("quantum chromodynamics lattice", chunks) == 0.0


@pytest.mark.asyncio
async def test_min_score_filters_weak_recall():
    # Force everything below threshold to be dropped.
    kb = KnowledgeBase(min_score=2.0)  # impossible cosine score => all dropped
    kb.add(Document(text="something", source="s"))
    results = await kb.retrieve("anything", k=5)
    assert results == []


@pytest.mark.asyncio
async def test_context_guard_drops_rejected_chunks():
    # Guard rejects any chunk mentioning "secret".
    def guard(rc: RetrievedChunk) -> bool:
        return "secret" not in rc.text.lower()

    kb = KnowledgeBase(context_guard=guard)
    kb.add(Document(text="The public docs are here.", source="pub"))
    kb.add(Document(text="The secret password is hunter2.", source="priv"))
    results = await kb.retrieve("docs password", k=5)
    assert all("secret" not in r.text.lower() for r in results)


@pytest.mark.asyncio
async def test_faithfulness_evaluator_llm_judge():
    from yaab.models.test_model import TestModel
    from yaab.rag import FaithfulnessEvaluator

    evaluator = FaithfulnessEvaluator(TestModel("0.9"))
    score = await evaluator.ascore("Paris is the capital", [_rc("Paris is the capital of France")])
    assert score == 0.9
