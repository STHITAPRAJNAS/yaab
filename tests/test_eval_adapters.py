"""Tests for the extensible eval adapter layer (RAGAS/DeepEval-style)."""

from __future__ import annotations

import pytest

from yaab import available_metrics, get_metric, register_metric
from yaab.eval import score
from yaab.governance.eval import Case
from yaab.rag.types import Chunk, RetrievedChunk


def test_builtin_metrics_registered():
    names = set(available_metrics())
    assert {
        "exact_match",
        "contains",
        "regex",
        "json_match",
        "numeric_tolerance",
        "levenshtein",
        "llm_judge",
        "faithfulness",
        "context_relevance",
    } <= names


def test_external_adapters_registered():
    names = set(available_metrics())
    # RAGAS + DeepEval adapters register lazily without importing the libs.
    assert "ragas:faithfulness" in names
    assert "ragas:answer_relevancy" in names
    assert "deepeval:answer_relevancy" in names
    assert "deepeval:hallucination" in names


def test_get_builtin_metric():
    m = get_metric("exact_match")
    assert m.name == "exact_match"
    assert m.evaluate(Case(expected="x"), "x") == 1.0


@pytest.mark.asyncio
async def test_score_normalizes_sync_and_async():
    # sync metric (evaluate)
    s1 = await score(get_metric("contains"), Case(expected="lo"), "hello")
    assert s1 == 1.0

    # async metric (ascore) — faithfulness reads chunks from case.metadata
    chunks = [RetrievedChunk(chunk=Chunk(text="Paris is the capital of France"), score=1.0)]
    faith = get_metric("faithfulness")
    s2 = await score(faith, Case(inputs="capital?", metadata={"chunks": chunks}), "Paris capital")
    assert 0.0 <= s2 <= 1.0


def test_register_custom_metric_extensible():
    class MyMetric:
        name = "exclaims"

        def evaluate(self, case, output):
            return 1.0 if str(output).endswith("!") else 0.0

    register_metric("exclaims", lambda **kw: MyMetric())
    assert "exclaims" in available_metrics()
    assert get_metric("exclaims").evaluate(Case(), "wow!") == 1.0


def test_ragas_adapter_requires_lib():
    # The factory builds the adapter object; scoring it raises without ragas.
    m = get_metric("ragas:faithfulness")
    assert m.name == "ragas:faithfulness"


def test_deepeval_adapter_requires_lib():
    m = get_metric("deepeval:answer_relevancy")
    assert m.name == "deepeval:answer_relevancy"


@pytest.mark.asyncio
async def test_ragas_adapter_raises_without_lib():
    m = get_metric("ragas:faithfulness")
    with pytest.raises(RuntimeError):
        await m.ascore(Case(inputs="q", expected="a"), "out")


@pytest.mark.asyncio
async def test_experiment_runs_async_metrics():
    from yaab.governance.eval import Dataset, Experiment

    chunks = [RetrievedChunk(chunk=Chunk(text="the sky is blue"), score=1.0)]
    ds = Dataset(
        name="rag",
        cases=[Case(name="c1", inputs="color of sky?", metadata={"chunks": chunks})],
    )
    exp = Experiment(ds, [get_metric("faithfulness")], name="rag-eval")
    result = await exp.run(lambda x: "the sky is blue")
    assert "faithfulness" in result.aggregate
    assert result.results[0].error is None
