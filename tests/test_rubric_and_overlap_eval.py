"""Rubric-based LLM judge + text-overlap metrics for evaluation.

`RubricJudge` scores an output against named criteria, returning a per-criterion
breakdown and an aggregate (vs the existing freeform `LLMJudge`). `ResponseMatch`
is a deterministic token-overlap (ROUGE-style) metric, alongside the exact/regex
metrics that already ship.
"""

from __future__ import annotations

import pytest

from yaab.governance.eval import Case, ResponseMatch, RubricJudge
from yaab.testing import FunctionModel


def test_response_match_token_overlap():
    m = ResponseMatch()
    case = Case(inputs="q", expected="the quick brown fox")
    # Full overlap -> 1.0; partial -> in (0, 1); none -> 0.
    assert m.evaluate(case, "the quick brown fox") == 1.0
    assert 0.0 < m.evaluate(case, "the quick fox") < 1.0
    assert m.evaluate(case, "completely different words here") < 0.5


def test_response_match_empty_expected():
    m = ResponseMatch()
    assert m.evaluate(Case(inputs="q", expected=""), "anything") == 0.0


@pytest.mark.asyncio
async def test_rubric_judge_per_criterion_and_aggregate():
    # A judge model that returns a JSON object scoring each criterion.
    def model_fn(messages):
        from yaab.models.base import ModelResponse

        return ModelResponse(content='{"accuracy": 1.0, "tone": 0.5}')

    judge = RubricJudge(
        FunctionModel(model_fn),
        rubric={"accuracy": "Is it factually correct?", "tone": "Is the tone professional?"},
    )
    case = Case(inputs="explain X", expected="the correct explanation")
    breakdown = await judge.ascore_rubric(case, "an explanation")
    assert breakdown.scores == {"accuracy": 1.0, "tone": 0.5}
    assert breakdown.aggregate == pytest.approx(0.75)  # mean of the criteria


@pytest.mark.asyncio
async def test_rubric_judge_ascore_returns_aggregate_float():
    def model_fn(messages):
        from yaab.models.base import ModelResponse

        return ModelResponse(content='{"correctness": 0.8}')

    judge = RubricJudge(FunctionModel(model_fn), rubric={"correctness": "right?"})
    score = await judge.ascore(Case(inputs="q", expected="e"), "o")
    assert score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_rubric_judge_tolerates_garbage_model_output():
    def model_fn(messages):
        from yaab.models.base import ModelResponse

        return ModelResponse(content="not json at all")

    judge = RubricJudge(FunctionModel(model_fn), rubric={"x": "y"})
    assert await judge.ascore(Case(inputs="q", expected="e"), "o") == 0.0
