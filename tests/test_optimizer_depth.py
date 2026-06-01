"""Optimizer depth (Phase E): a real BootstrapFewShotWithRandomSearch +
minibatched MIPROv2 candidate evaluation with iterative search loops.
"""

from __future__ import annotations

import pytest

from yaab.governance.eval import Case
from yaab.models.test_model import FunctionModel
from yaab.optimize import BootstrapFewShotWithRandomSearch, MIPROv2, Predict


def _answer_model() -> FunctionModel:
    """A model that answers arithmetic correctly by reading the question from
    the rendered prompt — so bootstrapping harvests demos and scoring works."""
    table = {"2+2": "4", "3+3": "6", "4+4": "8", "5+5": "10"}

    def fn(messages):
        prompt = messages[-1].content
        for q, a in table.items():
            if q in prompt:
                return f"answer: {a}"
        return "answer: ?"

    return FunctionModel(fn)


def _metric(case: Case, pred: dict) -> float:
    return 1.0 if str(case.expected) in str(pred.get("answer", "")) else 0.0


def _trainset() -> list[Case]:
    return [
        Case(name="a", inputs={"question": "2+2"}, expected="4"),
        Case(name="b", inputs={"question": "3+3"}, expected="6"),
        Case(name="c", inputs={"question": "4+4"}, expected="8"),
        Case(name="d", inputs={"question": "5+5"}, expected="10"),
    ]


@pytest.mark.asyncio
async def test_random_search_bootstraps_and_selects():
    module = Predict("question -> answer", model=_answer_model())
    train = _trainset()
    n_eval = {"n": 0}

    def counting_metric(case, pred):
        n_eval["n"] += 1
        return _metric(case, pred)

    opt = BootstrapFewShotWithRandomSearch(max_demos=2, num_candidates=5, seed=1)
    art = await opt.compile(module, train, counting_metric)

    assert art.optimizer == "bootstrap_rs"
    assert art.train_score == 1.0
    # Selected demos come from the bootstrapped pool and respect the cap.
    assert 0 <= len(art.demos) <= 2
    for demo in art.demos:
        assert "question" in demo and "answer" in demo
    # It evaluated multiple candidate demo sets (more than one scoring pass).
    assert n_eval["n"] > len(train)


@pytest.mark.asyncio
async def test_random_search_respects_max_demos():
    module = Predict("question -> answer", model=_answer_model())
    opt = BootstrapFewShotWithRandomSearch(max_demos=1, num_candidates=4, seed=2)
    art = await opt.compile(module, _trainset(), _metric)
    assert len(art.demos) <= 1


@pytest.mark.asyncio
async def test_random_search_handles_no_bootstrappable_demos():
    # A model that's always wrong -> empty pool -> falls back to zero-shot.
    module = Predict("question -> answer", model=FunctionModel(lambda m: "answer: wrong"))
    opt = BootstrapFewShotWithRandomSearch(max_demos=2, num_candidates=3, seed=0)
    art = await opt.compile(module, _trainset(), _metric)
    assert art.demos == []
    assert art.train_score == 0.0


@pytest.mark.asyncio
async def test_miprov2_minibatch_eval():
    module = Predict("question -> answer", model=_answer_model())
    # minibatch_size limits how many cases each candidate is scored on.
    opt = MIPROv2(minibatch_size=2)
    art = await opt.compile(module, _trainset(), _metric)
    assert art.optimizer == "miprov2"
    assert 0.0 <= art.train_score <= 1.0
