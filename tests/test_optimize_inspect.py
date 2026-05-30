"""Test compiled-prompt inspection (Tier 1, DSPy #7830)."""

from __future__ import annotations

from yaab.models.test_model import TestModel
from yaab.optimize import Predict


def test_inspect_prompt_includes_inputs_and_demos():
    module = Predict("question -> answer", model=TestModel("x"))
    module.signature.instructions = "Answer accurately."
    module.demos = [{"question": "2+2?", "answer": "4"}]

    prompt = module.inspect_prompt(question="3+5?")
    assert "Answer accurately." in prompt
    assert "3+5?" in prompt  # the live input
    assert "2+2?" in prompt  # the few-shot demo
    assert "4" in prompt


def test_inspect_prompt_no_model_call():
    model = TestModel("y")
    module = Predict("q -> a", model=model)
    module.inspect_prompt(q="hi")
    assert model.calls == []  # inspection must not call the model
