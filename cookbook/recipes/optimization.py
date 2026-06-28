"""Recipe: optimization — tune a prompt at build time, freeze for prod.

A ``Predict`` module over a typed ``Signature`` is compiled by an optimizer
(``BootstrapFewShot``) against a training set + metric into a frozen artifact.

    python -m cookbook.recipes.optimization
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab.governance.eval import Case
from yaab.models.test_model import TestModel
from yaab.optimize import BootstrapFewShot, Predict


async def run() -> dict:
    module = Predict("input -> output", model=TestModel("output: yes"))
    train = [Case(name="c1", inputs={"input": "is the sky blue?"}, expected="yes")]

    def metric(case, pred):
        return 1.0 if pred.get("output") == case.expected else 0.0

    artifact = await BootstrapFewShot().compile(module, train, metric)
    expect(artifact.optimizer == "bootstrap_few_shot", "expected a bootstrap artifact")
    expect(
        artifact.train_score == 1.0, f"expected a perfect train score, got {artifact.train_score}"
    )
    return {"optimizer": artifact.optimizer, "train_score": artifact.train_score}


if __name__ == "__main__":
    print(asyncio.run(run()))
