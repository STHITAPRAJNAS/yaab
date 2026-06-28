"""Recipe: evaluation — score an agent's output against expectations.

A metric is any object with a ``name`` and ``evaluate``/``ascore``. Here a
deterministic overlap metric scores a grounded answer, and the metric registry
resolves built-ins by name.

    python -m cookbook.recipes.evaluation
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import get_metric
from yaab.governance.eval import Case, ResponseMatch


async def run() -> dict:
    case = Case(inputs="capital of France?", expected="Paris is the capital of France")

    # Deterministic ROUGE-style overlap (no model call).
    overlap = ResponseMatch().evaluate(case, "Paris is the capital of France")
    expect(overlap == 1.0, f"a verbatim answer should score 1.0, got {overlap}")

    # Built-ins resolve by name through the registry.
    exact = get_metric("exact_match")
    score = exact.evaluate(Case(inputs="2+2?", expected="4"), "4")
    expect(score == 1.0, f"exact_match on a correct answer should be 1.0, got {score}")
    return {"overlap": overlap, "exact_match": score}


if __name__ == "__main__":
    print(asyncio.run(run()))
