"""Recipe: loop until done — re-run an agent until a quality bar (or a cap).

``LoopAgent`` re-runs its agent over accumulating state until ``stop=`` fires or
``max_iterations`` is hit. Offline a ``FunctionModel`` improves a score each pass.

    python -m cookbook.recipes.loop
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, LoopAgent
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel


def _make_drafter() -> Agent:
    calls = {"n": 0}

    def fn(messages):
        calls["n"] += 1
        return ModelResponse(content=f"draft v{calls['n']} (score {calls['n']})")

    return Agent("drafter", model=FunctionModel(fn), instructions="Improve the draft.")


async def run() -> dict:
    drafter = _make_drafter()
    # Stop once the draft reaches "score 3"; the cap is a safety net.
    refine = LoopAgent("refine", drafter, max_iterations=5, stop=lambda out: "score 3" in str(out))

    result = await refine.run("Write a tagline.")
    answer = str(result.output)
    expect("score 3" in answer, f"expected the loop to reach score 3, got {answer!r}")
    return {"final": answer}


if __name__ == "__main__":
    print(asyncio.run(run()))
