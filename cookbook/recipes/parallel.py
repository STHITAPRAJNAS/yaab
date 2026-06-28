"""Recipe: parallel fan-out — several agents review the same input at once.

``ParallelAgent`` runs its children concurrently and returns a ``name -> result``
map.

    python -m cookbook.recipes.parallel
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, ParallelAgent
from yaab.models.test_model import TestModel


async def run() -> dict:
    legal: Agent = Agent(
        "legal",
        model=TestModel(custom_output="No legal blockers."),
        instructions="Review legal risk.",
    )
    finance: Agent = Agent(
        "finance",
        model=TestModel(custom_output="Within budget."),
        instructions="Review financial terms.",
    )
    panel = ParallelAgent("panel", [legal, finance])

    result = await panel.run("Approve the new vendor contract.")
    reviews = result.output
    expect(isinstance(reviews, dict), f"expected a name->result map, got {type(reviews)!r}")
    expect({"legal", "finance"} <= set(reviews), f"expected both reviewers, got {list(reviews)}")
    return {"reviewers": sorted(reviews)}


if __name__ == "__main__":
    print(asyncio.run(run()))
