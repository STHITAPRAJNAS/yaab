"""Recipe: map fan-out — one agent across many inputs concurrently.

``MapAgent`` derives a list of inputs and runs the same agent over each, bounded
by ``max_concurrency``.

    python -m cookbook.recipes.map_agent
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, MapAgent
from yaab.models.test_model import TestModel


async def run() -> dict:
    classifier: Agent = Agent(
        "classifier",
        model=TestModel(custom_output="positive"),
        instructions="Classify the sentiment of the review in one word.",
    )
    # Split a semicolon-separated batch into one input per review.
    mapper = MapAgent(
        "sentiment", classifier, map_inputs=lambda text: text.split(";"), max_concurrency=4
    )

    result = await mapper.run("great product; fast shipping; works well")
    outputs = result.output
    expect(isinstance(outputs, list), f"expected a list of results, got {type(outputs)!r}")
    expect(len(outputs) == 3, f"expected 3 classified reviews, got {len(outputs)}")
    return {"count": len(outputs)}


if __name__ == "__main__":
    print(asyncio.run(run()))
