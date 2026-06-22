"""Recipe: agents — the typed unit of work.

Build an ``Agent`` with instructions and run it. Offline it answers from a
scripted ``TestModel``; set ``YAAB_SAMPLE_MODEL`` to run it against a real model.

    python -m cookbook.recipes.agents
"""

from __future__ import annotations

import asyncio

from yaab import Agent
from yaab.models.test_model import TestModel

from cookbook._harness import expect, resolve_model


async def run() -> dict:
    model = resolve_model(
        offline_default=TestModel(custom_output="The capital of France is Paris.")
    )
    agent = Agent(
        "geographer",
        model=model,
        instructions="Answer in one short, factual sentence.",
    )
    result = await agent.run("What is the capital of France?")
    answer = str(result.output)
    expect("paris" in answer.lower(), f"expected Paris in the answer, got {answer!r}")
    return {"answer": answer}


if __name__ == "__main__":
    print(asyncio.run(run()))
