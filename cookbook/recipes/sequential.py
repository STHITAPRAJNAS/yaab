"""Recipe: sequential pipeline — fixed steps, each building on the last.

``SequentialAgent`` runs children in order over one shared state; ``writes=``
captures a step's output where the next step reads it via ``{key}``.

    python -m cookbook.recipes.sequential
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, SequentialAgent
from yaab.models.test_model import TestModel


async def run() -> dict:
    classify: Agent = Agent(
        "classify",
        model=TestModel(custom_output="billing"),
        instructions="Classify the request topic in one word.",
        writes="topic",
    )
    reply: Agent = Agent(
        "reply",
        model=TestModel(custom_output="Your billing question is being handled."),
        instructions="The topic is {topic}. Answer the user's request.",
    )
    pipeline = SequentialAgent("triage", [classify, reply])

    result = await pipeline.run("I was double-charged this month.")
    answer = str(result.output)
    expect("billing" in answer.lower(), f"expected the reply step's output, got {answer!r}")
    return {"answer": answer}


if __name__ == "__main__":
    print(asyncio.run(run()))
