"""Recipe: conditions — guard what runs with when=/stop=/else_=.

One ``Condition`` concept (a callable or a safe expression over read-only state)
gates a step. Here an escalation step only runs for angry messages, else a
fallback apologizes.

    python -m cookbook.recipes.conditions
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, SequentialAgent, Step
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel, TestModel


def _classifier() -> FunctionModel:
    def fn(messages):
        text = messages[-1].content.lower() if messages else ""
        tone = "angry" if ("angry" in text or "unacceptable" in text) else "calm"
        return ModelResponse(content=tone)

    return FunctionModel(fn)


def _pipeline() -> SequentialAgent:
    classify: Agent = Agent(
        "classify", model=_classifier(), instructions="Classify tone in one word.", writes="tone"
    )
    escalate: Agent = Agent(
        "escalate",
        model=TestModel(custom_output="Escalated to a human agent."),
        instructions="Escalate to a human.",
    )
    fallback: Agent = Agent(
        "fallback",
        model=TestModel(custom_output="Sorry for the trouble; logged."),
        instructions="Apologize and log.",
    )
    # escalate runs only when the input contains "angry"; otherwise fallback runs.
    return SequentialAgent(
        "triage",
        [classify, Step(escalate, when='"angry" in input', else_=fallback)],
    )


async def run() -> dict:
    angry = await _pipeline().run("angry: this is unacceptable")
    calm = await _pipeline().run("could you please help")
    expect("escalat" in str(angry.output).lower(), "angry message should escalate")
    expect("sorry" in str(calm.output).lower(), "calm message should hit the fallback")
    return {"angry": str(angry.output), "calm": str(calm.output)}


if __name__ == "__main__":
    print(asyncio.run(run()))
