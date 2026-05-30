"""Research assistant: a Sequential pipeline of researcher → writer agents.

Pattern: decompose a task into specialized stages that hand output to the next
stage. The researcher gathers points; the writer turns them into prose.
"""

from __future__ import annotations

from typing import Any

from yaab import Agent, SequentialAgent
from yaab.models.test_model import TestModel

from .._common import resolve_model


def build(model: Any = None) -> SequentialAgent:
    """Return a researcher→writer sequential pipeline."""
    researcher = Agent(
        "researcher",
        model=resolve_model(model, offline_default=TestModel("- fact one\n- fact two")),
        instructions="Research the topic. Output a terse bulleted list of key facts.",
    )
    writer = Agent(
        "writer",
        model=resolve_model(
            model, offline_default=TestModel("A concise summary based on the research.")
        ),
        instructions="Write a short, clear paragraph from the bullet points you receive.",
    )
    return SequentialAgent("research_pipeline", [researcher, writer])


async def run(topic: str = "the benefits of unit testing", model: Any = None) -> str:
    pipeline = build(model)
    result = await pipeline.run(topic)
    return result.output


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(run()))
