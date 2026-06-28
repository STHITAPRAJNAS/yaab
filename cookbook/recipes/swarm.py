"""Recipe: swarm — autonomous hand-off between peer agents.

A ``Swarm`` augments each member with ``handoff_to_<peer>`` tools so a front-line
agent routes to whoever is best suited.

    python -m cookbook.recipes.swarm
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, Swarm
from yaab.models.test_model import TestModel


async def run() -> dict:
    # Offline: triage hands off to billing, which answers.
    triage: Agent = Agent(
        "triage",
        model=TestModel(call_tools=["handoff_to_billing"], custom_output="routing"),
        instructions="Route the user to billing or tech.",
    )
    billing: Agent = Agent(
        "billing",
        model=TestModel(custom_output="Billing department: your refund is on the way."),
        instructions="Handle billing questions.",
    )
    tech: Agent = Agent(
        "tech", model=TestModel(custom_output="Tech support here."), instructions="Handle tech."
    )
    swarm = Swarm("support", [triage, billing, tech], entry="triage", max_handoffs=4)

    result = await swarm.run("I need a refund on my last invoice")
    answer = str(result.output)
    expect(
        "billing" in answer.lower(), f"expected billing to answer after hand-off, got {answer!r}"
    )
    return {"answer": answer}


if __name__ == "__main__":
    print(asyncio.run(run()))
