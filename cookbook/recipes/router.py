"""Recipe: router — deterministic exclusive choice, zero model calls.

``RouterAgent`` runs exactly one of N branches, chosen by a plain picker — cheap,
deterministic, fully auditable.

    python -m cookbook.recipes.router
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, RouterAgent
from yaab.models.test_model import TestModel


async def run() -> dict:
    billing: Agent = Agent(
        "billing",
        model=TestModel(custom_output="Billing will help."),
        instructions="Handle billing.",
    )
    tech: Agent = Agent(
        "tech", model=TestModel(custom_output="Tech support here."), instructions="Handle tech."
    )
    router = RouterAgent.from_picker(
        "support",
        lambda text, ctx: "billing" if "invoice" in text.lower() else "tech",
        to={"billing": billing, "tech": tech},
    )

    invoice = await router.run("I have a question about my invoice")
    crash = await router.run("the app keeps crashing")
    expect("billing" in str(invoice.output).lower(), "invoice should route to billing")
    expect("tech" in str(crash.output).lower(), "crash should route to tech")
    return {"invoice": str(invoice.output), "crash": str(crash.output)}


if __name__ == "__main__":
    print(asyncio.run(run()))
