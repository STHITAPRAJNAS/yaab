"""Recipe: typed output — get a validated object back, not a string.

Set ``output_type`` to a Pydantic model and the agent returns that type
(validated, with reflection/retry on mismatch).

    python -m cookbook.recipes.typed_output
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from cookbook._harness import expect, resolve_model
from yaab import Agent
from yaab.models.test_model import TestModel


class Ticket(BaseModel):
    title: str
    priority: int


async def run() -> dict:
    model = resolve_model(
        offline_default=TestModel(structured_output={"title": "Reset password", "priority": 1})
    )
    agent: Agent = Agent(
        "intake",
        model=model,
        instructions="Turn the request into a support ticket.",
        output_type=Ticket,
    )
    result = await agent.run("I can't log in and need a password reset urgently.")
    ticket = result.output
    expect(isinstance(ticket, Ticket), f"expected a Ticket, got {type(ticket)!r}")
    expect(ticket.priority >= 1, "expected a priority")
    return {"title": ticket.title, "priority": ticket.priority}


if __name__ == "__main__":
    print(asyncio.run(run()))
