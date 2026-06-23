"""Recipe: tools — give an agent a typed function to call.

A support agent looks up an order's status with a ``@tool``. Offline, a scripted
``TestModel`` calls the tool then answers; live, the model decides when to call.

    python -m cookbook.recipes.tools
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect, resolve_model
from yaab import Agent, tool
from yaab.models.test_model import TestModel

_ORDERS = {"A100": "shipped", "A200": "processing"}


@tool
def order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id."""
    return _ORDERS.get(order_id, "unknown order")


async def run() -> dict:
    model = resolve_model(
        offline_default=TestModel(
            call_tools=["order_status"],
            custom_output="Order A100 has shipped.",
        )
    )
    agent: Agent = Agent(
        "support",
        model=model,
        instructions="Use order_status to answer questions about an order.",
        tools=[order_status],
    )
    result = await agent.run("Where is order A100?")
    answer = str(result.output)
    expect("ship" in answer.lower(), f"expected a shipped status, got {answer!r}")
    return {"answer": answer, "tool_used": "order_status"}


if __name__ == "__main__":
    print(asyncio.run(run()))
