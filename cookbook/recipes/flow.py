"""Recipe: flow — explicit, durable, branchable control flow.

``Flow`` is a builder (``.step/.route/.then/.start_at/.returns``) that lowers
onto the durable graph engine. Here a refund flow branches on the amount.

    python -m cookbook.recipes.flow
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Flow


def _refund_flow(amount: int) -> Flow:
    return (
        Flow[None, str]("refund")
        .step("parse", fn=lambda state, ctx: {"amount": amount})
        .route(
            "parse",
            lambda state, ctx: "auto" if state["amount"] < 100 else "review",
            to={"auto": "auto_refund", "review": "human_review"},
        )
        .step("auto_refund", fn=lambda state, ctx: {"out": "auto-refunded"})
        .step("human_review", fn=lambda state, ctx: {"out": "queued for review"})
        .then("auto_refund", Flow.DONE)
        .then("human_review", Flow.DONE)
        .start_at("parse")
        .returns("out")
    )


async def run() -> dict:
    small = await _refund_flow(50).run("refund #1")
    large = await _refund_flow(250).run("refund #2")
    expect(small.output == "auto-refunded", f"small refund should auto, got {small.output!r}")
    expect(large.output == "queued for review", f"large refund should review, got {large.output!r}")
    return {"small": small.output, "large": large.output}


if __name__ == "__main__":
    print(asyncio.run(run()))
