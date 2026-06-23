"""Support desk — a grounded, governed, safe-to-act support agent.

A realistic "talk to your docs + act" assistant that combines six capabilities:

* **RAG** — answers are grounded in a help-center ``KnowledgeBase``.
* **Tools** — an ``order_status`` lookup and a side-effecting ``issue_refund``.
* **HITL approval** — refunds over a threshold are gated by ``ToolApprovalPlugin``
  (a large refund is denied and never executes).
* **Spend governance** — a tenant already over budget is blocked before any call.
* Plus the same agent is servable behind the OpenAI-compatible API (see README).

Offline it runs deterministically with scripted models; set ``YAAB_SAMPLE_MODEL``
to drive it with a real model.

    python -m cookbook.apps.support_desk
"""

from __future__ import annotations

import asyncio
from typing import Any

from cookbook._harness import expect, resolve_model
from yaab import Agent, KnowledgeBase, Runner, tool
from yaab.governance import ToolApprovalPlugin
from yaab.governance.budget import Budget, InMemorySpendStore, SpendGovernancePlugin
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.plugins.builtins import BudgetExceeded
from yaab.types import ToolCall

_HELP_DOCS = [
    ("refunds.md", "Refunds are processed within 5 business days to the original payment method."),
    ("shipping.md", "Standard shipping takes 3-5 business days; express ships next day."),
]
_ORDERS = {"A100": "shipped", "A200": "processing"}

# Tracks whether the guarded refund actually executed (for the assertions).
_state = {"refunded": False}


@tool
def order_status(order_id: str) -> str:
    """Look up the delivery status of an order by id."""
    return _ORDERS.get(order_id, "unknown order")


@tool
def issue_refund(order_id: str, amount: int) -> str:
    """Refund an order (guarded — large refunds need human approval)."""
    _state["refunded"] = True
    return f"refunded ${amount} for {order_id}"


def _help_center() -> KnowledgeBase:
    kb = KnowledgeBase(name="helpcenter")
    for source, text in _HELP_DOCS:
        kb.add_text(text, source=source)
    return kb


def _refund_model(amount: int) -> FunctionModel:
    """Offline model: ask to refund ``amount``, then conclude after the decision."""
    calls = {"n": 0}

    def fn(messages: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(name="issue_refund", arguments={"order_id": "A100", "amount": amount})
                ]
            )
        return ModelResponse(content="Your refund request has been recorded.")

    return FunctionModel(fn)


# Every customer in this demo belongs to the "acme" tenant.
def _tenant_of(identity: str | None) -> str:
    return "acme"


def build(model: Any = None, *, refund_amount: int = 500) -> tuple[Agent, InMemorySpendStore]:
    """Wire the support agent with refund approval + per-tenant spend governance.

    Returns the agent and its spend store (so a caller can inspect/seed spend).
    """
    spend = InMemorySpendStore()

    async def approve_small_refunds(tool_name: str, args: dict, ctx: Any) -> bool:
        # Auto-approve small refunds; deny large ones (stand-in for a human reviewer).
        return args.get("amount", 0) <= 100

    approval = ToolApprovalPlugin(tools=["issue_refund"], approver=approve_small_refunds)
    governance = SpendGovernancePlugin(
        spend, {"tenant:acme": Budget(50.0, window="day")}, tenant_of=_tenant_of
    )
    runner = Runner(plugins=[approval, governance])
    agent: Agent = Agent(
        "support",
        model=resolve_model(model, offline_default=_refund_model(refund_amount)),
        instructions="Help customers; use order_status and issue_refund.",
        tools=[order_status, issue_refund],
        runner=runner,
    )
    return agent, spend


async def run() -> dict:
    _state["refunded"] = False
    agent, spend = build()

    # 1. Grounded answer from the help center (RAG).
    hits = await _help_center().retrieve("how long do refunds take?", k=1)
    expect(bool(hits) and "refund" in hits[0].chunk.text.lower(), "RAG should ground the answer")

    # 2. A large refund is denied by approval and never executes.
    result = await agent.run("Please refund $500 for order A100.", identity="alice")
    expect(not _state["refunded"], "a $500 refund must be denied, not executed")

    # 3. Spend governance: the acme tenant has already spent over its $50 daily cap,
    #    so the next request from any of its customers is blocked before a model call.
    import time

    await spend.record("tenant:acme", 60.0, at=time.time())
    blocked = False
    try:
        await agent.run("Any update on my order?", identity="bob")
    except BudgetExceeded:
        blocked = True
    expect(blocked, "an over-budget tenant must be blocked")

    return {
        "grounded": hits[0].chunk.source,
        "refund_executed": _state["refunded"],
        "tenant_blocked_over_budget": blocked,
        "answer": str(result.output),
    }


if __name__ == "__main__":
    print(asyncio.run(run()))
