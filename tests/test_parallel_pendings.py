"""Parallel pauses: two guarded tools in one model turn pause TOGETHER.

A single model turn that calls two guarded tools must pause on *both*, not drop
one. ``result.pending`` lists every parked decision; each is decided (possibly
with a different verb), and one multiplexed resume runs every held tool with its
own decision.
"""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.governance import InMemoryApprovalStore, ToolApprovalPlugin, approvals
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool
from yaab.types import ToolCall

_PAID: list[tuple[str, int]] = []


def _pay(invoice: str = "", amount: int = 0) -> str:
    """Pay an invoice."""
    _PAID.append((invoice, amount))
    return f"paid {invoice}: {amount}"


pay_invoice = FunctionTool(_pay, name="pay_invoice")


def _calls_two_then_answers() -> TestModel:
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[
                    ToolCall(name="pay_invoice", arguments={"invoice": "A", "amount": 100}),
                    ToolCall(name="pay_invoice", arguments={"invoice": "B", "amount": 200}),
                ],
                finish_reason="tool_calls",
            ),
            "both invoices handled",
        ]
    )


@pytest.fixture(autouse=True)
def _reset():
    _PAID.clear()
    yield
    _PAID.clear()


def _build(store, model):
    runner = Runner(
        run_checkpointer=MemorySaver(),
        plugins=[ToolApprovalPlugin(tools=["pay_invoice"], mode="queue", store=store)],
    )
    return Agent("payer", model=model, tools=[pay_invoice], runner=runner)


# --------------------------------------------------------------------------
# Both guarded calls in one turn pause together — neither is dropped.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_two_guarded_tools_pause_together():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_two_then_answers())

    result = await agent.run("pay both invoices", resume_id="par1")
    assert result.paused
    assert len(result.pending) == 2
    tools = sorted((p.tool, p.arguments["invoice"]) for p in result.pending)
    assert tools == [("pay_invoice", "A"), ("pay_invoice", "B")]
    # Two distinct durable records, both pending.
    rows = await store.list_pending()
    assert len(rows) == 2
    assert _PAID == []


# --------------------------------------------------------------------------
# Decide each (one approve, one edit) and resume once via multiplex.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multiplexed_resume_runs_each_held_tool_with_its_decision():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_two_then_answers())
    result = await agent.run("pay both invoices", resume_id="par2")
    assert len(result.pending) == 2

    by_invoice = {p.arguments["invoice"]: p for p in result.pending}
    d_a = await approvals.approve(
        result, approval_id=by_invoice["A"].approval_id, by="alice", store=store
    )
    d_b = await approvals.edit(
        result,
        approval_id=by_invoice["B"].approval_id,
        by="alice",
        arguments={"invoice": "B", "amount": 50},
        store=store,
    )
    bundle = await approvals.multiplex(result, {d_a.approval_id: d_a, d_b.approval_id: d_b})

    final = await agent.run(resume=bundle)
    assert not final.paused
    assert final.output == "both invoices handled"
    # A ran with its original amount, B ran with the EDITED amount (50, not 200).
    assert sorted(_PAID) == [("A", 100), ("B", 50)]
