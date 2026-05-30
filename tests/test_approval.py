"""Tests for fast-path human-in-the-loop tool approval."""

from __future__ import annotations

import pytest

from yaab import Agent, Runner, tool
from yaab.exceptions import ApprovalRequired
from yaab.governance import AuditLog, ToolApprovalPlugin
from yaab.governance.audit import AuditKind
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import ToolCall


@tool
def wire_transfer(amount: int = 0) -> str:
    """Send money."""
    return f"sent {amount}"


def _calls_wire_then_answers():
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="wire_transfer", arguments={"amount": 100})],
                finish_reason="tool_calls",
            ),
            "all done",
        ]
    )


@pytest.mark.asyncio
async def test_inline_approver_grants():
    async def approve(tool, args, ctx):
        return args["amount"] < 1000

    runner = Runner(plugins=[ToolApprovalPlugin(tools=["wire_transfer"], approver=approve)])
    agent = Agent("a", model=_calls_wire_then_answers(), tools=[wire_transfer])
    result = await runner.run(agent, "wire 100")
    assert result.output == "all done"


@pytest.mark.asyncio
async def test_inline_approver_rejects_feeds_model():
    async def reject(tool, args, ctx):
        return False

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="wire_transfer", arguments={"amount": 100})],
                finish_reason="tool_calls",
            ),
            "could not complete the transfer",
        ]
    )
    runner = Runner(plugins=[ToolApprovalPlugin(tools=["wire_transfer"], approver=reject)])
    agent = Agent("a", model=model, tools=[wire_transfer])
    result = await runner.run(agent, "wire 100")
    assert result.output == "could not complete the transfer"


@pytest.mark.asyncio
async def test_block_mode_raises_approval_required():
    runner = Runner(plugins=[ToolApprovalPlugin(tools=["wire_transfer"])])
    agent = Agent("a", model=_calls_wire_then_answers(), tools=[wire_transfer])
    with pytest.raises(ApprovalRequired) as ei:
        await runner.run(agent, "wire 100")
    assert ei.value.tool == "wire_transfer"
    assert ei.value.arguments == {"amount": 100}


@pytest.mark.asyncio
async def test_predicate_guards_by_value():
    # Only require approval for large amounts.
    def needs(tool, args, ctx):
        return tool == "wire_transfer" and args.get("amount", 0) >= 1000

    async def approve(tool, args, ctx):
        return False

    runner = Runner(plugins=[ToolApprovalPlugin(needs_approval=needs, approver=approve)])
    # amount is 100 < 1000 => no approval needed => tool runs
    agent = Agent("a", model=_calls_wire_then_answers(), tools=[wire_transfer])
    result = await runner.run(agent, "wire 100")
    assert result.output == "all done"


@pytest.mark.asyncio
async def test_approval_is_audited():
    log = AuditLog()

    async def approve(tool, args, ctx):
        return True

    runner = Runner(
        plugins=[ToolApprovalPlugin(tools=["wire_transfer"], approver=approve, audit=log)]
    )
    agent = Agent("a", model=_calls_wire_then_answers(), tools=[wire_transfer])
    await runner.run(agent, "wire 100")
    assert any(e.kind is AuditKind.APPROVAL for e in log.events)
