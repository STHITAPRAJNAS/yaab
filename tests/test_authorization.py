"""Tests for pre-tool authorization and idempotency (Tier 2)."""

from __future__ import annotations

import pytest

from yaab import Agent, Runner, tool
from yaab.exceptions import PolicyViolation
from yaab.governance import (
    AuditLog,
    CallableAuthorizer,
    IdempotencyPlugin,
    RBACAuthorizer,
    ToolAuthorizationPlugin,
)
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import RunContext, ToolCall


@tool
def update_inventory(sku: str = "x") -> str:
    """Mutate inventory."""
    return f"updated {sku}"


@tool
def check_stock(sku: str = "x") -> str:
    """Read-only check."""
    return f"in stock {sku}"


def test_rbac_authorizer_deny_list():
    auth = RBACAuthorizer(deny=["update_inventory"])
    ctx = RunContext()
    assert auth.authorize("check_stock", {}, ctx).allowed
    assert not auth.authorize("update_inventory", {}, ctx).allowed


def test_rbac_authorizer_allow_list_and_capability():
    auth = RBACAuthorizer(
        allow=["update_inventory"], require_capability={"update_inventory": "write"}
    )
    ctx = RunContext()
    # Not allowed without the capability.
    assert not auth.authorize("update_inventory", {}, ctx).allowed
    ctx.state["capabilities"] = ["write"]
    assert auth.authorize("update_inventory", {}, ctx).allowed
    # check_stock isn't on the allow list.
    assert not auth.authorize("check_stock", {}, ctx).allowed


def test_callable_authorizer():
    auth = CallableAuthorizer(lambda tool, args, ctx: tool != "update_inventory")
    ctx = RunContext()
    assert auth.authorize("check_stock", {}, ctx).allowed
    assert not auth.authorize("update_inventory", {}, ctx).allowed


@pytest.mark.asyncio
async def test_authorization_plugin_soft_block_feeds_model():
    audit = AuditLog()
    plugin = ToolAuthorizationPlugin([RBACAuthorizer(deny=["update_inventory"])], audit=audit)
    # Model calls the forbidden tool first, then answers.
    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="update_inventory", arguments={"sku": "a"})],
                finish_reason="tool_calls",
            ),
            "could not update, sorry",
        ]
    )
    runner = Runner(plugins=[plugin])
    agent = Agent("a", model=model, tools=[update_inventory])
    result = await runner.run(agent, "update sku a")
    assert result.output == "could not update, sorry"
    # The denial was audited.
    assert any(e.payload.get("action") == "deny" for e in audit.events)


@pytest.mark.asyncio
async def test_authorization_plugin_hard_raises():
    plugin = ToolAuthorizationPlugin([RBACAuthorizer(deny=["update_inventory"])], hard=True)
    model = TestModel(custom_output="x", call_tools=["update_inventory"])
    runner = Runner(plugins=[plugin])
    agent = Agent("a", model=model, tools=[update_inventory])
    with pytest.raises(PolicyViolation):
        await runner.run(agent, "go")


@pytest.mark.asyncio
async def test_idempotency_dedupes_side_effecting_tool():
    calls = {"n": 0}

    @tool
    def charge(order_id: str = "o1") -> str:
        """Charge an order once."""
        calls["n"] += 1
        return f"charged {order_id}"

    plugin = IdempotencyPlugin(tools=["charge"])
    # Model calls charge twice with identical args across two responses.
    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="charge", arguments={"order_id": "o1"})],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[ToolCall(name="charge", arguments={"order_id": "o1"})],
                finish_reason="tool_calls",
            ),
            "done",
        ]
    )
    runner = Runner(plugins=[plugin])
    agent = Agent("a", model=model, tools=[charge], max_steps=5)
    result = await runner.run(agent, "charge order o1 twice")
    assert result.output == "done"
    # The actual side effect happened only once despite two tool calls.
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_idempotency_custom_key():
    calls = {"n": 0}

    @tool
    def send(to: str = "a", body: str = "") -> str:
        """Send a message."""
        calls["n"] += 1
        return "sent"

    # Key only on `to`, so different bodies to the same recipient dedupe.
    plugin = IdempotencyPlugin(tools=["send"], key_fn=lambda t, a: a.get("to", ""))
    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="send", arguments={"to": "x", "body": "hi"})],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[ToolCall(name="send", arguments={"to": "x", "body": "different"})],
                finish_reason="tool_calls",
            ),
            "done",
        ]
    )
    runner = Runner(plugins=[plugin])
    agent = Agent("a", model=model, tools=[send], max_steps=5)
    await runner.run(agent, "send twice")
    assert calls["n"] == 1
