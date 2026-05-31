"""RemoteRegistryBackend — governance against a central HTTP agent registry.

Uses httpx.MockTransport to stand in for the central registry service, so the
test is fully offline/deterministic while exercising the real HTTP code path.
"""

from __future__ import annotations

import json

import httpx
import pytest

from yaab import Agent, Runner
from yaab.exceptions import NotRegisteredError
from yaab.governance import (
    AgentCard,
    AgentRegistry,
    ApprovalStatus,
    GovernanceMode,
    GovernanceService,
    RemoteRegistryBackend,
)
from yaab.models.test_model import TestModel


def _central_registry_transport() -> httpx.MockTransport:
    """A fake central registry: an in-memory dict behind the REST contract."""
    store: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")  # ["agents"] or ["agents", id]
        if request.method == "PUT" and len(parts) == 2:
            card = json.loads(request.content)
            store[parts[1]] = card
            return httpx.Response(200, json={"ok": True})
        if request.method == "GET" and len(parts) == 2:
            card = store.get(parts[1])
            return httpx.Response(200, json=card) if card else httpx.Response(404)
        if request.method == "GET" and parts == ["agents"]:
            return httpx.Response(200, json={"agents": list(store.values())})
        return httpx.Response(405)

    return httpx.MockTransport(handler)


def _remote_registry() -> AgentRegistry:
    client = httpx.Client(transport=_central_registry_transport(), base_url="http://registry.test")
    return AgentRegistry(RemoteRegistryBackend(client=client))


def test_register_and_fetch_roundtrips_custom_fields():
    reg = _remote_registry()
    reg.register(
        AgentCard(
            agent_id="support-bot",
            name="Support Bot",
            metadata={"cost_center": "CX-7"},
            usecase_id="UC-123",  # extra field, allowed + round-tripped
            blueprint="rag-support-v2",
        )
    )
    got = reg.get("support-bot")
    assert got is not None
    assert got.metadata == {"cost_center": "CX-7"}
    # extra (non-schema) fields survive the HTTP round-trip via extra="allow".
    assert got.usecase_id == "UC-123"
    assert got.blueprint == "rag-support-v2"


def test_fetch_missing_returns_none():
    reg = _remote_registry()
    assert reg.get("nope") is None


def test_list_and_inventory_from_remote():
    reg = _remote_registry()
    reg.register(AgentCard(agent_id="a", name="A", metadata={"usecase_id": "UC-1"}))
    reg.register(AgentCard(agent_id="b", name="B"))
    ids = {c.agent_id for c in reg.list()}
    assert ids == {"a", "b"}
    inv = {row["agent_id"]: row for row in reg.inventory()}
    assert inv["a"]["metadata"] == {"usecase_id": "UC-1"}


async def test_enforcing_gate_against_remote_registry():
    """The run-gate reads approval from the central registry on every run."""
    reg = _remote_registry()
    gov = GovernanceService(mode=GovernanceMode.ENFORCING, registry=reg)
    runner = Runner(governance=gov)

    # Unapproved (PENDING) -> refused.
    reg.register(AgentCard(agent_id="pending", name="P"))
    pending = Agent("pending", model=TestModel("hi"), registry_id="pending")
    with pytest.raises(NotRegisteredError):
        await runner.run(pending, "hello", identity="u1")

    # Approved in the central registry -> allowed.
    reg.register(
        AgentCard(agent_id="live", name="L", model_approval_status=ApprovalStatus.APPROVED)
    )
    live = Agent("live", model=TestModel("hello there"), registry_id="live")
    result = await runner.run(live, "hello", identity="u1")
    assert result.output == "hello there"
