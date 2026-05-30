"""Tests for A2A depth: task polling, streaming, OAuth tokens, back-handoff (Tier 3b)."""

from __future__ import annotations

import pytest

from yaab import Agent, Swarm
from yaab.multiagent import SwarmState
from yaab.models.test_model import TestModel


@pytest.mark.asyncio
async def test_a2a_task_poll_by_id():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.a2a import RemoteAgent
    from yaab.serve import fastapi_server_app

    server_agent = Agent("remote", model=TestModel("done"))
    client = TestClient(fastapi_server_app(server_agent))

    async def transport(method, path, json):
        return client.request(method, path, json=json).json()

    remote = RemoteAgent("http://server", transport=transport)
    result = await remote.run("do it")
    assert result.output == "done"
    # The task is now pollable by id.
    task = await remote.get_task(result.run_id)
    assert task["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_a2a_token_provider_used():
    pytest.importorskip("fastapi")
    seen = {}

    async def transport(method, path, json):
        return {"name": "remote"}  # minimal card

    from yaab.a2a import RemoteAgent

    calls = {"n": 0}

    def token_provider():
        calls["n"] += 1
        return f"tok-{calls['n']}"

    remote = RemoteAgent("http://s", token_provider=token_provider)
    # _headers() should call the provider for a fresh token each time.
    h1 = remote._headers()
    h2 = remote._headers()
    assert h1["Authorization"] == "Bearer tok-1"
    assert h2["Authorization"] == "Bearer tok-2"


def test_a2a_streaming_task_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.serve import fastapi_server_app

    agent = Agent("a", model=TestModel("streamed task result"))
    client = TestClient(fastapi_server_app(agent))
    with client.stream(
        "POST", "/a2a/tasks/stream", json={"message": {"parts": [{"text": "hi"}]}}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert '"state": "working"' in body
    assert '"state": "completed"' in body
    assert "[DONE]" in body


def test_a2a_get_unknown_task_404():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.serve import fastapi_server_app

    agent = Agent("a", model=TestModel("x"))
    client = TestClient(fastapi_server_app(agent))
    assert client.get("/a2a/tasks/does-not-exist").status_code == 404


@pytest.mark.asyncio
async def test_swarm_back_handoff_to_orchestrator():
    # Specialist hands control back to the triage/orchestrator agent.
    triage_model = TestModel(custom_output="triaged", call_tools=["handoff_to_specialist"])
    # specialist hands back to triage on its first turn, then triage finalizes.
    specialist_model = TestModel(custom_output="x", call_tools=["handoff_to_triage"])
    triage = Agent("triage", model=triage_model)
    specialist = Agent("specialist", model=specialist_model)

    swarm = Swarm("support", [triage, specialist], entry="triage", max_handoffs=4)
    # The specialist must have a handoff tool back to the orchestrator.
    assert any(t.name == "handoff_to_triage" for t in specialist.tools)
    result = await swarm.run("help", deps=SwarmState())
    assert result.output is not None
