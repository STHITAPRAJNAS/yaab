"""Tests for A2A client, MCP client, and token streaming."""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.testing import TestModel


@pytest.mark.asyncio
async def test_a2a_client_roundtrip_in_process():
    """RemoteAgent talks to a real get_fastapi_app server via an in-process transport."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.a2a import RemoteAgent
    from yaab.serve import get_fastapi_app

    server_agent = Agent("remote", model=TestModel("remote says hi"), registry_id="remote")
    client = TestClient(get_fastapi_app(server_agent, base_url="http://server"))

    async def transport(method, path, json):
        resp = client.request(method, path, json=json)
        return resp.json()

    remote = RemoteAgent("http://server", name="remote", transport=transport)
    card = await remote.fetch_card()
    assert card["name"] == "remote"

    result = await remote.run("hello")
    assert result.output == "remote says hi"

    # And it works as a tool the schema exposes.
    schema = remote.schema()
    assert schema["function"]["name"] == "remote"


@pytest.mark.asyncio
async def test_a2a_remote_agent_as_tool():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.a2a import RemoteAgent
    from yaab.serve import get_fastapi_app

    server_agent = Agent("remote", model=TestModel("delegated result"))
    client = TestClient(get_fastapi_app(server_agent))

    async def transport(method, path, json):
        return client.request(method, path, json=json).json()

    remote = RemoteAgent("http://server", name="remote_helper", transport=transport)
    local_model = TestModel(custom_output="local-done", call_tools=["remote_helper"])
    local = Agent("local", model=local_model, tools=[remote])
    result = await local.run("use the remote helper")
    assert result.output == "local-done"


@pytest.mark.asyncio
async def test_mcp_client_lists_and_calls_tools():
    """Drive MCPClient over a fake in-process JSON-RPC transport."""
    from yaab.tools.mcp_client import MCPClient

    async def fake_server(request):
        method = request["method"]
        rid = request["id"]
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": rid, "result": {"capabilities": {}}}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                            },
                        }
                    ]
                },
            }
        if method == "tools/call":
            text = request["params"]["arguments"].get("text", "")
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": f"echo: {text}"}]},
            }
        return {"jsonrpc": "2.0", "id": rid, "error": {"message": "unknown"}}

    client = MCPClient.from_transport(fake_server)
    await client.start()
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "echo"

    from yaab.types import RunContext

    result = await tools[0].execute(RunContext(), text="hi")
    assert result == "echo: hi"


@pytest.mark.asyncio
async def test_token_streaming():
    agent = Agent("a", model=TestModel("one two three"))
    tokens = [t async for t in agent.stream("go")]
    assert "".join(tokens).strip() == "one two three"


def test_chat_stream_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.serve import get_fastapi_app

    agent = Agent("a", model=TestModel("hello streamed world"))
    client = TestClient(get_fastapi_app(agent))
    with client.stream("POST", "/chat/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "hello" in body
    assert "[DONE]" in body
