"""Interop depth (Phase D): MCP server resources/prompts + A2A task polling.

The MCP *client* already consumes resources/prompts; this adds the *server* side
so YAAB can serve them, and round-trips client<->server. A2A gains a
poll-until-complete helper for long-running remote tasks.
"""

from __future__ import annotations

import pytest

from yaab import tool
from yaab.tools.mcp_client import MCPClient
from yaab.tools.mcp_server import MCPPrompt, MCPResource, MCPServer


def _client_for(server: MCPServer) -> MCPClient:
    async def transport(req):
        return await server.handle(req)

    return MCPClient.from_transport(transport)


# --- MCP server resources ----------------------------------------------
@pytest.mark.asyncio
async def test_server_lists_and_reads_resources():
    server = MCPServer(
        [],
        resources=[
            MCPResource(uri="file://readme", name="readme", text="hello docs"),
        ],
    )
    client = _client_for(server)
    await client.start()
    resources = await client.list_resources()
    assert any(r["uri"] == "file://readme" for r in resources)
    content = await client.read_resource("file://readme")
    assert content == "hello docs"


@pytest.mark.asyncio
async def test_server_resource_from_callable():
    server = MCPServer(
        [], resources=[MCPResource(uri="dyn://now", name="now", loader=lambda: "computed")]
    )
    client = _client_for(server)
    await client.start()
    assert await client.read_resource("dyn://now") == "computed"


@pytest.mark.asyncio
async def test_server_unknown_resource_errors():
    server = MCPServer([], resources=[])
    client = _client_for(server)
    await client.start()
    with pytest.raises(RuntimeError):
        await client.read_resource("nope://x")


# --- MCP server prompts ------------------------------------------------
@pytest.mark.asyncio
async def test_server_lists_and_gets_prompts():
    server = MCPServer(
        [],
        prompts=[
            MCPPrompt(
                name="greet",
                description="greeting",
                template="Hello {who}, welcome!",
                arguments=[{"name": "who", "required": True}],
            )
        ],
    )
    client = _client_for(server)
    await client.start()
    prompts = await client.list_prompts()
    assert any(p["name"] == "greet" for p in prompts)
    rendered = await client.get_prompt("greet", {"who": "Alice"})
    text = rendered["messages"][0]["content"]["text"]
    assert "Hello Alice, welcome!" in text


@pytest.mark.asyncio
async def test_server_capabilities_advertise_resources_and_prompts():
    server = MCPServer(
        [],
        resources=[MCPResource(uri="u", name="n", text="t")],
        prompts=[MCPPrompt(name="p", template="x")],
    )
    init = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    caps = init["result"]["capabilities"]
    assert "resources" in caps and "prompts" in caps


@pytest.mark.asyncio
async def test_tools_still_work_alongside_resources():
    @tool
    def add(a: int, b: int) -> int:
        """add"""
        return a + b

    server = MCPServer([add], resources=[MCPResource(uri="u", name="n", text="t")])
    client = _client_for(server)
    await client.start()
    tools = await client.list_tools()
    from yaab.types import RunContext

    assert str(await tools[0].execute(RunContext(), a=2, b=3)) == "5"


# --- A2A poll-until-complete -------------------------------------------
@pytest.mark.asyncio
async def test_a2a_poll_until_complete():
    from yaab.a2a import RemoteAgent

    # A fake remote whose task is "working" twice, then "completed".
    states = ["working", "working", "completed"]
    calls = {"n": 0}

    async def transport(method, path, json):
        if path == "/.well-known/agent.json":
            return {"name": "remote"}
        if path.startswith("/a2a/tasks/"):
            i = min(calls["n"], len(states) - 1)
            calls["n"] += 1
            state = states[i]
            return {
                "id": "t1",
                "status": {"state": state},
                "artifacts": (
                    [{"parts": [{"text": "final answer"}]}] if state == "completed" else []
                ),
            }
        return {}

    remote = RemoteAgent("http://server", name="remote", transport=transport)
    task = await remote.poll_task("t1", interval=0.0, timeout=5)
    assert task["status"]["state"] == "completed"
    assert calls["n"] >= 3  # polled until terminal
