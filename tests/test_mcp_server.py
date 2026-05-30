"""Tests for the MCP server and client resources/prompts (Tier 3a)."""

from __future__ import annotations

import pytest

from yaab import Agent, tool
from yaab.models.test_model import TestModel
from yaab.tools.mcp_client import MCPClient
from yaab.tools.mcp_server import MCPServer
from yaab.types import RunContext


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.mark.asyncio
async def test_mcp_server_lists_and_calls_tools_directly():
    server = MCPServer([add], name="calc")
    init = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["name"] == "calc"

    listed = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in listed["result"]["tools"]]
    assert "add" in names

    called = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
        }
    )
    assert called["result"]["content"][0]["text"] == "5"


@pytest.mark.asyncio
async def test_mcp_server_unknown_method_and_tool():
    server = MCPServer([add])
    bad_method = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})
    assert bad_method["error"]["code"] == -32601
    bad_tool = await server.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "ghost"}}
    )
    assert bad_tool["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_mcp_server_client_roundtrip():
    """The YAAB MCPClient drives a YAAB MCPServer in-process (full interop)."""
    server = MCPServer([add], name="calc")

    async def transport(request):
        return await server.handle(request)

    client = MCPClient.from_transport(transport)
    await client.start()
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["add"]
    result = await tools[0].execute(RunContext(), a=4, b=5)
    assert result == "9"


@pytest.mark.asyncio
async def test_mcp_server_from_agent():
    agent = Agent("a", model=TestModel("x"), tools=[add])
    server = MCPServer.from_agent(agent)
    listed = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "add" in [t["name"] for t in listed["result"]["tools"]]


@pytest.mark.asyncio
async def test_mcp_client_resources_and_prompts():
    """Client resources/read and prompts/get over a fake server."""

    async def fake(request):
        m, rid = request["method"], request["id"]
        if m == "initialize":
            return {"jsonrpc": "2.0", "id": rid, "result": {"capabilities": {}}}
        if m == "resources/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"resources": [{"uri": "file://x"}]}}
        if m == "resources/read":
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"contents": [{"uri": "file://x", "text": "hello"}]},
            }
        if m == "prompts/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"prompts": [{"name": "greet"}]}}
        if m == "prompts/get":
            return {"jsonrpc": "2.0", "id": rid, "result": {"messages": [{"role": "user"}]}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"message": "no"}}

    client = MCPClient.from_transport(fake)
    await client.start()
    assert (await client.list_resources())[0]["uri"] == "file://x"
    assert await client.read_resource("file://x") == "hello"
    assert (await client.list_prompts())[0]["name"] == "greet"
    got = await client.get_prompt("greet")
    assert "messages" in got
