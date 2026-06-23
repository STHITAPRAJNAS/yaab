"""Recipe: interop — expose YAAB tools over MCP (Model Context Protocol).

``MCPServer`` serves any YAAB tools to MCP clients (Claude Desktop, IDEs, …). It
speaks JSON-RPC; here we list and call a tool in-process.

    python -m cookbook.recipes.interop
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import tool
from yaab.tools.mcp_server import MCPServer


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def run() -> dict:
    server = MCPServer([add], name="calculator")

    listed = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    names = [t["name"] for t in listed["result"]["tools"]]
    expect("add" in names, f"expected the add tool to be exposed, got {names}")

    called = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
        }
    )
    text = called["result"]["content"][0]["text"]
    expect("5" in text, f"expected add(2,3)=5, got {text!r}")
    return {"tools": names, "result": text}


if __name__ == "__main__":
    print(asyncio.run(run()))
