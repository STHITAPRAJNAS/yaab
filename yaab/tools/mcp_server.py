"""MCP server — expose YAAB tools to any MCP client.

The complement to :class:`~yaab.tools.mcp_client.MCPClient`: wrap a set of YAAB
tools (or an agent's tools) and answer MCP JSON-RPC requests, so other agents
and IDEs can discover and call them over the open standard (Strands #151 asks
for this on the other side).

The server is transport-agnostic: :meth:`MCPServer.handle` takes one JSON-RPC
request dict and returns the response dict. Wire it to stdio, HTTP, or anything
else; an in-process handler is exactly what tests (and the YAAB MCPClient) drive.
"""

from __future__ import annotations

from typing import Any

from ..types import RunContext

PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """Serve a list of YAAB tools over the MCP JSON-RPC protocol."""

    def __init__(self, tools: list[Any], *, name: str = "yaab", version: str = "0.1.0") -> None:
        self.tools = {t.name: t for t in tools}
        self.name = name
        self.version = version

    @classmethod
    def from_agent(cls, agent: Any, **kwargs: Any) -> MCPServer:
        """Expose an agent's tools as an MCP server."""
        return cls(list(agent.tools), name=kwargs.pop("name", agent.name), **kwargs)

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process one JSON-RPC request and return the response dict."""
        method = request.get("method")
        rid = request.get("id")
        params = request.get("params") or {}

        try:
            result = await self._dispatch(method, params)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except _RpcError as err:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": err.code, "message": str(err)}}
        except Exception as exc:  # noqa: BLE001 - surface as a JSON-RPC error
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(exc)}}

    async def _dispatch(self, method: str | None, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": self.version},
            }
        if method == "tools/list":
            return {"tools": [self._descriptor(t) for t in self.tools.values()]}
        if method == "tools/call":
            return await self._call_tool(params)
        raise _RpcError(-32601, f"method not found: {method}")

    @staticmethod
    def _descriptor(tool: Any) -> dict[str, Any]:
        schema = tool.schema()
        fn = schema.get("function", {})
        return {
            "name": tool.name,
            "description": getattr(tool, "description", "") or fn.get("description", ""),
            "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
        }

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = self.tools.get(name)
        if tool is None:
            raise _RpcError(-32602, f"unknown tool: {name}")
        result = await tool.execute(RunContext(), **args)
        return {"content": [{"type": "text", "text": _as_text(result)}]}


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


__all__ = ["MCPServer"]
