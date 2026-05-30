"""Model Context Protocol (MCP) tool adapter.

MCP is the open standard for agent-to-tool interop. A full MCP client (stdio /
HTTP transports) lives behind the optional ``mcp`` package; this adapter keeps
YAAB's side transport-agnostic by wrapping an already-discovered MCP tool
(name, description, input schema) plus an async ``caller`` that performs the
JSON-RPC ``tools/call``. This lets MCP tools satisfy the :class:`Tool` protocol
without pulling a hard dependency into the core.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..types import RunContext

CallFn = Callable[[str, dict[str, Any]], Awaitable[Any]]


class MCPTool:
    """A single MCP tool exposed through the YAAB :class:`Tool` protocol."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        caller: CallFn,
    ) -> None:
        self.name = name
        self.description = description
        self._input_schema = input_schema
        self._caller = caller

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._input_schema or {"type": "object", "properties": {}},
            },
        }

    async def execute(self, ctx: RunContext, **kwargs: Any) -> Any:
        return await self._caller(self.name, kwargs)


def mcp_toolset(descriptors: list[dict[str, Any]], caller: CallFn) -> list[MCPTool]:
    """Build :class:`MCPTool` objects from a server's ``tools/list`` response."""
    tools: list[MCPTool] = []
    for d in descriptors:
        tools.append(
            MCPTool(
                name=d["name"],
                description=d.get("description", ""),
                input_schema=d.get("inputSchema", d.get("input_schema", {})),
                caller=caller,
            )
        )
    return tools
