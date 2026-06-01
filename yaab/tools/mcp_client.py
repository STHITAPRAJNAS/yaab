"""MCP client — connect to a Model Context Protocol server and import its tools.

Supports a stdio transport (spawn an MCP server subprocess and speak
line-delimited JSON-RPC) and an injectable async transport for HTTP/SSE servers
or tests. Discovered tools are returned as :class:`~yaab.tools.mcp.MCPTool`
objects that satisfy YAAB's :class:`Tool` protocol, so an MCP server's whole
toolset drops straight into an agent.

    client = MCPClient.stdio(["python", "my_mcp_server.py"])
    await client.start()
    agent = Agent("a", tools=await client.list_tools())
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from .mcp import MCPTool

# A transport sends a JSON-RPC request dict and returns the response dict.
RPCTransport = Callable[[dict], Awaitable[dict]]


class MCPClient:
    """A minimal JSON-RPC 2.0 client for MCP servers."""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, transport: RPCTransport, *, sampling_handler: Any | None = None) -> None:
        self._transport = transport
        self._id = 0
        self._initialized = False
        self._proc: asyncio.subprocess.Process | None = None
        #: Handler for server-initiated sampling/createMessage requests (lets a
        #: server delegate completions to this client's model).
        self.sampling_handler = sampling_handler

    # --- constructors --------------------------------------------------
    @classmethod
    def from_transport(
        cls, transport: RPCTransport, *, sampling_handler: Any | None = None
    ) -> MCPClient:
        """Build a client over a custom async transport (HTTP/SSE, in-process)."""
        return cls(transport, sampling_handler=sampling_handler)

    @staticmethod
    def sampler_from_model(model: Any) -> Callable[[dict], Awaitable[dict]]:
        """Build an MCP sampling handler that runs ``model`` on the request.

        Returns an ``async (params) -> result`` callback in MCP
        ``sampling/createMessage`` shape, suitable for ``MCPServer(
        request_sampling=…)`` or ``MCPClient(sampling_handler=…)``.
        """
        from ..types import Message, Role

        async def _sample(params: dict) -> dict:
            mcp_messages = params.get("messages", [])
            messages = []
            sys = params.get("systemPrompt")
            if sys:
                messages.append(Message(role=Role.SYSTEM, content=str(sys)))
            for m in mcp_messages:
                content = m.get("content", {})
                text = content.get("text", "") if isinstance(content, dict) else str(content)
                role = Role.ASSISTANT if m.get("role") == "assistant" else Role.USER
                messages.append(Message(role=role, content=text))
            resp = await model.complete(messages)
            return {
                "role": "assistant",
                "content": {"type": "text", "text": resp.content},
                "model": getattr(model, "name", "yaab"),
            }

        return _sample

    async def handle(self, request: dict) -> dict:
        """Handle a server-initiated JSON-RPC request (e.g. sampling/createMessage).

        For bidirectional transports where the server calls back into the client.
        """
        method = request.get("method")
        rid = request.get("id")
        if method == "sampling/createMessage" and self.sampling_handler is not None:
            result = self.sampling_handler(request.get("params") or {})
            if asyncio.iscoroutine(result):
                result = await result
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    @classmethod
    def stdio(cls, command: list[str]) -> MCPClient:
        """Build a client that will spawn ``command`` as a stdio MCP server."""
        client = cls.__new__(cls)
        client._id = 0
        client._initialized = False
        client._proc = None
        client._command = command  # type: ignore[attr-defined]
        client._transport = client._stdio_transport  # type: ignore[assignment]
        return client

    # --- lifecycle -----------------------------------------------------
    async def start(self) -> dict[str, Any]:
        """Spawn the subprocess (if stdio) and perform the MCP handshake."""
        if getattr(self, "_command", None) and self._proc is None:
            self._proc = await asyncio.create_subprocess_exec(
                *self._command,  # type: ignore[attr-defined]
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
        result = await self._call(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "yaab", "version": "0.1.0"},
            },
        )
        self._initialized = True
        return result

    async def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                await self._proc.wait()
            except ProcessLookupError:  # pragma: no cover
                pass
            self._proc = None

    async def __aenter__(self) -> MCPClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # --- RPC -----------------------------------------------------------
    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _call(self, method: str, params: dict | None = None) -> Any:
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        response = await self._transport(request)
        if "error" in response and response["error"]:
            raise RuntimeError(f"MCP error from '{method}': {response['error']}")
        return response.get("result")

    async def _stdio_transport(self, request: dict) -> dict:  # pragma: no cover - needs subprocess
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write((json.dumps(request) + "\n").encode())
        await self._proc.stdin.drain()
        line = await self._proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection")
        return json.loads(line.decode())

    # --- tools ---------------------------------------------------------
    async def list_tools(self) -> list[MCPTool]:
        """Discover the server's tools as YAAB :class:`MCPTool` objects."""
        if not self._initialized:
            await self.start()
        result = await self._call("tools/list", {})
        descriptors = result.get("tools", []) if isinstance(result, dict) else []

        async def caller(name: str, arguments: dict) -> Any:
            out = await self._call("tools/call", {"name": name, "arguments": arguments})
            return _flatten_content(out)

        from .mcp import mcp_toolset

        return mcp_toolset(descriptors, caller)

    # --- resources (MCP beyond tools; Strands #151) --------------------
    async def list_resources(self) -> list[dict[str, Any]]:
        """List the server's resources (uri + metadata descriptors)."""
        if not self._initialized:
            await self.start()
        result = await self._call("resources/list", {})
        return result.get("resources", []) if isinstance(result, dict) else []

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI; returns flattened text when possible."""
        out = await self._call("resources/read", {"uri": uri})
        if isinstance(out, dict) and "contents" in out:
            texts = [c.get("text", "") for c in out["contents"] if "text" in c]
            if texts:
                return "\n".join(texts)
        return out

    # --- prompts (MCP beyond tools) ------------------------------------
    async def list_prompts(self) -> list[dict[str, Any]]:
        """List the server's prompt templates."""
        if not self._initialized:
            await self.start()
        result = await self._call("prompts/list", {})
        return result.get("prompts", []) if isinstance(result, dict) else []

    async def get_prompt(self, name: str, arguments: dict | None = None) -> Any:
        """Fetch a rendered prompt by name."""
        return await self._call("prompts/get", {"name": name, "arguments": arguments or {}})


def _flatten_content(result: Any) -> Any:
    """Reduce an MCP ``tools/call`` result to plain text where possible."""
    if isinstance(result, dict) and "content" in result:
        texts = [c.get("text", "") for c in result["content"] if c.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return result


__all__ = ["MCPClient"]
