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

from collections.abc import Callable
from typing import Any

from ..types import RunContext

PROTOCOL_VERSION = "2024-11-05"


class MCPResource:
    """A resource an :class:`MCPServer` exposes (MCP ``resources/*``).

    Provide static ``text`` or a ``loader`` callable computed on read (sync or
    async). ``mime_type`` defaults to plain text.
    """

    def __init__(
        self,
        *,
        uri: str,
        name: str,
        text: str | None = None,
        loader: Callable[[], Any] | None = None,
        description: str = "",
        mime_type: str = "text/plain",
    ) -> None:
        if text is None and loader is None:
            raise ValueError("MCPResource needs either text or a loader")
        self.uri = uri
        self.name = name
        self.text = text
        self.loader = loader
        self.description = description
        self.mime_type = mime_type

    def descriptor(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }

    async def read(self) -> str:
        import inspect

        if self.loader is not None:
            value = self.loader()
            if inspect.isawaitable(value):
                value = await value
            return _as_text(value)
        return self.text or ""


class MCPPrompt:
    """A prompt template an :class:`MCPServer` exposes (MCP ``prompts/*``).

    ``template`` is ``str.format``-rendered with the call arguments.
    """

    def __init__(
        self,
        *,
        name: str,
        template: str,
        description: str = "",
        arguments: list[dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.template = template
        self.description = description
        self.arguments = arguments or []

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }

    def render(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            text = self.template.format(**(arguments or {}))
        except KeyError as exc:
            raise _RpcError(-32602, f"missing prompt argument: {exc}") from exc
        return {
            "description": self.description,
            "messages": [
                {"role": "user", "content": {"type": "text", "text": text}},
            ],
        }


class MCPServer:
    """Serve YAAB tools (and optionally resources/prompts) over MCP JSON-RPC."""

    def __init__(
        self,
        tools: list[Any],
        *,
        name: str = "yaab",
        version: str = "0.1.0",
        resources: list[MCPResource] | None = None,
        prompts: list[MCPPrompt] | None = None,
        request_sampling: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.tools = {t.name: t for t in tools}
        self.name = name
        self.version = version
        self.resources = {r.uri: r for r in (resources or [])}
        self.prompts = {p.name: p for p in (prompts or [])}
        #: Callback to ask the client to run a completion (MCP sampling); usually
        #: wired to the client's model via ``MCPClient.sampler_from_model``.
        self.request_sampling = request_sampling

    async def sample(self, messages: list[dict[str, Any]], **options: Any) -> str:
        """Request a completion from the client's model (MCP sampling).

        Lets a server-side tool delegate reasoning to the client's LLM instead of
        hard-coding its own. Raises if no sampler is wired.
        """
        if self.request_sampling is None:
            raise RuntimeError(
                "MCP sampling is not available: construct MCPServer(request_sampling=…) "
                "(e.g. MCPClient.sampler_from_model(model))."
            )
        import inspect

        result = self.request_sampling({"messages": messages, **options})
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            content = result.get("content", {})
            return content.get("text", "") if isinstance(content, dict) else str(content)
        return str(result)

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
            caps: dict[str, Any] = {"tools": {}}
            if self.resources:
                caps["resources"] = {}
            if self.prompts:
                caps["prompts"] = {}
            if self.request_sampling is not None:
                caps["sampling"] = {}
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": caps,
                "serverInfo": {"name": self.name, "version": self.version},
            }
        if method == "tools/list":
            return {"tools": [self._descriptor(t) for t in self.tools.values()]}
        if method == "tools/call":
            return await self._call_tool(params)
        if method == "resources/list":
            return {"resources": [r.descriptor() for r in self.resources.values()]}
        if method == "resources/read":
            return await self._read_resource(params)
        if method == "prompts/list":
            return {"prompts": [p.descriptor() for p in self.prompts.values()]}
        if method == "prompts/get":
            return self._get_prompt(params)
        raise _RpcError(-32601, f"method not found: {method}")

    async def _read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        resource = self.resources.get(uri)
        if resource is None:
            raise _RpcError(-32602, f"unknown resource: {uri}")
        text = await resource.read()
        return {
            "contents": [
                {"uri": uri, "mimeType": resource.mime_type, "text": text},
            ]
        }

    def _get_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        prompt = self.prompts.get(name)
        if prompt is None:
            raise _RpcError(-32602, f"unknown prompt: {name}")
        return prompt.render(params.get("arguments") or {})

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


__all__ = ["MCPServer", "MCPResource", "MCPPrompt"]
