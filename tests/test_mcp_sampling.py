"""MCP sampling (G3): a server can request an LLM completion from the client.

MCP's sampling/createMessage lets a server ask the client (which owns the model)
to run a completion. This wires the client's model as the server's sampler and
round-trips both the direct call and the JSON-RPC reverse request.
"""

from __future__ import annotations

import pytest

from yaab.models.test_model import TestModel
from yaab.tools.mcp_client import MCPClient
from yaab.tools.mcp_server import MCPServer


@pytest.mark.asyncio
async def test_server_sample_uses_injected_sampler():
    model = TestModel("sampled reply")
    server = MCPServer([], request_sampling=MCPClient.sampler_from_model(model))
    text = await server.sample([{"role": "user", "content": {"type": "text", "text": "hi"}}])
    assert text == "sampled reply"


@pytest.mark.asyncio
async def test_server_sample_without_sampler_raises():
    server = MCPServer([])
    with pytest.raises(RuntimeError):
        await server.sample([{"role": "user", "content": {"type": "text", "text": "hi"}}])


@pytest.mark.asyncio
async def test_client_handles_sampling_createmessage_request():
    # The client routes a server-initiated sampling/createMessage to its model.
    model = TestModel("client says hi")
    client = MCPClient.from_transport(
        lambda req: None, sampling_handler=MCPClient.sampler_from_model(model)
    )
    response = await client.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": {"type": "text", "text": "yo"}}]},
        }
    )
    text = response["result"]["content"]["text"]
    assert text == "client says hi"
    assert response["result"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_capabilities_advertise_sampling():
    server = MCPServer([], request_sampling=lambda params: None)
    init = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert "sampling" in init["result"]["capabilities"]
