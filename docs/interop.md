# Interop: MCP & A2A

YAAB speaks the two open standards of the agent ecosystem out of the box:
**MCP** (agent-to-tool) and **A2A** (agent-to-agent).

## MCP (Model Context Protocol)

### Import an MCP server's tools

`MCPClient` connects to an MCP server (stdio subprocess or a custom transport),
performs the handshake, and returns the server's tools as YAAB tools.

```python
from yaab import Agent
from yaab.tools.mcp_client import MCPClient

client = MCPClient.stdio(["python", "weather_mcp_server.py"])
await client.start()
tools = await client.list_tools()

agent = Agent("a", model="openai/gpt-4o", tools=tools)
# ... use the agent ...
await client.close()
```

Use it as an async context manager to handle lifecycle:

```python
async with MCPClient.stdio(["my-mcp-server"]) as client:
    agent = Agent("a", model="openai/gpt-4o", tools=await client.list_tools())
```

### Custom transports (HTTP/SSE)

Provide an async `transport(request_dict) -> response_dict` for non-stdio servers:

```python
client = MCPClient.from_transport(my_http_jsonrpc_transport)
```

### Wrapping already-discovered tools

If you discovered tools yourself, wrap them with `mcp_toolset`:

```python
from yaab.tools.mcp import mcp_toolset
tools = mcp_toolset(descriptors, caller)   # caller: async (name, args) -> result
```

## A2A (Agent-to-Agent)

### Serve an agent as an A2A endpoint

`fastapi_server_app` exposes an A2A Agent Card and task endpoint
(see [Serving](serving.md)):

```
GET  /.well-known/agent.json   # discovery: the Agent Card
POST /a2a/tasks                # submit a task
```

```python
from yaab.serve import fastapi_server_app
app = fastapi_server_app(agent, base_url="https://my-service")
```

The Agent Card is generated from the governance registry entry and includes
risk tier, approval status, decision authority, and skills.

### Call a remote agent

`RemoteAgent` discovers a remote agent via its card and delegates tasks. It is
both an agent (has `run`) and a tool (satisfies the `Tool` protocol):

```python
from yaab.a2a import RemoteAgent

remote = RemoteAgent("https://billing-service", name="billing", auth_token="...")
card = await remote.fetch_card()
result = await remote.run("Refund order 123")

# Or hand it to a local agent as a delegatable tool:
local = Agent("concierge", model="openai/gpt-4o", tools=[remote])
```

Authentication uses a bearer token (`auth_token`); for OAuth 2.1 token exchange,
wire your IdP and pass the resulting access token. See [Serving](serving.md) for
the server-side auth schemes.
