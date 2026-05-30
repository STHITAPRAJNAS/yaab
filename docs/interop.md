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

### Resources & prompts (beyond tools)

The client also speaks MCP resources and prompts:

```python
resources = await client.list_resources()
text = await client.read_resource("file:///docs/policy.md")

prompts = await client.list_prompts()
rendered = await client.get_prompt("summarize", {"style": "brief"})
```

### Serve YAAB tools as an MCP server

Expose your agent's tools to other MCP clients (IDEs, other agents) with
`MCPServer`. It's transport-agnostic — `handle(request)` answers one JSON-RPC
message; wire it to stdio or HTTP.

```python
from yaab.tools.mcp_server import MCPServer

server = MCPServer.from_agent(agent)        # or MCPServer([tool_a, tool_b])
response = await server.handle(json_rpc_request)
```

Because both ends are YAAB, a `MCPServer` can be driven directly by an
`MCPClient` over an in-process transport — handy for testing and for embedding
one agent's tools into another.

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

Authentication uses a static bearer token (`auth_token`) or, for OAuth 2.1, a
`token_provider` callable that returns a fresh access token per request (wire it
to your IdP's token exchange/refresh):

```python
remote = RemoteAgent("https://billing", token_provider=lambda: idp.access_token())
```

### Long-running tasks: poll & stream

The server stores submitted tasks so clients can poll by id, and exposes a
streaming variant that emits `working` → `completed` status events:

```python
result = await remote.run("generate the report")
task = await remote.get_task(result.run_id)     # poll: {"status": {"state": "completed"}, ...}
```

```
POST /a2a/tasks         # submit, returns the completed task
GET  /a2a/tasks/{id}    # poll a task by id
POST /a2a/tasks/stream  # SSE: working → completed task-status events
```

### Hand back to the orchestrator

In a [`Swarm`](multi-agent.md#swarm-autonomous-hand-off), every member gets a
`handoff_to_<peer>` tool for *each* peer — including the entry/orchestrator
agent — so a specialist can return control once its part is done.
