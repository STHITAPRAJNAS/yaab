# Tools

A tool exposes a JSON schema (for the model) and an async `execute` (for the
runtime). YAAB supports typed function tools, agent-as-tool, MCP tools, and
remote A2A agents — all satisfying the same `Tool` protocol.

## Typed function tools

Decorate a typed function with `@tool`. The parameter schema is generated from
the type hints, arguments are validated by Pydantic, and the description comes
from the docstring.

```python
from yaab import tool

@tool
def search(query: str, limit: int = 5) -> list[str]:
    """Search the knowledge base."""
    return [...]
```

Tools may be sync or async, and may optionally take a `RunContext` first
parameter (named `ctx`) for dependency injection — it is excluded from the
model-facing schema:

```python
from yaab import RunContext, tool

@tool
def charge(ctx: RunContext, amount: int) -> str:
    """Charge the current customer."""
    return ctx.deps.payments.charge(ctx.deps.customer_id, amount)
```

Bad arguments raise a `ToolError`, which the runtime feeds back to the model as a
tool result so it can correct itself rather than crashing the run.

## Custom Tool objects

Implement the protocol directly for full control:

```python
from yaab.tools import Tool   # typing.Protocol

class MyTool:
    name = "my_tool"
    description = "Does a thing."
    def schema(self) -> dict: ...
    async def execute(self, ctx, **kwargs): ...
```

## Controlling tool use (`tool_choice`)

Set `tool_choice` on the agent to steer whether/which tools the model calls:

```python
Agent("a", model="openai/gpt-4o", tools=[...], tool_choice="auto")      # default
Agent("a", model="openai/gpt-4o", tools=[...], tool_choice="required")  # must call some tool
Agent("a", model="openai/gpt-4o", tools=[...], tool_choice="none")      # answer without tools
Agent("a", model="openai/gpt-4o", tools=[search], tool_choice="search") # force this tool
```

A bare tool name is expanded to the provider's
`{"type": "function", "function": {"name": ...}}` form. Use `"required"` to force
the agent to call a tool before producing a final answer.

## Repairing malformed tool args

When a model emits tool-call arguments that don't match the schema, a plugin can
coerce/repair them *before* validation via the `repair_tool_args` hook:

```python
from yaab.plugins import Plugin

class CoerceInts(Plugin):
    async def repair_tool_args(self, ctx, agent, tool, args):
        if tool == "add":
            return {k: int(v) for k, v in args.items()}   # "2" -> 2
        return None   # leave unchanged

runner = Runner(plugins=[CoerceInts()])
```

If repair isn't enough, invalid args still raise a `ToolError` that is fed back
to the model as a tool result so it can retry.

## Agent as a tool

```python
sub = Agent("researcher", model="openai/gpt-4o")
main = Agent("writer", model="openai/gpt-4o", tools=[sub.as_tool(name="research")])
```

## MCP tools

Import an MCP server's whole toolset (see [Interop](interop.md)):

```python
from yaab.tools.mcp_client import MCPClient

client = MCPClient.stdio(["python", "my_mcp_server.py"])
await client.start()
agent = Agent("a", model="openai/gpt-4o", tools=await client.list_tools())
```

## Remote A2A agents as tools

A `RemoteAgent` is also a tool, so a local agent can delegate to a remote one:

```python
from yaab.a2a import RemoteAgent

remote = RemoteAgent("https://other-service", name="billing")
agent = Agent("a", model="openai/gpt-4o", tools=[remote])
```

## Coercion

`Agent(tools=[...])` accepts a mix of plain functions and `Tool` objects;
functions are wrapped in `FunctionTool` automatically.
