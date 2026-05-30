# Agents

`Agent[Deps, Output]` is the typed unit of work — a model, instructions, tools,
and an output contract. It is generic over a **dependency type** (`Deps`, for
dependency injection) and an **output type** (`Output`, for validated results).

```python
from yaab import Agent

agent = Agent(
    name="assistant",
    model="openai/gpt-4o",              # str (LiteLLM id) or a ModelProvider
    instructions="You are helpful.",     # str or Callable[[RunContext], str]
    tools=[...],                          # functions or Tool objects
    deps_type=MyDeps,                     # DI payload type
    output_type=MyModel,                  # str (default) or a Pydantic model / type
    guardrails=[...],                     # optional per-agent scanners
    skills=[...],                         # reusable bundles (see prompts-skills.md)
    registry_id="assistant",             # link to the governance registry
    max_steps=8,                          # tool-loop budget
    output_retries=2,                     # reflection/retry on output validation
)
```

## Running

```python
result = await agent.run("prompt", deps=..., session_id="s1", identity="alice")
result = agent.run_sync("prompt")                      # sync wrapper
async for token in agent.stream("prompt"): ...         # token streaming
async for event in agent._get_runner().run_stream(agent, "prompt"): ...
```

`identity` flows into the run context and the audit log; `session_id` enables
durable, multi-turn conversations (see [State](state.md)).

## Dynamic instructions

Instructions can be a callable that builds the system prompt from the run
context — useful for injecting per-request facts:

```python
def instructions(ctx) -> str:
    return f"You are helping {ctx.deps.user_name}. Be brief."

agent = Agent("a", model="openai/gpt-4o", instructions=instructions, deps_type=Deps)
```

## Registering tools after construction

```python
agent = Agent("a", model="openai/gpt-4o")

@agent.tool
def now() -> str:
    """Return the current time."""
    return "2026-01-01T00:00:00Z"
```

## Agent as a tool

Any agent (or workflow agent) can be exposed as a tool to another, enabling
hierarchical delegation:

```python
specialist = Agent("specialist", model="openai/gpt-4o", instructions="You are a tax expert.")
generalist = Agent("generalist", model="openai/gpt-4o", tools=[specialist.as_tool()])
```

## The Runner

`Agent.run` delegates to a `Runner`, which owns the services, the plugin chain,
and (optionally) governance. Construct one explicitly to share configuration:

```python
from yaab import Runner
from yaab.sessions import SQLiteSessionService

runner = Runner(session_service=SQLiteSessionService("sessions.db"))
result = await runner.run(agent, "hi", session_id="s1")
```

See [Models](models.md), [Tools](tools.md), and [Governance](governance.md) for
the pieces an agent composes.
