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

## Per-agent callbacks

`before_agent` / `after_agent` fire around an agent's *own* loop — so in a
composition (Sequential, Parallel, Swarm, …) each child gets its own pair when it
runs, not once at the top. Both may be sync or async.

```python
from yaab import Agent
from yaab.testing import TestModel

events: list[str] = []
agent = Agent(
    "a",
    model=TestModel("hi"),
    before_agent=lambda ag, prompt: events.append(f"before:{ag.name}:{prompt}"),
    after_agent=lambda ag, result: events.append(f"after:{ag.name}:{result.output}"),
)
# before_agent(agent, prompt) runs first; after_agent(agent, result) runs last.
```

## Filtering context to what's relevant

A `context_strategy` rewrites the message history before each model call.
`RelevanceFilter` keeps only the turns relevant to the latest message (system
messages and the latest user turn are always kept), alongside the built-in
truncate/summarize strategies.

```python
from yaab import Agent, RelevanceFilter

agent = Agent(
    "a",
    model="openai/gpt-4o",
    context_strategy=RelevanceFilter(min_score=0.15),   # drop low-relevance history
)
```

The default scorer is keyword overlap; inject any `(query, text) -> float` (e.g.
an embedding similarity) for semantic relevance:

```python
from yaab import RelevanceFilter

def embed_score(query: str, text: str) -> float:
    ...                       # your similarity in [0, 1]

strategy = RelevanceFilter(min_score=0.2, scorer=embed_score)
```

## Declarative agents (YAML / dict)

`agent_from_yaml` / `agent_from_dict` build an agent from a spec. `output_type`
resolves by name — the built-in scalars `str` / `int` / `float` / `bool` need no
registration; any Pydantic model is referenced by the name it was registered
under, so a declarative agent can emit a typed object, not just text.

```python
from pydantic import BaseModel
from yaab import agent_from_dict, register_component
from yaab.testing import TestModel

class Ticket(BaseModel):
    title: str
    priority: int

register_component("output_type", "Ticket", lambda: Ticket)   # resolve the name

agent = agent_from_dict({
    "name": "tk",
    "model": TestModel('{"title": "Reset password", "priority": 1}'),
    "output_type": "Ticket",        # -> agent.output_type is Ticket
})
```

Callbacks and plugins wire by registered name too — `callbacks: {before_agent:
…, after_agent: …}` resolves `callback` components onto the agent's hooks, and
`plugins: [name]` resolves `plugin` components onto its Runner. An unknown name is
a clear load-time error:

```python
from yaab import agent_from_dict, register_component
from yaab.testing import TestModel

register_component("callback", "audit", lambda: (lambda ag, prompt: None))
agent = agent_from_dict({
    "name": "a",
    "model": TestModel("hi"),
    "callbacks": {"before_agent": "audit"},
})
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
