# Quickstart

## Your first agent (three lines)

```python
from yaab import Agent

agent = Agent("assistant", model="openai/gpt-4o", instructions="Be concise.")
print(agent.run_sync("Say hello in one sentence.").output)
```

`run_sync` is the synchronous wrapper; in async code use `await agent.run(...)`.

## No API key? Run fully offline

Swap any model string for `TestModel` and the same code runs without a network —
that's how YAAB's own test suite and `examples/` run in CI. (Snippets that show a
provider model string like `"openai/gpt-4o"` need that provider's API key.)

```python
from yaab import Agent
from yaab.testing import TestModel

agent = Agent("assistant", model=TestModel("Hi there!"))
assert agent.run_sync("hello").output == "Hi there!"
```

## Add a tool

Tools are plain typed functions. The JSON schema is generated from the signature,
arguments are validated by Pydantic, and the description comes from the docstring.

```python
from yaab import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"It's sunny in {city}."

agent = Agent("weather", model="openai/gpt-4o", tools=[get_weather])
print(agent.run_sync("What's the weather in Paris?").output)
```

## Typed, validated output

Pass a Pydantic model as `output_type`; YAAB validates the model's response and
retries with the validation error on failure.

```python
from pydantic import BaseModel
from yaab import Agent

class Weather(BaseModel):
    city: str
    temp_c: int

agent = Agent("weather", model="openai/gpt-4o", output_type=Weather)
result = agent.run_sync("weather in Paris")
print(result.output.city, result.output.temp_c)   # typed access
```

## Dependency injection

Give the agent a typed `deps` object; tools receive it through `RunContext`:

```python
from dataclasses import dataclass
from yaab import Agent, RunContext, tool

@dataclass
class Deps:
    db: object

@tool
def lookup(ctx: RunContext, user_id: str) -> str:
    """Look up a user."""
    return ctx.deps.db.get(user_id)

agent = Agent("lookup", model="openai/gpt-4o", tools=[lookup], deps_type=Deps)
agent.run_sync("find user 42", deps=Deps(db=my_db))
```

## What you get back

`run`/`run_sync` return a `RunResult`:

```python
result = agent.run_sync("hi")
result.output      # the (typed) output
result.messages    # the full message history
result.usage       # tokens + cost (Usage)
result.events      # the semantic event stream (see streaming-events.md)
```

## Next

- [Agents](agents.md) for the full `Agent` surface.
- [State](state.md) for sessions, memory, and artifacts.
- [Governance](governance.md) to register, gate, audit, and prove compliance.
