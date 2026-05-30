# Get started

This is the guided path from zero to a governed, deployed agent — the same arc
Google ADK users will recognize (install → first agent → tools → memory/RAG →
multi-agent → serve → govern → deploy), but Python-first and provider-neutral.

## 1. Install

```bash
pip install yaab                 # SDK + high-performance async-first Python core
pip install 'yaab[rust]'         # + the prebuilt Rust performance core (optional)
pip install 'yaab[litellm]'      # talk to real models (OpenAI, Anthropic, Bedrock, …)
```

`pip install yaab` works everywhere with no build tooling. Add `[rust]` for the
accelerated core; YAAB auto-selects it and falls back transparently. Check:

```python
import yaab; print(yaab.BACKEND)   # "rust" or "python"
```

## 2. Your first agent

```python
from yaab import Agent

agent = Agent("assistant", model="openai/gpt-4o", instructions="Be concise.")
print(agent.run_sync("Say hello in one sentence.").output)
```

No API key yet? Develop fully offline with `TestModel`:

```python
from yaab import Agent
from yaab.testing import TestModel

agent = Agent("assistant", model=TestModel("Hi there!"))
assert agent.run_sync("hello").output == "Hi there!"
```

## 3. Add tools

A tool is a typed function. The schema, validation, and description are derived
automatically.

```python
from yaab import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"It's sunny in {city}."

agent = Agent("weather", model="openai/gpt-4o", tools=[get_weather])
print(agent.run_sync("Weather in Paris?").output)
```

Or grab the built-in toolbox:

```python
from yaab.tools.builtin import default_toolset   # calculator, time, http_get, web_search
agent = Agent("a", model="openai/gpt-4o", tools=default_toolset())
```

## 4. Typed output

```python
from pydantic import BaseModel
from yaab import Agent

class Weather(BaseModel):
    city: str
    temp_c: int

agent = Agent("weather", model="openai/gpt-4o", output_type=Weather)
result = agent.run_sync("weather in Paris")
print(result.output.city, result.output.temp_c)   # validated, typed
```

## 5. Give it knowledge (RAG)

```python
from yaab import Agent, KnowledgeBase
from yaab.rag import load_directory

kb = KnowledgeBase()
kb.add(load_directory("./docs", glob="**/*.md"))      # pdf/html/csv/json too
agent = Agent("support", model="openai/gpt-4o", tools=[kb.as_tool()])
```

See [RAG](rag.md) for citations, rerankers, per-user isolation, and cloud vector
stores (pgvector/Aurora, OpenSearch, Chroma, Qdrant, Oracle).

## 6. Remember across sessions

```python
from yaab import Runner, SessionManager
from yaab.sessions import SQLiteSessionService

runner = Runner(session_service=SQLiteSessionService("sessions.db"))
await agent.run("My name is Alice.", session_id="s1")
await agent.run("What's my name?", session_id="s1")   # remembers
```

Swap in Postgres/Aurora or Redis for production — see
[Storage & backends](storage-backends.md).

## 7. Compose multiple agents

```python
from yaab import SequentialAgent, ParallelAgent

pipeline = SequentialAgent("etl", [extractor, transformer, loader])
panel = ParallelAgent("review", [legal, finance, risk])
```

Also `MapAgent`, `LoopAgent`, `Swarm`, and agent-as-tool — see
[Multi-agent](multi-agent.md).

## 8. Add governance (the differentiator)

```python
from yaab import Runner
from yaab.governance import GovernanceService, GovernanceMode, AgentCard, RiskTier

gov = GovernanceService(mode=GovernanceMode.ENFORCING)
gov.registry.register(AgentCard(agent_id="support", name="Support", risk_tier=RiskTier.LIMITED))
runner = Runner(governance=gov)   # registry gate + guardrails + tamper-evident audit
```

See [Governance](governance.md) for the lifecycle FSM, policy/guardrail engine,
tool authorization & approval, audit lineage, and compliance reports.

## 9. Serve & deploy

```python
from yaab.serve import fastapi_server_app
app = fastapi_server_app(agent)        # /run, /run/stream, /a2a/tasks, agent card
```

```bash
yaab web mymodule:agent                # local browser playground
yaab serve mymodule:agent              # HTTP + A2A
```

Containerize and deploy to Cloud Run / Fargate / Lambda / K8s — see
[Deployment](DEPLOYMENT.md) and [Serving & auth](serving.md).

## Where next

- [Agents](agents.md) — the full `Agent` surface.
- [Storage & backends](storage-backends.md) — sessions, memory, vector stores.
- [Extending YAAB](extending.md) — add models, tools, stores, metrics, mappers.
- [Comparison & gaps](COMPARISON.md) — how YAAB relates to ADK/LangGraph/etc.
