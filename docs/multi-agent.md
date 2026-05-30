# Multi-agent orchestration

YAAB ships the common multi-agent topologies as **workflow agents**. Each one
composes other agents and exposes the same `run` / `run_sync` / `as_tool`
surface as a plain `Agent`, so they nest arbitrarily and drop into tools,
graphs, and servers. Usage (tokens + cost) is rolled up across all sub-agents.

## Agent-as-Tool (delegation)

The simplest pattern: one agent calls another as a tool.

```python
from yaab import Agent

researcher = Agent("researcher", model="openai/gpt-4o", instructions="Find facts.")
writer = Agent("writer", model="openai/gpt-4o", tools=[researcher.as_tool(name="research")])
```

## Sequential pipeline

Run agents in order, piping each output into the next prompt.

```python
from yaab import SequentialAgent

pipeline = SequentialAgent("etl", [extractor, transformer, loader])
result = await pipeline.run("process this document")
# result.output is the last agent's output
```

## Parallel fan-out

Run agents concurrently on the same input; the output is a `name → result` map.

```python
from yaab import ParallelAgent

panel = ParallelAgent("panel", [legal, finance, risk])
result = await panel.run("Review this contract.")
print(result.output["legal"], result.output["finance"], result.output["risk"])
```

## Map fan-out

Run one agent across many inputs concurrently; the output is the list of
results. Give an explicit list, or derive inputs from the prompt with
`map_inputs`. `max_concurrency` bounds simultaneous runs.

```python
from yaab import MapAgent

summarize = MapAgent("summarize", summarizer, max_concurrency=4)
results = await summarize.run([doc1, doc2, doc3])     # -> [summary1, summary2, summary3]

# or derive inputs:
per_line = MapAgent("classify", classifier, map_inputs=lambda text: text.splitlines())
```

## Loop until done

Re-run an agent, feeding its output back, until a condition or a cap.

```python
from yaab import LoopAgent

refiner = LoopAgent(
    "refiner", drafting_agent,
    max_iterations=5,
    until=lambda out: "FINAL" in out,
)
```

A `SequentialAgent` can also stop early via `stop_when`:

```python
from yaab import SequentialAgent

pipeline = SequentialAgent("triage", [classify, escalate, resolve],
                           stop_when=lambda out: "RESOLVED" in str(out))
```

## Swarm (autonomous hand-off)

Peer agents that hand off to whoever is best suited. The swarm augments each
member with `handoff_to_<peer>` tools; when an agent calls one, the swarm
continues the task with that peer.

```python
from yaab import Swarm
from yaab.multiagent import SwarmState

triage = Agent("triage", model="openai/gpt-4o",
               instructions="Route the user to billing or tech support.")
billing = Agent("billing", model="openai/gpt-4o", instructions="Handle billing.")
tech = Agent("tech", model="openai/gpt-4o", instructions="Handle technical issues.")

support = Swarm("support", [triage, billing, tech], entry="triage", max_handoffs=4)
result = await support.run("I was double charged", deps=SwarmState())
```

## Composing patterns

Because workflow agents *are* agents, you can nest them:

```python
ParallelAgent("board", [
    SequentialAgent("legal_review", [intake, legal]),
    LoopAgent("budget", finance, max_iterations=3),
])
```

For explicit, durable, inspectable control flow (cycles, fan-in, HITL), reach
for the [graph engine](graph.md) instead.
