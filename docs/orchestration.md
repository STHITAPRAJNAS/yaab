# The orchestration model

In YAAB **everything you run is an agent, and every way of composing agents is
itself an agent.** A single `Agent[Deps, Output]` runs a model-driven loop; when
one agent is not enough you reach for a *workflow agent* — `SequentialAgent`,
`ParallelAgent`, `MapAgent`, `LoopAgent`, `Swarm`, `RouterAgent`, or `Flow`. All
of them share one base, one `run` / `run_sync` / `as_tool` surface, one typed
`RunResult`, one shared `State`, and one rolled-up usage total — so they nest
inside each other without adapters, drop into tools and servers, and a value one
step writes is read by the next by key. There is no separate "graph framework"
bolted on the side: the durable engine (BSP supersteps, checkpointing, a
Rust-accelerated state fold) is what `Flow` lowers onto, and the rest of the
patterns are thin orchestrators over the same `Agent.run`.

## Which pattern do I reach for?

| Pattern | Use when | One-liner |
|---|---|---|
| **`sub_agents=`** | the *model* should decide which specialist handles a turn | `Agent("desk", sub_agents=[billing, tech])` |
| **agent-as-tool** | one agent should *call* another and use its answer | `Agent("writer", tools=[researcher.as_tool()])` |
| **`SequentialAgent`** | fixed steps run in order, each building on the last | `SequentialAgent("etl", [extract, transform, load])` |
| **`ParallelAgent`** | the same input is reviewed by several agents at once | `ParallelAgent("panel", [legal, finance, risk])` |
| **`MapAgent`** | one agent runs across many inputs concurrently | `MapAgent("summarize", summarizer)` |
| **`LoopAgent`** | one agent re-runs until a quality bar or a cap | `LoopAgent("refine", drafter, stop="state.score >= 0.9")` |
| **`Swarm`** | peer agents hand off to whoever is best suited | `Swarm("support", [triage, billing, tech])` |
| **`RouterAgent`** | exactly one of N branches runs, chosen with **zero model calls** | `RouterAgent.from_picker("r", pick, to={...})` |
| **`Flow`** | the *control flow itself* must be explicit, branchable, and durable | `Flow("refund").step(...).route(...).loop(...)` |

The rule of thumb: let the **model** route (`sub_agents`, agent-as-tool, `Swarm`)
when the decision needs judgment; let **code** route (`RouterAgent`, `Flow`) when
it must be deterministic, cheap, and auditable.

## sub_agents — model-driven delegation

List specialists as `sub_agents=` and YAAB injects a `transfer_to_agent` tool so
the model hands the conversation to the best-matching one by name.

```python
from yaab import Agent

billing = Agent("billing", instructions="Handle billing questions.")
tech = Agent("tech", instructions="Handle technical issues.")
desk = Agent("desk", instructions="Route the user.", sub_agents=[billing, tech])
```

## Agent-as-tool — explicit delegation

`agent.as_tool()` turns any agent (or workflow agent) into a tool another agent
can call and read the result of.

```python
from yaab import Agent

researcher = Agent("researcher", instructions="Find facts.")
writer = Agent("writer", tools=[researcher.as_tool(name="research")])
```

## SequentialAgent — run in order

Each child runs against one shared state; `writes="key"` captures a step's typed
output where the next step can read it (see [State](state.md)).

```python
from yaab import Agent, SequentialAgent

classify = Agent("classify", instructions="Classify the request.", writes="topic")
reply = Agent("reply", instructions="The topic is {topic}. Answer it.")
pipeline = SequentialAgent("triage", [classify, reply])
```

## ParallelAgent — fan out concurrently

All branches see the same input; the output is a `name -> result` map.

```python
from yaab import Agent, ParallelAgent

legal = Agent("legal", instructions="Review legal risk.")
finance = Agent("finance", instructions="Review financial terms.")
panel = ParallelAgent("panel", [legal, finance])
```

## MapAgent — one agent, many inputs

Run one agent across a list of inputs concurrently; `max_concurrency` bounds
simultaneous runs.

```python
from yaab import Agent, MapAgent

summarizer = Agent("summarizer", instructions="Summarize the text.")
summarize = MapAgent("summarize", summarizer, max_concurrency=4)
```

## LoopAgent — repeat until done

Re-run an agent over an accumulating state until `stop=` fires or the iteration
cap is hit. `stop=` is a [Condition](conditions.md) and can read state.

```python
from yaab import Agent, LoopAgent

drafter = Agent("drafter", instructions="Improve the draft; set state.score.")
refine = LoopAgent("refine", drafter, max_iterations=5, stop="state.score >= 0.9")
```

## Swarm — autonomous hand-off

Peers hand off to whoever is best suited; the swarm augments each member with
`handoff_to_<peer>` tools.

```python
from yaab import Agent, Swarm
from yaab.multiagent import SwarmState

triage = Agent("triage", instructions="Route to billing or tech.")
billing = Agent("billing", instructions="Handle billing.")
tech = Agent("tech", instructions="Handle technical issues.")
support = Swarm("support", [triage, billing, tech], entry="triage", max_handoffs=4)
```

## RouterAgent — deterministic exclusive choice

Exactly one branch runs, chosen by a plain-Python/expression picker — **zero
model calls**, fully auditable. Build it from labelled branches or a picker. See
[Conditions](conditions.md) for the full routing model.

```python
from yaab import Agent, RouterAgent

simple = Agent("simple", instructions="Answer briefly.")
deep = Agent("deep", instructions="Answer in depth.")
router = RouterAgent.from_picker(
    "router",
    lambda text, ctx: "deep" if len(text) > 100 else "simple",
    to={"simple": simple, "deep": deep},
)
```

## Flow — explicit, durable control flow

When the control flow itself must be inspectable and crash-proof — branches,
cycles, fan-out, and human pauses — reach for [Flow](flow.md). It is a builder
(`.step / .then / .route / .loop / .fan_out / .start_at / .returns`) that lowers
onto the durable engine.

```python
from yaab import Flow

flow = (
    Flow[None, str]("refund")
    .step("parse", fn=lambda state, ctx: {"amount": 50})
    .route(
        "parse",
        lambda state, ctx: "auto" if state["amount"] < 100 else "human",
        to={"auto": "execute", "human": "review"},
    )
    .step("execute", fn=lambda state, ctx: {"out": "auto"})
    .step("review", fn=lambda state, ctx: {"out": "human"})
    .then("execute", Flow.DONE).then("review", Flow.DONE)
    .start_at("parse").returns("out")
)
```

## They compose

Because workflow agents *are* agents, you nest them freely — and any one can be
exposed `.as_tool()`:

```python
from yaab import Agent, SequentialAgent, ParallelAgent, LoopAgent

intake = Agent("intake", instructions="Read the case.")
legal = Agent("legal", instructions="Legal review.")
finance = Agent("finance", instructions="Budget review.")

board = ParallelAgent("board", [
    SequentialAgent("legal_review", [intake, legal]),
    LoopAgent("budget", finance, max_iterations=3),
])
```

## Where to go next

* [Flow](flow.md) — the builder API, shared state, routing, loops, HITL pauses,
  and run history.
* [Conditions](conditions.md) — `when=` / `stop=` / `else=`, combinators, and
  `RouterAgent` routing.
* [Multi-agent](multi-agent.md) — the workflow-agent patterns in depth.
* [Graph orchestration](graph.md) — the `StateGraph` engine Flow lowers onto.
* [Human-in-the-loop](hitl.md) and [Durable runs](durable-runs.md) — pausing for
  a person and surviving restarts across replicas.
