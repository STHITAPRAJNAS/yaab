# Conditions — one decision concept everywhere

A *condition* is a small, composable boolean test that reads the run's shared
state. The same concept answers three questions, and it behaves **identically**
on every pattern, sub-agent list, and tool list:

| Keyword | Question | Phase | Sees |
|---|---|---|---|
| `when=` | *do I run?* | **before** a unit runs | `(input, state, ctx)` — False **skips** the unit |
| `stop=` | *does the pattern stop now?* | **after** a unit runs | `(output, state, ctx)` — True stops the enclosing pattern |
| `else_=` | *what runs instead?* | on skip or failure | a fallback **unit** |

Every condition is evaluated against one read-only view of the run's shared
[`State`](state.md) — the *same* view instruction rendering uses — so a condition
reads exactly the values a tool wrote, a branch produced, or an instruction
rendered. A condition can read state; it physically cannot mutate it.

## Two ways to write one

A condition is either a **Python callable** or a **safe expression string**, and
the two compile to the same test over the same data, so swapping one for the
other never changes behavior.

```python
from yaab import when

# A callable — full Python, gets (input) or (input, ctx):
big = when(lambda amount, ctx: amount >= ctx.deps.threshold)

# A sandboxed expression — references input / state / deps:
urgent = when('"urgent" in input')
gold = when('deps.tier == "gold"')
ready = when("state.score >= 0.9")
```

The expression sandbox allows comparisons, membership (`in`), attribute access on
`input` / `state` / `deps` / `output`, and `and` / `or` / `not` — but not
function calls, so a guard can never have a side effect. `output.x` is only
meaningful under a `stop=` (output-phase) call site; using it under a `when=`
guard is a load-time error, never a silent mis-read.

## Combinators: `&` `|` `~`

Conditions compose, and so do the named helpers (`and_`, `or_`, `not_`):

```python
from yaab import when, and_, or_, not_, failed, output_contains, state_ge

guard = when('"refund" in input') & ~when('deps.blocked')
either = or_(when("state.tier == 'gold'"), when("state.vip"))
done = output_contains("FINAL") | state_ge("score", 0.9)
clean = not_(failed())
```

Status helpers read the *terminal status* of a guarded unit, kept separate from
its output value so "this step failed" and "this step's output is bad" are
distinct, testable conditions: `failed()`, `timed_out()`, `ok()`, `skipped()`,
and `loop_exhausted()` (true when a loop hit its cap without `stop=` firing).

## `when=` / `stop=` / `else_=` on any pattern

Wrap any unit in a `Step` to attach guards; the same three keywords behave the
same on `SequentialAgent`, `ParallelAgent`, `MapAgent`, `LoopAgent`, and `Swarm`.

```python
from yaab import Agent, SequentialAgent, Step

classify = Agent("classify", instructions="Classify the request.")
escalate = Agent("escalate", instructions="Escalate to a human.")
resolve = Agent("resolve", instructions="Resolve it.")
fallback = Agent("fallback", instructions="Apologize and log.")

pipeline = SequentialAgent(
    "triage",
    [
        classify,
        Step(escalate, when='"angry" in input', else_=fallback),   # skip -> fallback
        resolve,
    ],
    stop='"RESOLVED" in output',     # output guard: stop the pipeline early
)
```

* `stop=` on `SequentialAgent` / `LoopAgent` / `Swarm` is the output guard checked
  after each child — `LoopAgent("refine", drafter, stop="state.score >= 0.9")`
  stops the loop the moment the quality bar is met.
* `when=` on a `ParallelAgent` branch excludes it *before* scheduling, so an
  excluded branch never runs and is absent from the result map.
* `when=` on the agent inside a `MapAgent` *filters* inputs — each derived input
  whose guard is false is dropped from the results.

## `when=` on tools and sub-agents

A guarded tool only runs when its `when=` is true; a guarded sub-agent transfer
is only offered to the model when its guard holds — the same condition concept,
the same read-only state view.

```python
from yaab import Agent, tool, Step
from yaab.tools import FunctionTool

def wire(amount: int) -> str:
    """Wire money."""
    return f"sent ${amount}"

# Gate a tool with a Step(when=...): only offered when the guard holds.
banker = Agent("banker", tools=[Step(FunctionTool(wire), when="deps.can_wire")])

# Gate a model-driven transfer to a sub-agent likewise:
specialist = Agent("specialist", instructions="Handle escalations.")
desk = Agent("desk", instructions="Front desk.",
             sub_agents=[Step(specialist, when='"escalate" in input')])
```

## RouterAgent — deterministic exclusive choice

`RouterAgent` is the condition concept turned into a routing pattern: it
evaluates input guards in declared order and runs **exactly one** branch (or the
default). The decision spends **zero model calls**, so it is deterministic and
fully auditable.

Build it from labelled `Branch`es:

```python
from yaab import Agent, RouterAgent, Branch

billing = Agent("billing", instructions="Handle billing.")
tech = Agent("tech", instructions="Handle tech.")
general = Agent("general", instructions="Handle anything.")

router = RouterAgent(
    "support",
    [
        Branch(when='"invoice" in input', agent=billing, name="billing"),
        Branch(when='"error" in input', agent=tech, name="tech"),
    ],
    default=general,
    on_no_match="default",     # or "error" to raise when nothing matches
)
```

…or, more concisely, from a label-returning picker with `from_picker`:

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

A `from_picker` picker may only return a key present in `to` — an unknown label
raises immediately, so a typo is a loud error rather than a silent fall-through.

`RouterAgent` is a workflow agent, so it nests and works as a tool, and the same
`Branch` / `when=` model drives `.route()` in [Flow](flow.md).

## Decision events

Every skip, stop, fallback, and route decision is recorded on the run's event
stream — what was decided, which condition fired, the resolved operands, and the
unit's status — so a trace can answer *why* a guard fired without re-running
anything.

```python
from yaab import Agent, SequentialAgent, Step

a = Agent("a", instructions="A.")
b = Agent("b", instructions="B.")
pipeline = SequentialAgent("p", [Step(a, when="False"), b])
result = pipeline.run_sync("hi")
for event in result.events:
    print(event.type, event.payload.get("decision"), event.payload.get("condition"))
```

The router emits `ROUTER_EVALUATED` / `ROUTER_MATCHED`; guarded steps emit
`CONDITION_SKIP`, `CONDITION_STOP`, and `CONDITION_FALLBACK`. They surface in the
[dev console](durable-runs.md#the-trace-debug-console-yaab-web) and any event consumer.

## See also

* [Flow](flow.md) — `.route` and `.loop` use this exact routing model.
* [Multi-agent](multi-agent.md) — the patterns `when=` / `stop=` / `else_=` apply to.
* [The orchestration model](orchestration.md) — when to let code route vs the model.
