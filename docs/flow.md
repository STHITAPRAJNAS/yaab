# Flow — explicit, durable control flow

`Flow` is the workflow pattern you reach for when the **control flow itself**
must be inspectable and crash-proof: a refund pipeline that branches on an
amount, a research loop that iterates until a quality bar, a step that pauses for
a human and resumes days later on a different replica. It is a *builder* — a
chain of `.step / .then / .route / .loop / .fan_out / .start_at / .returns` — that
lowers onto YAAB's durable graph engine and runs through the same superstep
runtime everything else does.

`Flow` owns no state object, no checkpoint format, and no pause type of its own.
It threads the one shared [`State`](state.md), routes on the one
[`Condition`](conditions.md), and pauses into the one `Pending` — so a flow nests
inside any other pattern and composes as a tool, exactly like a plain `Agent`.

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
print(flow.run_sync("refund #42").output)   # -> "auto"
```

`Flow[Deps, Output]` parameterizes exactly like `Agent[Deps, Output]`: `deps`
flows to every step's `RunContext`, and the value at the `.returns(key)` key
becomes the typed `RunResult.output`.

## A step is a function or an agent

A step body is either a plain function `(state, ctx)` or an `Agent` used
directly. Both read and write the one shared `State` and receive the same
`RunContext` tools receive.

```python
from yaab import Agent, Flow

drafter = Agent("drafter", instructions="Draft a reply from the request.")
polisher = Agent("polisher", instructions="Tighten and proofread the draft.")

pipeline = (
    Flow[None, str]("reply_pipeline")
    .step("draft", agent=drafter, writes="draft")    # agent output -> state["draft"]
    .step("polish", agent=polisher, writes="final")
    .then("draft", "polish").then("polish", Flow.DONE)
    .start_at("draft").returns("final")
)
```

An **agent step** runs through the *parent* Runner, so usage rolls up, events
bubble to the parent stream, and the session, identity, and governance all
thread through — it is not a fresh `Runner()`.

## The builder API

| Method | What it declares |
|---|---|
| `.step(name, agent=A)` / `.step(name, fn=f)` | a unit of work (an `Agent` or a `(state, ctx)` function) |
| `.then(src, dst)` | a plain "then go to" edge (`dst` may be `Flow.DONE`) |
| `.route(src, picker, to={...})` | branch from `src` on the label the picker returns |
| `.loop(step, until=..., max_iterations=5)` | an explicit, bounded cycle on `step` |
| `.fan_out(src, [a, b, c])` | run several successors of `src` in parallel |
| `.start_at(name)` | set the entry step |
| `.returns(key)` | name the state key whose value becomes `RunResult.output` |

`Flow.DONE` (the terminal marker) and `Flow.ENTRY` are the public spellings of
the engine's `END` / `START`.

## Shared state via `writes=`

Steps communicate through the one shared `State`. A bare function step merges its
returned dict into state; `writes="key"` captures a step's typed output (or an
agent step's `RunResult.output`) under a key the next step reads — by `{key}`
instruction injection on an agent, or directly on `state`.

```python
from yaab import Agent, Flow

classify = Agent("classify", instructions="Classify the request topic.")
answer = Agent("answer", instructions="The topic is {topic}. Answer the request.")

flow = (
    Flow[None, str]("triage")
    .step("classify", agent=classify, writes="topic")   # output -> state["topic"]
    .step("answer", agent=answer, writes="reply")        # reads {topic}
    .then("classify", "answer").then("answer", Flow.DONE)
    .start_at("classify").returns("reply")
)
```

The key's prefix chooses the scope (`temp:` / `user:` / `app:` / session); two
parallel branches that write the *same* unprefixed key need a reducer, declared
as `merge="append"` (or `"add"`) on the step. See [State](state.md) for scoping
rules.

## Conditional routing

A **picker** computes a label from the read-only state view and the
`RunContext`; the `to` map sends each label to a successor step (or `Flow.DONE`).
A picker is a callable `(state, ctx) -> label`, a `Condition`, or a sandboxed
expression string — and it runs against a **read-only** state, so it physically
cannot mutate state.

```python
from yaab import Flow

flow = (
    Flow[None, str]("refund_router")
    .step("parse", fn=lambda state, ctx: {"amount": 250})
    .route(
        "parse",
        lambda state, ctx: "human" if state["amount"] >= 100 else "auto",
        to={"auto": "execute", "human": "await_approval"},
    )
    .step("execute", fn=lambda state, ctx: {"out": "auto-refunded"})
    .step("await_approval", fn=lambda state, ctx: {"out": "queued for review"})
    .then("execute", Flow.DONE).then("await_approval", Flow.DONE)
    .start_at("parse").returns("out")
)
```

A picker that returns a label not in the `to` map is a loud error, never a
silent fall-through. Route completeness is also validated at build time: every
target must be a real step or a terminal marker.

## Loops

`.loop()` is sugar for "run this step, then route back to it until the exit
predicate says stop" — an explicit, bounded cycle. State accumulates across
iterations on the one shared `State`; the loop also stops at `max_iterations`.

```python
from yaab import Flow

flow = (
    Flow[None, int]("refine")
    .step("refine", fn=lambda state, ctx: {"score": state.get("score", 0) + 1})
    .loop("refine", until=lambda state, ctx: state["score"] >= 3, max_iterations=10)
    .start_at("refine").returns("score")
)
print(flow.run_sync("go").output)   # -> 3
```

The `until` predicate reads **state** (not an output value): its first argument
is the read-only state view, so `until="state.score >= 3"` and the callable form
both see the accumulated loop state.

## The HITL pause: `ctx.pause_for`

A step calls `ctx.pause_for(value)` to suspend the **whole run**. The flow
checkpoints at the superstep boundary, surfaces `value` to the caller as
`result.paused` / `result.pause_value`, and — per the durable design — also lands
an `ApprovalRequest` row with `kind="flow_pause"`, so the pause is visible in
`GET /approvals` and `approvals.respond()` works on it identically to a tool
approval. The caller resumes the same `session_id` with the human's decision,
which is what `pause_for` returns on continuation.

```python
from yaab import Flow, Runner
from yaab.flow import MemoryCheckpoints   # or SQLiteCheckpoints("refunds.db")

def gate(state, ctx):
    decision = ctx.pause_for({"needs": "approval", "amount": state["amount"]})
    return {"out": f"refunded {state['amount']}" if decision == "approve" else "denied"}

flow = (
    Flow[None, str]("refund")
    .step("parse", fn=lambda state, ctx: {"amount": 250})
    .step("gate", fn=gate)
    .then("parse", "gate").then("gate", Flow.DONE)
    .start_at("parse").returns("out")
)

runner = Runner(run_checkpointer=MemoryCheckpoints())

# First call — the gate pauses.
result = runner.run_sync(flow, "refund order #42", session_id="cust-42")
if result.paused:
    print(result.pause_value)               # {"needs": "approval", "amount": 250}
    # ... a human reviews out of band, then ...
    result = runner.run_sync(flow, resume="approve", session_id="cust-42")
print(result.output)                         # "refunded 250"
```

A crash between any two steps is invisible: re-running with the same
`session_id` rehydrates from the last checkpoint. For a durable backend that
outlives a restart and spans replicas, swap `MemoryCheckpoints()` for
`SQLiteCheckpoints("runs.db")` (or `PostgresCheckpoints` / `RedisCheckpoints`) —
nothing else changes. See [Durable runs](durable-runs.md) and
[Human-in-the-loop](hitl.md) for the full pause/decide/resume model.

## Fan-out and merge

`.fan_out(src, [a, b, c])` runs several successors in one superstep. When two
branches both write the same unprefixed key, declare a reducer with `merge=` on
the step so the engine folds the concurrent writes instead of raising a conflict.

```python
from yaab import Flow

flow = (
    Flow[None, list]("research")
    .step("plan", fn=lambda state, ctx: {})
    .step("web", fn=lambda state, ctx: {"notes": ["web result"]}, writes="notes", merge="append")
    .step("docs", fn=lambda state, ctx: {"notes": ["doc result"]}, writes="notes", merge="append")
    .fan_out("plan", ["web", "docs"])
    .then("web", Flow.DONE).then("docs", Flow.DONE)
    .start_at("plan").returns("notes")
)
```

## Flow composes like any agent

A `Flow` is a workflow agent, so it nests and works as a tool:

```python
from yaab import Agent, Flow, SequentialAgent

flow = Flow[None, str]("process").step("x", fn=lambda s, c: {"out": "done"}) \
    .then("x", Flow.DONE).start_at("x").returns("out")

supervisor = Agent("ops", tools=[flow.as_tool(name="process")])     # as a tool
classifier = Agent("classifier", instructions="Classify.")
notifier = Agent("notifier", instructions="Notify.")
intake = SequentialAgent("intake", [classifier, flow, notifier])     # nested
```

## RunHistory — list, inspect, fork

A flow run with a checkpointer leaves a per-step trail. `RunHistory` is a thin
read/transform layer over the same checkpointer the Runner uses, so you can
render a timeline, inspect any captured snapshot (including a paused one), and
**fork** from a step into a new thread — optionally editing the state — to
explore an alternate timeline without touching the original.

```python
from yaab import Runner
from yaab.flow import MemoryCheckpoints
from yaab.runs.history import RunHistory

checkpointer = MemoryCheckpoints()
runner = Runner(run_checkpointer=checkpointer)
runner.run_sync(flow, "refund order #42", session_id="cust-42")

history = RunHistory(checkpointer)
for step, snapshot in history.list("cust-42"):       # oldest first
    print(step, snapshot["state"])

snapshot = history.inspect("cust-42", step=1)         # one checkpoint's state
latest = history.latest("cust-42")                    # most recent (incl. paused)

# Fork from step 1 into a new thread with an edited value, then continue from it:
history.fork("cust-42", step=1, to_thread="what-if", edits={"amount": 50})
result = runner.run_sync(flow, "", session_id="what-if", resume_from_checkpoint=True)
```

Forking never mutates the source timeline — it writes to a *different* thread, so
it is a non-destructive "what if I had changed X here?".

## See also

* [The orchestration model](orchestration.md) — when to reach for Flow vs the
  other patterns.
* [Conditions](conditions.md) — the routing/guard model `.route` and `.loop` use.
* [Graph orchestration](graph.md) — the `StateGraph` engine Flow lowers onto.
* [Durable runs](durable-runs.md) — durable checkpoints, cross-replica resume,
  and the trace console.
