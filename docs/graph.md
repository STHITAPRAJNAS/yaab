# Graph orchestration

When you need explicit, deterministic, **durable** control flow — cycles,
fan-out/fan-in, retries, and human-in-the-loop — use `StateGraph`. It executes
in Bulk-Synchronous-Parallel supersteps (planned by the Rust core), checkpoints
state at every step, and resumes by `thread_id` after a crash or an interrupt.

## Nodes, edges, state

A node is a function `(state) -> updates` (or `(state, ctx) -> updates`). Edges
wire nodes together; `START` and `END` are sentinels.

```python
from yaab.graph import StateGraph, START, END

g = StateGraph()
g.add_node("fetch", lambda s: {"docs": fetch(s["query"])})
g.add_node("answer", lambda s: {"answer": summarize(s["docs"])})
g.add_edge(START, "fetch")
g.add_edge("fetch", "answer")
g.set_finish_point("answer")

result = g.compile().invoke({"query": "yaab"})
print(result.state["answer"])
```

## Channels & reducers

A `Channel` declares how writes to a state key are merged:

```python
from yaab.graph import StateGraph, Channel

g = StateGraph(channels={
    "count": Channel("add", default=0),       # numeric sum
    "logs":  Channel("append", default=[]),   # accumulate into a list
    "answer": Channel("last_value"),          # overwrite (default)
})
```

Reducers run in Rust (`last_value`, `append`, `add`).

## Conditional edges & cycles

Route dynamically and loop:

```python
g.add_conditional_edges(
    "inc",
    lambda s: "inc" if s["count"] < 3 else END,
    {"inc": "inc", END: END},
)
```

## Durable execution & checkpoints

Pass a checkpointer; state is persisted at every superstep, enabling crash
recovery and time-travel:

```python
from yaab.graph import MemorySaver, SQLiteSaver

app = g.compile(checkpointer=SQLiteSaver("checkpoints.db"))
app.invoke({"query": "x"}, thread_id="job-42")

# Inspect the full history (time-travel debugging):
for step, snapshot in app.checkpointer.history("job-42"):
    print(step, snapshot["state"])
```

## Human-in-the-loop

A node calls `ctx.interrupt(value)` to pause. The runtime checkpoints, returns
`interrupted=True` with the value; you resume the same thread with the human's
decision and the call returns it.

```python
def approve(state, ctx):
    decision = ctx.interrupt({"review": state["draft"]})   # pauses here
    return {"approved": decision}

app = g.compile(checkpointer=MemorySaver())
paused = app.invoke({...}, thread_id="t1")
assert paused.interrupted
done = app.invoke(thread_id="t1", resume=True)    # human approved
```

## Choosing the engine (Python vs Rust)

A compiled graph advances each superstep's state with one of two engines — your
choice:

```python
app = g.compile(engine="auto")     # rust if yaab-core is built, else python (default)
app = g.compile(engine="rust")     # force the native whole-superstep fold
app = g.compile(engine="python")   # force the pure-Python engine
print(app.engine)                   # "rust" | "python"
```

Both produce **identical results**. The Rust engine folds an entire superstep's
node updates in a single native call (one cross-language hop per superstep
instead of one per state key), which helps wide fan-outs and large state; the
Python engine has zero native dependency. `engine="rust"` raises if the
`yaab-core` extension isn't built — use `"auto"` to degrade gracefully.

What is *not* in Rust: the graph's control flow — routing, conditional edges,
HITL interrupts, checkpoint orchestration, and your node functions — all run in
Python regardless of engine. Rust only does the deterministic state fold.

## Mixing with agents

Nodes are plain functions, so call agents inside them:

```python
async def research(state, ctx):
    result = await researcher.run(state["question"])
    return {"findings": result.output}

g.add_node("research", research)   # async nodes are awaited automatically
```

This is the deterministic counterpart to the [model-driven fast path](agents.md)
and the [multi-agent workflows](multi-agent.md) — reach for it when an auditor or
SLA needs inspectable, resumable control flow.
