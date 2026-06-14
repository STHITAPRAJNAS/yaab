# Human in the loop

Some actions need a human. A wire transfer, an account deletion, a refund over a
threshold — these should stop and wait for a person, not run on the model's say-so
alone.

YAAB gives you one building block — `ToolApprovalPlugin` — with three modes that
scale from a synchronous prompt to a fully durable, out-of-band sign-off that
survives a restart and spans replicas:

> **pause → decide → resume.**

You attach the plugin to the tools that matter, the run pauses when the model
tries to call one, a human decides over any channel, and the run resumes from
exactly where it stopped — running the approved tool or feeding the model a
denial. The model never re-decides; the captured turns are never re-requested.

The same pause/decide/resume idiom answers an agent's question (`ask_user`) and
pauses a [Flow](flow.md) step (`ctx.pause_for`) — one mechanism, three front
doors. A human decides with one of four verbs — `approve`, `deny`, `edit`,
`respond` — and resume is always `agent.run(resume=decision)`.

---

## Pick the tools that need approval

Gate by name or by a predicate on the proposed arguments, so you only stop on
the calls that matter:

```python
from yaab.governance import ToolApprovalPlugin

# By name:
ToolApprovalPlugin(tools=["wire_transfer", "delete_account"])

# By a rule — only large transfers pause; small ones run straight through:
ToolApprovalPlugin(needs_approval=lambda tool, args, ctx: args["amount"] >= 1000)
```

That same plugin, with a different `mode`, drives all three approval styles
below.

---

## Mode 1 — inline approval (synchronous)

The simplest case. Give an async `approver(tool, args, ctx) -> bool`; it is
awaited before the tool runs. Return `True` to allow it, `False` to reject. A
rejection short-circuits the tool with a message the model can adapt to, instead
of failing the run.

```python
from yaab import Agent, Runner, tool
from yaab.governance import ToolApprovalPlugin

@tool
def wire_transfer(amount: int, to: str) -> str:
    return f"sent ${amount} to {to}"

async def ask_a_human(tool, args, ctx) -> bool:
    # prompt a CLI, call a Slack approval bot, check a queue — anything async
    return args["amount"] < 10_000

plugin = ToolApprovalPlugin(tools=["wire_transfer"], approver=ask_a_human)
agent = Agent("banker", tools=[wire_transfer], runner=Runner(plugins=[plugin]))

result = await agent.run("wire $5000 to ACME")
print(result.output)
```

Inline mode holds the run for the duration of the `approver` call — perfect for a
CLI confirmation or a bot that replies within the request. For anything that may
take minutes or hours (a person who is away), use **queue** mode below, which
does not hold a thread.

---

## Mode 2 — block (surface the pending call)

With no approver, a guarded tool raises `ApprovalRequired`, surfacing the tool
and the proposed arguments so an out-of-band flow can decide and re-run. No
durable store and no checkpointer are involved — this is the lightweight "stop
and tell me" signal:

```python
from yaab.exceptions import ApprovalRequired

plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="block")
agent = Agent("banker", tools=[wire_transfer], runner=Runner(plugins=[plugin]))

try:
    await agent.run("wire $5000 to ACME")
except ApprovalRequired as pending:
    print(pending.tool, pending.arguments)   # "wire_transfer" {"amount": 5000, "to": "ACME"}
```

---

## Mode 3 — queue (durable, out-of-band sign-off)

This is the production path: the run **pauses durably** instead of blocking. Give
the plugin an `ApprovalStore` (where the decision lives) and the runner a
`run_checkpointer` (where the run sleeps), and run with a `resume_id` (the key
that ties the two together).

When the model calls a guarded tool:

1. a pending `ApprovalRequest` is persisted (any replica can see it),
2. the run's state is checkpointed with a pending-approval marker,
3. an `APPROVAL_REQUIRED` event is emitted and the run **ends** — consuming zero
   compute while it waits. No thread is held; the process can exit.

```python
from yaab import Agent, Runner
from yaab.governance import ToolApprovalPlugin
from yaab.governance.approvals import SQLiteApprovalStore
from yaab.graph.checkpoint import SQLiteSaver

store = SQLiteApprovalStore("approvals.db")          # where the decision lives
plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)
runner = Runner(run_checkpointer=SQLiteSaver("runs.db"), plugins=[plugin])  # where it sleeps
agent = Agent("banker", tools=[wire_transfer], runner=runner)

# 1. Run pauses durably instead of running the tool.
async for ev in runner.run_stream(agent, "wire $5000 to ACME", resume_id="run-42"):
    if ev.type.value == "approval_required":
        print(ev.payload["approval_id"], ev.payload["tool"], ev.payload["arguments"])
```

A paused run is two durable rows — a checkpoint and a pending approval — keyed by
the same `resume_id`. Approve it from any replica, days later, and it resumes
from its last completed step.

### Decide — one verb set

A reviewer decides with one of four verbs from `yaab.governance.approvals`. Each
returns a `Decision` — the single value resume consumes. The store is durable, so
the reviewer can be on a different process than the one that paused the run:

```python
from yaab.governance import approvals

# Either pass the paused RunResult directly, or look up what is waiting:
req = (await store.list_pending())[0]

decision = await approvals.approve(req.approval_id, by="alice", store=store)
# decision = await approvals.deny(req.approval_id, by="bob", reason="too large", store=store)
# decision = await approvals.edit(req.approval_id, by="alice",
#                                 arguments={"amount": 1000}, store=store)   # run with corrected args
```

* **`approve`** — let the held tool run.
* **`deny`** — refuse it, with a `reason` the model reads and can revise from.
* **`edit`** — approve with corrected `arguments`; the tool runs with those.
* **`respond`** — answer an `ask_user` question with a typed `answer`.

The `Decision` is self-correlating: it carries the `approval_id` and the
`resume_id` (the checkpoint key), so resume needs no `session_id` and works from a
fresh process given only the `approval_id` and the same store config. `decide` is
first-write-wins, so a double-approve resumes the run exactly once.

### Resume

Resume the **same** run by threading the `Decision` into `agent.run(resume=...)`.
The captured model turns are never re-requested — on approve the guarded tool
runs now (the model already decided to call it; a human just unblocked it); on
deny the model receives the denial and continues:

```python
result = await agent.run(resume=decision)
print(result.output)                  # the tool ran; the run finished
```

Because the decision carries its own correlation keys, this works from a fresh
process — the replica that paused the run and the one that resumes it need not be
the same. When one model turn guarded several tools, decide each and resume once
with `approvals.multiplex(result, {approval_id: decision, ...})`.

---

## Ask the human a question (`ask_user`)

Approval gates a tool the *model* chose; sometimes the agent needs a fact it does
not have ("for how many people?", "which address?"). The built-in `ask_user` tool
pauses the run with a `question` pending and resumes with the human's validated
answer returned inline — reusing the *same* pause machinery, decided with
`respond`.

```python
from yaab import Agent
from yaab.tools.builtin import ask_user
from yaab.governance import ToolApprovalPlugin, InMemoryApprovalStore, approvals

store = InMemoryApprovalStore()
agent = Agent("concierge", tools=[ask_user],
              hitl=ToolApprovalPlugin(tools=["ask_user"], mode="queue", store=store))

result = await agent.run("Book me a table tonight", resume_id="r")
if result.paused and result.pending[0].kind == "question":
    answer = await approvals.respond(result, by="user", answer=4, store=store)
    result = await agent.run(resume=answer)
```

`ask_user` accepts an optional `answer_schema` (a JSON Schema, e.g.
`{"type": "integer", "minimum": 1}`); the human's answer is validated against it
*before* anything is stored, so a mistyped answer leaves the run paused rather
than half-committing.

## Pause a Flow step (`ctx.pause_for`)

Inside a [Flow](flow.md) step, `ctx.pause_for(value)` suspends the whole run and
lands an `ApprovalRequest` with `kind="flow_pause"` — so the pause shows up in
`GET /approvals` and `approvals.respond()` works on it identically. The caller
resumes the same `session_id` with the decision, which is what `pause_for`
returns on continuation. See [Flow → the HITL pause](flow.md#the-hitl-pause-ctxpause_for).

---

## Decide over HTTP

Serve the agent with a run store and an approval store, and the sign-off
endpoints appear automatically. Approving over HTTP records the decision and
re-enqueues the run; a worker picks it up and finishes it on whatever replica is
free — the caller never holds a connection open.

```python
from yaab import durable_backends
from yaab.serve import serve

backends = durable_backends(dsn="sqlite://app.db")    # run + approval + checkpoint, one DB
serve(agent, **backends.serve_kwargs())
```

```
GET  /approvals?status=pending        # list what is waiting
GET  /approvals/{approval_id}         # one request (tool + arguments)
POST /approvals/{approval_id}/approve # {"reviewer": "alice"}
POST /approvals/{approval_id}/deny    # {"reviewer": "alice", "reason": "too large"}
POST /runs/{run_id}/resume            # idempotent manual re-enqueue
```

On approve, the run resumes and runs the held tool. On deny, it resumes with the
denial fed back to the model. Either way the run finishes on a worker, decoupled
from the request that approved it.

---

## Who may approve what

`ToolApprovalPlugin` records every pause and decision to the audit log when you
pass one — who asked, what for, who decided, and why:

```python
from yaab.governance import AuditLog, ToolApprovalPlugin

audit = AuditLog()
plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store, audit=audit)

# later
for e in audit.events:
    print(e.kind, e.payload)   # APPROVAL  {"tool": "wire_transfer", "decision": "pending", ...}
assert audit.verify()          # the hash chain is intact — tamper-evident
```

Because the chain folds each entry's hash into the next, a retroactively edited
decision breaks `verify()`. This is the same ledger that backs the rest of
YAAB's compliance evidence. To restrict *who* may run a guarded tool in the first
place, compose the plugin with the authorization layer
(`ToolAuthorizationPlugin` / `RBACAuthorizer`) that already gates tool calls.

---

## Make pauses survive a restart

In single-process dev, an in-memory store and checkpointer are fine:

```python
from yaab.governance.approvals import InMemoryApprovalStore
from yaab.graph.checkpoint import MemorySaver

store = InMemoryApprovalStore()
runner = Runner(run_checkpointer=MemorySaver(), plugins=[
    ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store),
])
```

For anything that must outlive a restart or span replicas, point the store and
the checkpointer at durable backends — the code above is unchanged except the
backend names. The one-call `durable_backends(dsn=...)` wires the approval store,
the checkpointer, and the run queue against one database for you:

```python
from yaab import durable_backends

backends = durable_backends(dsn="postgresql://user:pw@db/app")
# backends.approval_store, backends.run_checkpointer, backends.run_store
# all share one database — pass them to the plugin, the Runner, and serve().
```

Now a paused run is durable: approve it from any replica, days later, and it
resumes from its last completed step. Swap `sqlite://` for a Postgres DSN to go
multi-replica; nothing else in your agent code changes.

---

## One model

That is human-in-the-loop in YAAB. One plugin, three modes that share the same
shape:

> **pause → decide → resume.**

Inline for a synchronous prompt, block for a lightweight signal, and queue for a
durable, out-of-band sign-off that survives a restart and resumes on any replica.
