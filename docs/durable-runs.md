# Durable background runs

A run does not have to be a fleeting in-process task. With a **run store** it
becomes a durable row that survives a restart, is visible from any replica behind
a load balancer, and carries everything needed to poll, cancel, lease, and resume
it. This is what lets you fire a long job, return immediately, scale to N
replicas, and have a paused run resume on whichever replica is free.

This page covers the durable-run machinery. For the full N-replica deployment
recipe (Kubernetes manifests, the boot-time durability check, observability) see
[Deployment](DEPLOYMENT.md); for the human-pause model see
[Human-in-the-loop](hitl.md).

## The run store and the worker

The **`RunStore`** is the cross-process system-of-record: `create` / `get` /
`update` / `list` track a run from queued to a terminal state. The **`RunWorker`**
drains the queue with bounded concurrency, leases each run it claims, heartbeats
the lease while the run is in flight, and records the terminal outcome.

```python
import time
from yaab import InMemoryRunStore, RunRecord, RunWorker, Agent

store = InMemoryRunStore()              # or SQLiteRunStore / PostgresRunStore / RedisRunStore
now = time.time()
await store.create(RunRecord(
    run_id="r1", agent="assistant", prompt="summarize Q3",
    background=True, created_at=now, updated_at=now,
))

agent = Agent("assistant", instructions="Be concise.")
worker = RunWorker(agent, store, max_concurrency=10)
# worker.run_forever() drains the queue, leases each run, and reaps crashed ones.
```

Three properties make this safe to run as a fleet:

* **Bounded concurrency.** A semaphore caps in-flight runs, so a thousand
  submissions enqueue a thousand rows but never spawn a thousand tasks — queue
  depth is the natural backpressure signal.
* **Crash and rolling-deploy survival.** Each running row holds a lease the
  worker refreshes; if a replica dies mid-run its lease expires and the reaper
  (on any replica) re-queues the run, which resumes from its last checkpoint.
* **Eviction-on-pause.** When a run parks for human sign-off, the worker releases
  the lease and frees the slot, so a paused run consumes zero worker capacity and
  can resume on any replica.

## Survive a restart, resume from the last step

A run becomes fault-tolerant when the Runner has a checkpointer: loop progress is
persisted under a `resume_id` after every completed step, so a crashed or paused
run re-invoked with the same `resume_id` continues from where it left off —
without re-requesting the model turns already captured.

```python
from yaab import Runner
from yaab.graph.checkpoint import SQLiteSaver

runner = Runner(run_checkpointer=SQLiteSaver("runs.db"))
# Re-invoke with the same resume_id after a crash; it rehydrates from the last step.
result = await runner.run(agent, "long job", resume_id="job-42")
```

It is inert (zero overhead) when the runner has no checkpointer.

## Cross-replica cancel and resume

A cancel issued on one replica must stop a run executing on another. `RunStore`
exposes a durable cancel flag any replica observes, and the runner's cooperative
cancellation bridges to it via `StoreCancellationToken`:

```python
await store.request_cancel("r1")        # any replica; the running replica stops between steps
record = await store.get("r1")
print(record.cancel_requested)          # True — observed everywhere
```

Resume is the mirror image: a paused run (`RunStatus.PAUSED`) sleeps in the store
consuming no compute, and a guarded compare-and-set (`update(..., expect_status=
RunStatus.PAUSED)`) lets exactly one replica win the race to flip it back to
running after a human decides. See [Human-in-the-loop](hitl.md) for the
pause/decide/resume verbs.

## Multi-replica deployment with `durable_backends()`

The in-memory defaults are **single-process only**: the moment you run more than
one replica, each keeps its own private copy of state, so background runs vanish
on restart, an approval queued on one replica is invisible to another, and a
`rate=10` budget silently becomes `10 x replicas`. `durable_backends()` removes
the footgun — give it one database URL and it returns a coherent set of backends
all pointed at the same place.

```python
from yaab import Runner, durable_backends
from yaab.serve import serve

backends = durable_backends(dsn="postgresql://user:pw@db/app")
runner = Runner(**backends.runner_kwargs())   # sessions, artifacts, checkpoint, trace
serve(agent, **backends.serve_kwargs())        # run queue, approvals, trace, spend, fault tolerance
```

With no `dsn` the same call returns process-local backends — the dev/test default
— so the wiring is identical from laptop to cluster. `sqlite://path.db` is durable
on a single node; `postgresql://...` is the multi-replica backend.

### The ephemeral guardrail

Misconfiguration should scream at boot, not surface in production. `warn_if_ephemeral`
checks at startup whether any backend is still in-memory while more than one
replica is configured, and emits a `RuntimeWarning` naming exactly which ones
will lose data. The server runs this check for you, reading the replica count
from `YAAB_REPLICAS`:

```bash
export YAAB_REPLICAS=3           # the server warns if any backend is in-memory
export YAAB_STRICT_DURABILITY=1  # also warn on a single replica (CI/staging gate)
```

A single replica (the default) stays silent, so existing single-process setups
are unchanged.

## The trace-debug console: `yaab web`

`yaab web mymodule:agent` serves a zero-build local dev console — a single
self-contained HTML page (no bundler, no npm) that mounts the agent's full API
and layers inspector tabs over it:

* **Chat** — token streaming.
* **Events** — a live, colour-coded event-stream timeline with payload JSON and a
  run-summary header (tokens, cost, latency).
* **Runs** — lists runs with status badges, a per-run **Cancel**, plus **Trace**
  and **Replay** actions, auto-refreshing while open.
* **Trace** — a per-step span/waterfall: typed spans (model call, tool call,
  transfer, approval) with per-span latency and token/cost badges, and run totals.
* **State** — a session-state inspector.
* **Approvals** — lists pending sign-offs with Approve/Deny buttons.
* **Agent** — the agent card, tools, and instructions.

```python
from yaab.web import web_app

app = web_app(agent, trace_store=backends.trace_store, approval_store=backends.approval_store)
# uvicorn module:app    — or: yaab web mymodule:agent
```

Pass a `trace_store` to light up the Trace and State tabs (per-step
model/tool/token/cost/latency detail), and an `approval_store` (+ `run_store`) to
light up the Approvals tab. Tabs whose backing store is not configured degrade
gracefully — the endpoint returns a clean `404` and the tab shows a "configure a
store" hint.

## Durable schedules and artifacts

**Schedules** are a durable store kind too. A `CronStore` holds recurring
schedules, and the same worker materializes each due schedule into exactly one
queued run, reusing the one run-creation path — so a fleet of workers never
double-fires a schedule.

```python
from yaab.runs import SQLiteCronStore, RunWorker

cron_store = SQLiteCronStore("crons.db")
worker = RunWorker(agent, store, cron_store=cron_store)
# worker.cron_tick() materializes due schedules; run_forever() interleaves it.
```

**Artifacts** (binary/file blobs produced or consumed by tools) have the same
durable-backend story: the in-memory default swaps for `SQLiteArtifactService`,
`PostgresArtifactService`, or `RedisArtifactService`, all behind one
`ArtifactService` protocol. `durable_backends()` wires the artifact service
alongside everything else, so a file written on one replica is readable on
another. See [State](state.md) for the artifact manager API.

## See also

* [Deployment](DEPLOYMENT.md) — N-replica recipe, Kubernetes, observability.
* [Human-in-the-loop](hitl.md) — the durable pause/decide/resume model.
* [Flow](flow.md) — durable control flow with `RunHistory` time-travel.
* [Storage & backends](storage-backends.md) — the full backend matrix.
