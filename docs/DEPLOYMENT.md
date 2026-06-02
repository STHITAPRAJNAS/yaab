# Deploying YAAB — local to cloud

YAAB runs the same way everywhere: an agent is an ASGI app
(`fastapi_server_app`), so anything that hosts ASGI/containers hosts YAAB.

## Local

```bash
pip install 'yaab-sdk[litellm]' fastapi uvicorn
yaab serve examples.serve_app:agent          # http://127.0.0.1:8000
```

Endpoints exposed:

| Method & path | Purpose |
|---|---|
| `GET  /health` | liveness probe |
| `GET  /.well-known/agent.json` | A2A Agent Card (discovery) |
| `POST /run` | run the agent, return JSON |
| `POST /run/stream` | run the agent, stream **SSE** events |
| `POST /a2a/tasks` | A2A task submission (agent-to-agent) |

## Authentication

Wrap with any scheme from `yaab.auth` — the resolved identity flows into the
run context and the audit log:

```python
from yaab.serve import fastapi_server_app
from yaab.auth import BearerTokenAuth, APIKeyAuth, OAuth2

app = fastapi_server_app(agent, auth=BearerTokenAuth({"token-123": "alice"}))
# OAuth2(validator=verify_jwt, authorization_url=..., token_url=...) for A2A.
```

The chosen scheme is advertised in the agent card's `securitySchemes`.

## Container (Cloud Run / Fargate / K8s)

```bash
docker build -t yaab .
docker run -p 8000:8000 -e YAAB_AGENT="examples.serve_app:agent" yaab
```

The image compiles the Rust core in a build stage and runs the pure-Python
fallback automatically if a wheel is ever unavailable. Point `YAAB_AGENT` at
your own `module:agent`.

- **Cloud Run / App Runner:** deploy the image; set the port to 8000.
- **Fargate / ECS:** same image as a task definition behind an ALB.
- **Kubernetes:** Deployment + Service; use the `/health` path for probes.
- **AWS Lambda (container):** wrap the ASGI app with an adapter (e.g. Mangum).

## Durable state in production

Swap the in-memory backends for durable ones — same protocols, different
constructor:

```python
from yaab import Runner
from yaab.sessions import SQLiteSessionService
from yaab.graph import SQLiteSaver
from yaab.governance import AuditLog, SQLiteAuditSink, SQLiteRegistryBackend, AgentRegistry

runner = Runner(session_service=SQLiteSessionService("sessions.db"))
checkpointer = SQLiteSaver("checkpoints.db")
audit = AuditLog(sinks=[SQLiteAuditSink("audit.db")])
registry = AgentRegistry(SQLiteRegistryBackend("registry.db"))
```

For Postgres, install the extra and swap one line — same `SessionService`
protocol, so agent code is untouched:

```bash
pip install 'yaab-sdk[postgres]'
```

```python
from yaab.sessions import PostgresSessionService
runner = Runner(session_service=PostgresSessionService("postgresql://user:pw@host/db"))
```

Lists are paginated for large tenants:

```python
ids = await session_manager.list_sessions(app_name="app", user_id="u", limit=50, offset=0)
```

## Running N replicas

The in-memory defaults are **single-process only**. The moment you run more than
one replica behind a load balancer, each replica keeps its own private copy of
state, so they never see one another's data: background runs vanish on restart,
an approval queued on one replica is invisible to another, and a `rate=10`
budget silently becomes `10 × replicas`. Before scaling past one replica, every
stateful concern must point at a *shared* backend.

### Mandatory swaps

| Stateful concern | In-memory default (single process) | Durable backend (N replicas) |
|---|---|---|
| Conversation sessions | `InMemorySessionService` | `SQLiteSessionService` / `PostgresSessionService` / `RedisSessionService` |
| Artifacts (files/blobs) | `InMemoryArtifactService` | `SQLiteArtifactService` / `PostgresArtifactService` / `RedisArtifactService` |
| Run store (the run queue) | `InMemoryRunStore` | `SQLiteRunStore` / `PostgresRunStore` / `RedisRunStore` |
| Approval store (human sign-off) | `InMemoryApprovalStore` | `SQLiteApprovalStore` / `PostgresApprovalStore` / `RedisApprovalStore` |
| Trace store (run history) | `InMemoryTraceStore` | `SQLiteTraceStore` / `PostgresTraceStore` / `RedisTraceStore` |
| Run checkpointer (fault tolerance) | `MemorySaver` | `SQLiteSaver` / `PostgresSaver` / `RedisSaver` |
| Audit sink (compliance ledger) | `InMemoryAuditSink` | `SQLiteAuditSink` |
| Agent registry | `InMemoryRegistryBackend` | `SQLiteRegistryBackend` / `RemoteRegistryBackend` |
| Rate-limit budget | in-process `RateLimiter` (per replica) | `RedisRateLimiter` (one global budget) |

> Postgres (or Aurora PostgreSQL) is the recommended multi-replica backend for
> the run/approval/trace/session/artifact stores; add Redis for the shared
> rate-limit budget. SQLite is durable on a *single* node only.

### One call to wire it all

`durable_backends` builds the whole set against one database URL and hands each
consumer the slice it needs — no backend wired by hand:

```python
from yaab import Runner, durable_backends
from yaab.serve import serve

# One Postgres for everything; one Redis for the global rate-limit budget.
backends = durable_backends(
    dsn="postgresql://user:pw@db.internal/app",
    redis_url="redis://cache.internal:6379/0",
)

runner = Runner(**backends.runner_kwargs())   # sessions, artifacts, checkpoint, trace
serve(agent, **backends.serve_kwargs())        # run queue, approvals, trace, fault tolerance
```

With no `dsn` the same call returns process-local backends — the dev/test
default — so the wiring is identical from laptop to cluster. A `sqlite://path.db`
DSN gives durable single-node storage for staging.

### Fail loudly, not silently

The server checks at startup whether any backend is still in-memory while more
than one replica is configured, and emits a `RuntimeWarning` naming exactly which
ones will lose data. It reads the replica count from `YAAB_REPLICAS`:

```bash
export YAAB_REPLICAS=3          # the server warns if any backend is in-memory
export YAAB_STRICT_DURABILITY=1 # also warn on a single replica (CI/staging gate)
```

A single replica (the default) stays silent, so existing single-process setups
are unchanged. The guardrail turns "discovered in production" into "screamed at
boot."

### Kubernetes (N replicas + autoscaling)

Because run state is shared, no sticky sessions are needed — any replica can
serve any request and resume any run. A standard Deployment + Service + HPA:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yaab
spec:
  replicas: 3
  selector:
    matchLabels: { app: yaab }
  template:
    metadata:
      labels: { app: yaab }
    spec:
      containers:
        - name: yaab
          image: your-registry/yaab:latest
          ports: [{ containerPort: 8000 }]
          env:
            - { name: YAAB_AGENT, value: "examples.serve_app:agent" }
            - { name: YAAB_REPLICAS, value: "3" }          # boot-time durability check
            - name: DATABASE_URL                            # consumed by your wiring
              valueFrom: { secretKeyRef: { name: yaab-db, key: dsn } }
            - name: REDIS_URL
              valueFrom: { secretKeyRef: { name: yaab-redis, key: url } }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
---
apiVersion: v1
kind: Service
metadata:
  name: yaab
spec:
  selector: { app: yaab }
  ports: [{ port: 80, targetPort: 8000 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: yaab
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: yaab }
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
```

A background `RunWorker` drains the durable run queue with bounded concurrency
and lease-based crash recovery: enqueueing 1,000 runs creates 1,000 durable rows
but never 1,000 in-flight tasks, and a run abandoned by a recycled pod is
re-queued and resumes from its last checkpoint on another replica. Durable
schedules (`/crons`) and per-run completion webhooks let callers trigger and be
notified without holding a connection open.

## Observability

Install the OTel extra and configure an exporter; YAAB emits spans following the
OpenTelemetry **GenAI semantic conventions** (`gen_ai.*`) for every model call,
plus token/cost attributes.

```bash
pip install 'yaab-sdk[otel]'
```

Control tracing at runtime — disable it entirely, or scrub PII from span
attributes before they're recorded:

```python
from yaab.observability import set_tracing_enabled, set_trace_redactor

set_tracing_enabled(False)                 # global off switch (or YAAB_DISABLE_TRACING=1)

def scrub(key, value):                     # redact sensitive attributes
    return "[REDACTED]" if key.endswith("prompt") else value
set_trace_redactor(scrub)
```

## Governance in production

Run with `GovernanceMode.ENFORCING` so unregistered/unapproved agents are
refused and guardrail blocks stop the run. Generate evidence on demand:

```bash
yaab compliance report eu_ai_act --db registry.db
```
