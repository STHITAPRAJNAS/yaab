# Deploying YAAB — local to cloud

YAAB runs the same way everywhere: an agent is an ASGI app
(`fastapi_server_app`), so anything that hosts ASGI/containers hosts YAAB.

## Local

```bash
pip install 'yaab[litellm]' fastapi uvicorn
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
pip install 'yaab[postgres]'
```

```python
from yaab.sessions import PostgresSessionService
runner = Runner(session_service=PostgresSessionService("postgresql://user:pw@host/db"))
```

Lists are paginated for large tenants:

```python
ids = await session_manager.list_sessions(app_name="app", user_id="u", limit=50, offset=0)
```

## Observability

Install the OTel extra and configure an exporter; YAAB emits spans following the
OpenTelemetry **GenAI semantic conventions** (`gen_ai.*`) for every model call,
plus token/cost attributes.

```bash
pip install 'yaab[otel]'
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
