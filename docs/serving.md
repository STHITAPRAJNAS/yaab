# Serving & authentication

Any agent is an ASGI app, so anything that hosts ASGI hosts YAAB — from a local
one-liner to Cloud Run / Fargate / Lambda / K8s.

## get_fastapi_app

```python
from yaab import Agent
from yaab.serve import get_fastapi_app

agent = Agent("assistant", model="openai/gpt-4o", registry_id="assistant")
app = get_fastapi_app(agent, base_url="https://my-service")
# uvicorn module:app
```

Endpoints:

| Method & path | Purpose |
|---|---|
| `GET  /health` | liveness probe |
| `GET  /.well-known/agent.json` | A2A Agent Card (discovery) |
| `POST /run` | run the agent, return JSON |
| `POST /run/stream` | semantic events as SSE |
| `POST /chat/stream` | token deltas as SSE |
| `POST /a2a/tasks` | A2A task submission |

Or run it directly:

```bash
yaab serve mymodule:agent --host 0.0.0.0 --port 8000
```

## Authentication

Auth is pluggable; the resolved identity flows into the run context and the
audit log, and the scheme is advertised in the agent card's `securitySchemes`.

```python
from yaab.auth import NoAuth, BearerTokenAuth, APIKeyAuth, OAuth2

# Development: open access
get_fastapi_app(agent, auth=NoAuth())

# Static bearer tokens → identities
get_fastapi_app(agent, auth=BearerTokenAuth({"secret-token": "alice"}))

# API key header
get_fastapi_app(agent, auth=APIKeyAuth({"key-123": "service-a"}, header="x-api-key"))

# OAuth 2.1 (A2A standard): delegate token validation to your IdP
def verify(token: str) -> str | None:
    claims = my_idp.introspect(token)
    return claims.get("sub") if claims.get("active") else None

get_fastapi_app(agent, auth=OAuth2(verify,
    authorization_url="https://idp/authorize", token_url="https://idp/token"))
```

Implement the `AuthScheme` protocol (`authenticate(headers) -> identity`,
`describe() -> dict`) for custom schemes.

## Calling a served agent

```python
import httpx
httpx.post("https://my-service/run", json={"prompt": "hi"},
           headers={"Authorization": "Bearer secret-token"})
```

Or from another agent via [A2A](interop.md):

```python
from yaab.a2a import RemoteAgent
remote = RemoteAgent("https://my-service", auth_token="secret-token")
await remote.run("hi")
```

## Production state

Swap in durable backends (same protocols) — see [Deployment](DEPLOYMENT.md):

```python
from yaab import Runner
from yaab.sessions import SQLiteSessionService

runner = Runner(session_service=SQLiteSessionService("sessions.db"))
app = get_fastapi_app(agent, runner=runner)
```
