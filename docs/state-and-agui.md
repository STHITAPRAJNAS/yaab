# State scoping & the AG-UI layer

## Prefix-scoped state (ADK-compatible)

A session's `state` is session-scoped by default. Real apps also need values that
outlive one session, span the whole app, or must never be persisted. YAAB uses
the same key-prefix convention as Google ADK, via the `State` object:

| Prefix | Scope | Persisted? |
|---|---|---|
| `app:<key>` | shared across all users & sessions | yes (app store) |
| `user:<key>` | shared across one user's sessions | yes (user store) |
| `temp:<key>` | current run only | **no** |
| `<key>` | this session | yes (session store) |

```python
from yaab import SessionManager

mgr = SessionManager()
s = await mgr.create_session(app_name="bank", user_id="alice")

state = await mgr.resolve_state(s.id, app_name="bank", user_id="alice")
state["app:region"] = "eu"        # global to the app
state["user:tier"] = "gold"       # all of alice's sessions
state["draft"] = "..."            # only this session
state["temp:otp_ok"] = True       # ephemeral, never written

await mgr.save_state(s.id, state)  # persists everything except temp:
```

A second session for alice sees `app:` and `user:` keys; a different user sees
`app:` but not alice's `user:` state. `State` is a `MutableMapping`, so it behaves
like a dict (`in`, `len`, iteration, `del`) while routing each key to the right
backing store. `state.persisted()` returns the durable subset.

## AG-UI compatibility middleware

[AG-UI](https://docs.ag-ui.com) is the emerging protocol (popularized by
CopilotKit) for connecting agent backends to chat/coagent frontends over a
standard streamed event schema. YAAB ships a **middleware** that translates its
native event stream into AG-UI events — so any AG-UI frontend drives a YAAB agent
with no custom glue. It's a translation layer, not a dependency.

### Stream AG-UI events

```python
from yaab.agui import run_agui

async for event in run_agui(agent, "Summarize the Q3 report"):
    print(event["type"], event)   # RUN_STARTED, TEXT_MESSAGE_CONTENT, TOOL_CALL_START, ...
```

YAAB events map to the AG-UI vocabulary:

| YAAB event | AG-UI event(s) |
|---|---|
| run start | `RUN_STARTED` |
| `MODEL_DELTA` (token) | `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` / `TEXT_MESSAGE_END` |
| `MODEL_DELTA` (reasoning) | `THINKING_TEXT_MESSAGE_CONTENT` |
| `TOOL_CALL` | `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END` |
| `TOOL_RESULT` | `TOOL_CALL_RESULT` |
| `FINAL_OUTPUT` | text message (if not already streamed) |
| run end | `RUN_FINISHED` |
| error | `RUN_ERROR` |

### Serve over SSE

```python
from yaab.agui import agui_sse_app

app = agui_sse_app(agent)   # POST /agui  → SSE stream of AG-UI events
# uvicorn module:app
```

The endpoint accepts either a simple `{"prompt": "..."}` body or an AG-UI run
input `{"threadId": ..., "messages": [{"role": "user", "content": "..."}]}`. Auth
is pluggable via `yaab.auth` (same schemes as [serving](serving.md)).
