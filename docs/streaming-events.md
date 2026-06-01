# Streaming & events

YAAB gives you two complementary streams:

1. **Token stream** — text deltas for a single answering turn (chat UX).
2. **Event stream** — typed semantic events across the whole tool loop (UIs,
   tracing, audit).

## Token streaming

```python
async for token in agent.stream("tell me a joke"):
    print(token, end="", flush=True)
```

`agent.stream(...)` returns an async iterator of strings. It runs a single
answering turn (no tool loop) and respects input guardrails. Over HTTP, the
serve layer exposes this as Server-Sent Events at `POST /chat/stream`
(see [Serving](serving.md)).

## The semantic event stream

`agent.stream_events(...)` drives the **whole tool loop** and yields a typed
`Event` for every step — token deltas, tool calls, sub-agent transfers, and the
final result:

```python
from yaab import EventType

async for event in agent.stream_events("What's the weather in Paris?"):
    if event.type is EventType.TEXT_DELTA:
        print(event.payload["delta"], end="", flush=True)
    elif event.type is EventType.TOOL_CALL:
        print(f"\n[calling {event.payload['name']}]")
```

Event types (`yaab.EventType`) and when the runner emits them:

| Type | When | Key payload |
|---|---|---|
| `RUN_START` | run begins | `prompt` |
| `USER_MESSAGE` | user turn recorded | `content` |
| `TEXT_DELTA` | token delta (only on `stream_events` / `Runner.stream_run`) | `delta` |
| `MODEL_DELTA` | a reasoning/thinking trace arrives | `reasoning` |
| `MODEL_RESPONSE` | a model reply | `content`, `tool_calls` |
| `TOOL_CALL` | a tool is invoked | `name`, `arguments` |
| `TOOL_RESULT` | a tool returns | `name`, `result` |
| `AGENT_TRANSFER` | run delegated to a sub-agent | `to` |
| `FINAL_OUTPUT` | final answer produced | `output` |
| `RUN_END` | run complete | `result` |
| `ERROR` | run failed | `error` |

The non-streaming `run`/`run_sync` collect the same events (minus `TEXT_DELTA`)
into `result.events`, so you can audit a finished run the same way you'd watch a
live one.

### Reasoning traces

When a provider exposes a reasoning/thinking trace (o-series, DeepSeek R1,
Anthropic extended thinking), it is captured on `ModelResponse.reasoning` and
emitted as a `MODEL_DELTA` event carrying a `reasoning` payload — so you can
surface the model's thinking without parsing it out of the answer.

## Streaming over HTTP (SSE)

The FastAPI app exposes both streams:

```
POST /run/stream    # semantic events as SSE (event: <type>, data: <json>)
POST /chat/stream   # token deltas as SSE (data: <token>), terminated by [DONE]
```

```python
from yaab.serve import fastapi_server_app
app = fastapi_server_app(agent)   # mount with uvicorn / your ASGI server
```

Consume with any SSE client:

```javascript
const es = new EventSource("/run/stream");  // or fetch() streaming for POST
es.addEventListener("tool_call", e => console.log(JSON.parse(e.data)));
es.addEventListener("final_output", e => console.log(JSON.parse(e.data)));
```
