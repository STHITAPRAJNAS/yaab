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

`run_stream` yields a typed `Event` per step of the full loop:

```python
runner = agent._get_runner()
async for event in runner.run_stream(agent, "What's the weather in Paris?"):
    print(event.type, event.payload)
```

Event types (`yaab.EventType`):

| Type | When | Key payload |
|---|---|---|
| `RUN_START` | run begins | `prompt` |
| `USER_MESSAGE` | user turn recorded | `content` |
| `MODEL_DELTA` | token delta, or a reasoning trace | `delta` / `reasoning` |
| `MODEL_RESPONSE` | a model reply | `content`, `tool_calls` |
| `TOOL_CALL` | a tool is invoked | `name`, `arguments` |
| `TOOL_RESULT` | a tool returns | `name`, `result` |
| `GUARDRAIL` | a guard fires | `scanner`, `action` |
| `FINAL_OUTPUT` | final answer produced | `output` |
| `RUN_END` | run complete | `result` |
| `ERROR` | run failed | `error` |

The non-streaming `run`/`run_sync` collect these into `result.events`.

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
