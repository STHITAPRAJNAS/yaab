# OpenAI-compatible API

Serve any yaab agent behind the OpenAI **Chat Completions** API, so existing
OpenAI-SDK clients, eval harnesses, and gateways drive it with no code change —
only a `base_url` swap.

## Serve it

```python
from yaab import Agent, openai_compat_app

support = Agent("support", instructions="Help with billing.")
app = openai_compat_app({"support": support})   # uvicorn module:app
```

The OpenAI `model` string selects an agent by name; pass a single `Agent` to
serve it as the default. To add the OpenAI routes to the native server instead,
use the flag:

```python
from yaab import Agent
from yaab.serve import fastapi_server_app

app = fastapi_server_app(Agent("support"), openai_compat=True)
```

## Call it with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")
reply = client.chat.completions.create(
    model="support",
    messages=[{"role": "user", "content": "Where is my invoice?"}],
)
print(reply.choices[0].message.content)
```

- **Streaming:** pass `stream=True` for `chat.completion.chunk` SSE deltas,
  terminated by `data: [DONE]`.
- **`GET /v1/models`** lists the served agents.
- **Auth:** the server's `AuthScheme` validates `Authorization: Bearer …`; the
  resolved identity flows into the run (audit, governance, spend).

## Tools

Without `tools` in the request, the **agent's own tools run internally** and you
get the final answer (`finish_reason="stop"`). With `tools` in the request, a
single model turn surfaces `tool_calls` for you to execute (OpenAI
function-calling passthrough, `finish_reason="tool_calls"`).
