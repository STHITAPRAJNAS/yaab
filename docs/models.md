# Models

The model layer is a thin `ModelProvider` protocol. The default implementation,
`LiteLLMModel`, gives you one OpenAI-compatible interface to 100+ providers and
thousands of models, plus fallbacks, retries, and cost tracking.

## Selecting a model

A string is treated as a LiteLLM model id:

```python
Agent("a", model="openai/gpt-4o")
Agent("a", model="anthropic/claude-sonnet-4-6")
Agent("a", model="bedrock/anthropic.claude-3-sonnet")
Agent("a", model="gemini/gemini-2.0-flash")
Agent("a", model="ollama/llama3")
```

Or pass a provider instance for full control:

```python
from yaab.models import LiteLLMModel

model = LiteLLMModel(
    "openai/gpt-4o",
    fallbacks=["anthropic/claude-sonnet-4-6", "gemini/gemini-2.0-flash"],
    max_retries=3,
    temperature=0.2,
    track_cost=True,
)
Agent("a", model=model)
```

Fallbacks are tried in order; each is retried with exponential backoff before
moving on. Cost is computed per call via LiteLLM and recorded into `Usage` and
the audit log.

## Structured output

When `output_type` is set, YAAB passes a JSON schema to the provider and
validates the response, retrying with the validation error on failure
(`output_retries`). See [Quickstart](quickstart.md#typed-validated-output).

## Streaming

```python
async for token in agent.stream("tell me a joke"):
    print(token, end="", flush=True)
```

See [Streaming & events](streaming-events.md) for the full picture.

## Observability

Every model call is wrapped in an `InstrumentedModel` that emits an
OpenTelemetry span following the **GenAI semantic conventions**
(`gen_ai.system`, `gen_ai.request.model`, token + cost attributes). Install the
extra and configure an exporter:

```bash
pip install 'yaab[otel]'
```

Disable wrapping per agent with `Agent(..., instrument=False)`.

## Testing without a network

`TestModel` and `FunctionModel` (in `yaab.testing`) are deterministic doubles:

```python
from yaab.testing import TestModel, FunctionModel

TestModel("fixed answer")                                  # always returns this
TestModel(call_tools=["search"], custom_output="done")     # exercises the tool loop
TestModel(structured_output={"city": "Paris"})             # for output_type tests
FunctionModel(lambda messages: messages[-1].content[::-1]) # compute the reply
```

## Pointing at a LiteLLM proxy

Set `api_base` (and `api_key`) on `LiteLLMModel`, or configure LiteLLM's env
vars, to route through a central LiteLLM proxy/gateway for shared keys, budgets,
and rate limits — optional, never required.
