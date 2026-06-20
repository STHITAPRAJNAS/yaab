# Record & replay

`CassetteModel` records a real model's responses to a JSON **cassette** once, then
replays them offline — so tests exercise true model behaviour with no API keys, no
cost, and no flakiness. It wraps any model exactly like the resilience wrappers do.

## Record once

Set a real key, run with `mode="record"` and an `inner` model. Every completion and
stream is appended to the cassette file (provider keys are stripped before
writing — they never land in a cassette).

```python
from yaab import Agent
from yaab.testing import CassetteModel

# Record once (with a real key in the environment):
model = CassetteModel("tests/cassettes/run.json", inner=real_model, mode="record")
agent = Agent("a", model=model)
```

## Replay forever after

In CI, construct the model with **no `inner`** and `mode="replay"`. Requests are
matched by a hash of the normalized request; an unmatched request raises
`CassetteMiss` rather than silently calling a provider.

```python
from yaab.testing import use_cassette

# CI: no inner — replays the committed cassette deterministically.
with use_cassette("tests/cassettes/run.json") as model:
    agent = Agent("a", model=model)
```

`use_cassette` picks the mode from the environment: with `YAAB_RECORD=1` **and** an
`inner` it records; otherwise it replays. So a developer records once locally and
CI replays the committed file.

## Modes

| mode | request on file | request not on file |
|------|-----------------|---------------------|
| `record` | overwrite (call `inner`) | call `inner`, append |
| `once` (default) | replay | call `inner`, append |
| `replay` | replay | raise `CassetteMiss` |

Repeated identical requests (a multi-step tool loop) replay in recorded order.
Streaming is recorded and replayed chunk-for-chunk.
