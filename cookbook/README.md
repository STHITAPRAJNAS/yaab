# YAAB cookbook

Mature, runnable examples for every major capability. Each **recipe** is one
focused file; each **app** is a realistic multi-feature application. Everything
runs offline with a deterministic `TestModel` (so it and its CI test need no
key), and runs live by setting one env var:

```bash
export YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash   # any LiteLLM model id
python -m cookbook.recipes.agents
```

## Recipes

| # | Recipe | Capability |
|---|--------|------------|
| 01 | [`agents`](recipes/agents.py) | The typed `Agent` unit of work |
| 02 | [`tools`](recipes/tools.py) | Give an agent a typed function to call |
| 03 | [`typed_output`](recipes/typed_output.py) | Get a validated Pydantic object back |
| 04 | [`sessions`](recipes/sessions.py) | Durable, multi-turn conversations |
| 05 | [`memory`](recipes/memory.py) | Long-term recall, scoped per user |
| 06 | [`rag`](recipes/rag.py) | Ground answers in your documents, with citations |
| 07 | [`hybrid_retrieval`](recipes/hybrid_retrieval.py) | Fuse sparse (BM25) + dense recall |
| 08 | [`streaming`](recipes/streaming.py) | Token-by-token output |
| 09 | [`sequential`](recipes/sequential.py) | Fixed steps, each building on the last |
| 10 | [`parallel`](recipes/parallel.py) | Fan out the same input to several agents |
| 11 | [`map_agent`](recipes/map_agent.py) | One agent across many inputs concurrently |
| 12 | [`loop`](recipes/loop.py) | Re-run until a quality bar (or a cap) |
| 13 | [`swarm`](recipes/swarm.py) | Autonomous hand-off between peers |
| 14 | [`conditions`](recipes/conditions.py) | Guard steps with `when=`/`stop=`/`else_=` |
| 15 | [`router`](recipes/router.py) | Deterministic exclusive choice, zero model calls |
| 16 | [`flow`](recipes/flow.py) | Explicit, durable, branchable control flow |
| 17 | [`hitl`](recipes/hitl.py) | Gate a sensitive tool on human approval |
| 18 | [`durable_runs`](recipes/durable_runs.py) | Runs as durable records; multi-pod backends |
| 19 | [`spend_governance`](recipes/spend_governance.py) | Per-identity / per-tenant budget caps |
| 20 | [`governance`](recipes/governance.py) | Guardrails that block unsafe input |
| 21 | [`evaluation`](recipes/evaluation.py) | Score outputs against expectations |
| 22 | [`optimization`](recipes/optimization.py) | Tune a prompt at build time, freeze it |
| 23 | [`serving`](recipes/serving.py) | Expose an agent over HTTP |
| 24 | [`openai_compat`](recipes/openai_compat.py) | Serve `/v1/chat/completions` for OpenAI clients |
| 25 | [`record_replay`](recipes/record_replay.py) | Deterministic offline tests of real model behaviour |
| 26 | [`browser`](recipes/browser.py) | Drive a real browser, safely (allowlist-gated) |
| 27 | [`interop`](recipes/interop.py) | Expose YAAB tools over MCP |

## Apps

| App | Scenario |
|-----|----------|
| _(coming in Wave 3)_ | |

## How they're tested

`tests/test_cookbook.py` auto-discovers every recipe and runs it offline, so a
recipe that imports clean but does the wrong thing fails CI. Each recipe also
self-checks its result with `cookbook._harness.expect`.
