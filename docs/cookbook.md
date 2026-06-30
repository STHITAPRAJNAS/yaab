# Cookbook

Mature, runnable examples for every major capability, in the
[`cookbook/`](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook) tree.
Each **recipe** is one focused file; each **app** is a realistic multi-feature
application. Everything runs offline with a deterministic `TestModel` (so it and
its CI test need no key) and runs live by setting `YAAB_SAMPLE_MODEL`.

```bash
export YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash   # any LiteLLM model id
python -m cookbook.recipes.agents
python -m cookbook.apps.support_desk
```

## Flagship apps

Realistic applications that combine several capabilities end-to-end.

| App | Scenario | Capabilities |
|-----|----------|--------------|
| [`support_desk`](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook/apps/support_desk) | Grounded, governed, safe-to-act support | RAG · tools · HITL approval · per-tenant spend caps · OpenAI-compat serving |
| [`refund_pipeline`](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook/apps/refund_pipeline) | Durable workflow with a human gate | Flow · pause→decide→resume · durable checkpoint + approval store |
| [`research_assistant`](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook/apps/research_assistant) | Durable multi-step research | Flow · hybrid (BM25+dense) retrieval · streaming · browser + record/replay (live) |
| [`coding_agent`](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook/apps/coding_agent) | Plan, then run code in a sandbox | multi-agent (planner→coder) · sandboxed exec · tool approval |

## Recipes

One focused, self-checking example per capability (`python -m cookbook.recipes.<name>`).

| Recipe | Capability | Docs |
|--------|------------|------|
| `agents` | The typed `Agent` unit of work | [Agents](agents.md) |
| `tools` | Give an agent a typed function to call | [Tools](tools.md) |
| `typed_output` | Get a validated Pydantic object back | [Agents](agents.md) |
| `sessions` | Durable, multi-turn conversations | [State](state.md) |
| `memory` | Long-term recall, scoped per user | [State](state.md) |
| `rag` | Ground answers in your documents | [RAG](rag.md) |
| `hybrid_retrieval` | Fuse sparse (BM25) + dense recall | [RAG](rag.md) |
| `streaming` | Token-by-token output | [Streaming](streaming-events.md) |
| `sequential` | Fixed steps, each building on the last | [Multi-agent](multi-agent.md) |
| `parallel` | Fan out the same input to several agents | [Multi-agent](multi-agent.md) |
| `map_agent` | One agent across many inputs | [Multi-agent](multi-agent.md) |
| `loop` | Re-run until a quality bar (or a cap) | [Multi-agent](multi-agent.md) |
| `swarm` | Autonomous hand-off between peers | [Multi-agent](multi-agent.md) |
| `conditions` | Guard steps with `when=`/`stop=`/`else_=` | [Conditions](conditions.md) |
| `router` | Deterministic exclusive choice | [Conditions](conditions.md) |
| `flow` | Explicit, durable control flow | [Flow](flow.md) |
| `hitl` | Gate a sensitive tool on approval | [Human-in-the-loop](hitl.md) |
| `durable_runs` | Runs as durable records; multi-pod | [Durable runs](durable-runs.md) |
| `spend_governance` | Per-identity / per-tenant budget caps | [Governance](governance.md) |
| `governance` | Guardrails that block unsafe input | [Governance](governance.md) |
| `evaluation` | Score outputs against expectations | [Evaluation](evaluation.md) |
| `optimization` | Tune a prompt at build time, freeze it | [Optimization](optimization.md) |
| `serving` | Expose an agent over HTTP | [Serving](serving.md) |
| `openai_compat` | Serve `/v1/chat/completions` | [OpenAI-compatible API](openai-compat.md) |
| `record_replay` | Deterministic offline tests of real runs | [Record & replay](record-replay.md) |
| `browser` | Drive a real browser, safely | [Tools](tools.md) |
| `interop` | Expose YAAB tools over MCP | [Interop](interop.md) |
| `harness` | Sandboxed, approval-gated coding agent that edits a file | [Coding harness](harness.md) |

## How they're tested

`tests/test_cookbook.py` auto-discovers every recipe and app and runs it offline,
and each one self-checks its result — so an example that imports clean but does
the wrong thing fails CI.
