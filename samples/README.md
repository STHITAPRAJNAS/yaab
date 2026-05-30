# YAAB samples

End-to-end sample agents and patterns, each in its own package. Every sample
runs **fully offline** with a deterministic `TestModel` (so it — and its test —
works in CI with no API key), and runs against a **real model** by setting one
environment variable or passing a model string.

## Catalog

| Sample | Pattern | YAAB features shown |
|---|---|---|
| [`customer_support`](customer_support/) | Knowledge-grounded support bot that can act | RAG (`KnowledgeBase`), a side-effecting tool, enforcing **governance** (registry + guardrails + audit) |
| [`research_assistant`](research_assistant/) | Staged pipeline (researcher → writer) | `SequentialAgent` multi-agent, output piping |
| [`document_qa`](document_qa/) | Answer from documents with citations | RAG retrieval, `augment()`, source attribution |
| [`approval_pipeline`](approval_pipeline/) | Workflow that pauses for human approval | durable `StateGraph`, `Checkpointer`, `ctx.interrupt()` HITL |
| [`triage_swarm`](triage_swarm/) | Front-line agent routes to a specialist | `Swarm` autonomous hand-off |
| [`coding_helper`](coding_helper/) | Agent runs code, gated for safety | sandboxed `python_exec` + `ToolApprovalPlugin` (defense in depth) |

## Run a sample

Offline (deterministic, no key):

```bash
python -m samples.customer_support
python -m samples.approval_pipeline
```

Against a real model — set `YAAB_SAMPLE_MODEL` (any LiteLLM model id):

```bash
# Local & free (no API key) — run Ollama, then:
export YAAB_SAMPLE_MODEL=ollama/llama3
python -m samples.research_assistant

# Free hosted tiers (set the provider's API key per LiteLLM docs):
export YAAB_SAMPLE_MODEL=gemini/gemini-2.0-flash     # Google AI Studio free tier
export YAAB_SAMPLE_MODEL=groq/llama-3.3-70b-versatile # Groq free tier
```

Each sample also exposes a `build(model=...)` you can call from your own code,
and an async `run(...)` convenience.

## How they're tested

`tests/test_samples.py` runs every sample on the offline `TestModel` and asserts
the expected behavior (the support bot cites refunds, the approval gate executes
or rejects, the swarm routes to the right specialist, the coding helper computes
`45`, …). Because the offline models are scripted, the assertions are
deterministic — these are validated, not illustrative.

## Use one as a starting point

Copy a sample directory, swap `resolve_model(...)`'s offline default for your
model, replace the demo data/tools with yours, and you have a working app. The
patterns compose: e.g. put `document_qa`'s `KnowledgeBase` behind
`customer_support`'s governance, or gate `coding_helper`'s tool inside an
`approval_pipeline` graph.
