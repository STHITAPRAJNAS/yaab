# YAAB samples

End-to-end sample agents and patterns, each in its own package. Every sample
runs **fully offline** with a deterministic `TestModel` (so it — and its test —
works in CI with no API key), and runs against a **real model** by setting one
environment variable or passing a model string.

## Catalog

| Sample | Pattern | YAAB features shown |
|---|---|---|
| [`personal_assistant`](personal_assistant/) | Chat assistant with durable history + memory | **SQLite `SessionService`** (persistent history), long-term **`MemoryManager`** recall folded into the prompt, **callbacks** (`Plugin` writeback + usage logging) |
| [`memory_patterns`](memory_patterns/) | Episodic vs. long-term memory, done right | per-session **episodic** history vs. cross-session **long-term** memory, and `add_session_to_memory` consolidation; `(app_name, user_id)` scoping |
| [`multi_agent_state`](multi_agent_state/) | Hand data from one agent to the next | cross-agent state via shared, typed **`deps`** (`ctx.deps`) through a `SequentialAgent` |
| [`customer_support`](customer_support/) | Knowledge-grounded support bot that can act | RAG (`KnowledgeBase`), a side-effecting tool, enforcing **governance** (registry + guardrails + audit) |
| [`approval_pipeline`](approval_pipeline/) | Workflow that pauses for human approval | durable `StateGraph`, `Checkpointer`, `ctx.interrupt()` HITL |
| [`triage_swarm`](triage_swarm/) | Front-line agent routes to a specialist | `Swarm` autonomous hand-off |
| [`coding_helper`](coding_helper/) | Agent runs code, gated for safety | sandboxed `python_exec` + `ToolApprovalPlugin` (defense in depth) |

## Run a sample

Offline (deterministic, no key):

```bash
python -m samples.personal_assistant
python -m samples.memory_patterns
python -m samples.multi_agent_state
python -m samples.customer_support
python -m samples.approval_pipeline
```

Against a real model — set `YAAB_SAMPLE_MODEL` (any LiteLLM model id):

```bash
# Local & free (no API key) — run Ollama, then:
export YAAB_SAMPLE_MODEL=ollama/llama3
python -m samples.personal_assistant

# Free hosted tiers (set the provider's API key per LiteLLM docs):
export YAAB_SAMPLE_MODEL=gemini/gemini-2.0-flash     # Google AI Studio free tier
export YAAB_SAMPLE_MODEL=groq/llama-3.3-70b-versatile # Groq free tier
```

Each sample also exposes a `build(model=...)` you can call from your own code,
and an async `run(...)` convenience.

## How they're tested

`tests/test_samples.py` runs every sample on the offline `TestModel` /
`FunctionModel` and asserts the expected behavior (the assistant recalls a fact
in a new session, episodic vs. long-term memory behave as described, the ticket
written by one agent is read by the next, the approval gate executes or rejects,
the swarm routes to the right specialist, the coding helper computes `45`, …).
Because the offline models are scripted, the assertions are deterministic —
these are validated, not illustrative.

## Use one as a starting point

Copy a sample directory, swap `resolve_model(...)`'s offline default for your
model, replace the demo data/tools with yours, and you have a working app. The
patterns compose: e.g. give `customer_support`'s bot the
`personal_assistant`'s SQLite sessions + long-term memory, or gate
`coding_helper`'s tool inside an `approval_pipeline` graph.
