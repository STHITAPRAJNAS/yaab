# Closing the industry-SDK gaps — design & roadmap

**Date:** 2026-05-31
**Goal:** Bring YAAB to ADK/LangGraph/Pydantic-AI/DSPy-class maturity by closing
the gaps surfaced during live end-to-end testing, shipping "batteries included"
defaults, and adding an industry-standard guardrail-adapter layer.

Each phase is an independently-shippable spec → tested implementation cycle.
Phases A–E are committed; Phase F is scaffolded/spec'd only (multi-week or needs
external infra).

---

## Phase A — Runtime hardening

### A1. Parallel tool execution
**Problem:** `Runner.run_stream` executes a turn's tool calls sequentially
(`for tc in response.tool_calls: await …`). Models routinely emit several
parallel tool calls; serial execution is an N× latency hit. OpenAI Agents SDK
and ADK run them concurrently.

**Design:** When a response has >1 tool call, execute them concurrently with
`asyncio.gather`, preserving:
- **Deterministic event order** — `TOOL_CALL`/`TOOL_RESULT` events are emitted in
  the model's original call order even though execution overlaps. (Gather
  results, then emit events in order.)
- **Per-tool error isolation** — one tool raising still yields an `error: …`
  result for that call only (unchanged `_run_tool` semantics).
- **Plugins & limits** — `before_tool`/`after_tool`/`repair_tool_args` hooks,
  `ToolApprovalPlugin`, and per-tool usage caps run per call as today.
- **Opt-out** — `Agent(parallel_tools=False)` forces the old sequential path for
  ordering-sensitive tools (default `True`).
- **Bound** — optional `Agent(max_parallel_tools=int)` caps concurrency via a
  semaphore (default unbounded).

### A2. Per-tool timeout / cancellation
**Problem:** Only a run-level timeout exists; one hung tool blocks the whole run.

**Design:** `Tool` gains an optional `timeout: float | None`; `Runner` gains a
`default_tool_timeout`. Tool execution is wrapped in `asyncio.wait_for`. A
timeout becomes a normal `error: tool '<name>' timed out after <n>s` result (does
not crash the loop), and the existing `CancellationToken` is honored between/within
tool dispatch.

### A3. ParallelAgent session/identity propagation
**Problem:** `ParallelAgent.run` drops `session_id` when calling children.

**Design:** Thread `session_id` (and confirm `identity`) into children. Document
that concurrent children sharing one session may interleave appends — reads are
safe; for isolated write history, give each child its own session id.

### A4. Embedder footgun → auto-upgrade
**Problem:** The default embedder is a silent deterministic hashing stub; RAG /
long-term memory "work" but retrieve poorly, which looks like success.

**Design (auto-upgrade if key present):**
1. A `default_embedder()` factory: if `litellm` is importable **and** an
   embedding-provider key is in the environment (`OPENAI_API_KEY`,
   `COHERE_API_KEY`, `MISTRAL_API_KEY`, `VOYAGE_API_KEY`, …), default to a
   `LiteLLMEmbedder` with that provider's standard small embedding model; else
   fall back to the hashing stub.
2. When the hashing stub is selected, emit a **one-time** `logging.warning` that
   semantic recall will be weak and how to configure a real embedder.
3. **String shorthand:** `KnowledgeBase(embedder="openai/text-embedding-3-small")`
   and `MemoryManager(embedder="…")` build a `LiteLLMEmbedder` from the name.
4. The auto-upgrade makes a network call / incurs cost only because a key is
   present (explicit opt-in by configuration); offline/test runs (no key) stay on
   the deterministic stub with zero network.

**Testing:** auto-upgrade selection is unit-tested by monkeypatching env + a fake
litellm; the warning path and string shorthand are unit-tested; no network in CI.

---

## Phase B — Guardrails & the adapter pattern
First-class `GuardrailScanner` adapter layer + registry with **LLM-Guard**,
**NeMo Guardrails**, and **Presidio** (PII) adapters shipped as optional extras,
lazily imported. Detailed in its own spec when Phase A lands.

## Phase C — Streaming through the tool loop
Interleave token + `TOOL_CALL` + `TOOL_RESULT` events across a multi-step run.

## Phase D — Interop depth
A2A task streaming/polling + OAuth2 token exchange; MCP streamable-HTTP transport.

## Phase E — Optimizer depth
Move `MIPROv2`/`GEPA` closer to real DSPy search loops.

## Phase F — Scaffold/spec only (not faked)
Build + publish the `yaab-core` Rust wheel (maturin + CI + PyPI — partly owner's
infra); Rust Tokio actor bus; `yaab web` dev UI; time-travel fork/replay.

---

## Cross-cutting principles
- **TDD:** every change is a failing test → implementation → green, plus a live
  re-verify via `scripts/live_e2e.py` where a real model is in the loop.
- **No silent behavior changes:** defaults that add network/cost are gated on
  explicit configuration (a key in env counts as opt-in for A4 per decision).
- **Optional extras stay optional:** new integrations import lazily; the core
  install footprint does not grow.
