# Changelog

All notable changes to YAAB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `samples/` — six end-to-end sample apps & patterns (customer support, research
  assistant, document Q&A, approval pipeline, triage swarm, coding helper), each
  runnable offline and against a real/free model via `YAAB_SAMPLE_MODEL`, with a
  test that validates each on a deterministic model.
- `docs/concepts.md` — what every component is for, with disambiguation of the
  confusable pairs (Checkpointer vs Session, Memory vs RAG, authorization vs
  approval vs guardrails, optimizer vs evaluator).
- `scripts/live_e2e.py` — comprehensive live-LLM end-to-end harness (28 complex
  scenarios, provider-agnostic, rate-limit aware) complementing the offline
  `scripts/smoke_all.py`.
- `Runner(memory_app_name=...)` — the Runner now threads the run `identity` →
  `user_id` (and `memory_app_name` → `app_name`) into namespace-aware memory
  backends, so scoped long-term memory is reachable from the Agent path and
  isolated per user.
- **Parallel tool execution** — a turn's multiple tool calls now run concurrently
  (`asyncio.gather`) with deterministic event order; opt out with
  `Agent(parallel_tools=False)`, bound with `Agent(max_parallel_tools=N)`.
- **Per-tool timeouts** — `tool(timeout=…)` / `FunctionTool(timeout=…)` and
  `Runner(default_tool_timeout=…)`; a timeout becomes an `error:` tool result.
- **Embedder auto-upgrade** — the default embedder upgrades to a real
  `LiteLLMEmbedder` when litellm + an embedding-provider key are present
  (OpenAI/Gemini/Cohere/Mistral/Voyage), else falls back to the hashing stub with
  a one-time warning; `embedder="provider/model"` string shorthand on
  `KnowledgeBase`/`MemoryManager`/`InMemoryVectorMemory`.
- **Industry guardrail adapters** — `PresidioPIIScanner`, `LLMGuardScanner`, and
  `NeMoGuardrailsScanner` in `yaab.governance.guardrails`, behind the existing
  `GuardrailScanner` protocol and registered in the component registry (optional
  extras `yaab[presidio]` / `yaab[llm-guard]` / `yaab[nemo]`, imported lazily).

### Fixed
- `tool_choice="required"` (or a pinned tool name) no longer loops until
  `MaxStepsExceeded`: it forces the first model call only, then relaxes to
  `"auto"` so the model can finalize ("force at least one tool call").
- Structured-output streaming now tolerates Markdown code fences (```json …```)
  that providers like Gemini/Claude emit despite a JSON-only instruction —
  previously it yielded no partials.
- `output_retries` is no longer permanently decremented on the shared `Agent`
  across runs; the per-run retry budget is local, so a reused agent keeps its
  configured budget.
- `pip install 'yaab[all]'` no longer fails: the `all` extra no longer bundles
  the `rust` extra (the `yaab-core` accelerator wheel is published separately
  and has a pure-Python fallback). Install it explicitly with `yaab[rust]`.

### Changed
- Docs/README reworded to be descriptive rather than promotional (removed
  "differentiator"/marketing framing); refreshed the project-layout tree.

## [0.1.0] — initial alpha

First public release.

### Core
- Type-safe `Agent[Deps, Output]` with dependency injection, structured-output
  validation + reflection/retry, and an event-driven `Runner`.
- Universal model layer over LiteLLM (fallbacks, retries, cost tracking) plus a
  deterministic `TestModel`/`FunctionModel` for offline use.
- First-class multimodal `Content`/`Part` type; token + semantic-event streaming
  and SSE endpoints.
- `tool_choice`, reasoning-trace capture, and a tool-arg repair hook.

### Orchestration
- Model-driven fast path; durable `StateGraph` with checkpointing, channel
  reducers, BSP supersteps, human-in-the-loop interrupt/resume, and a selectable
  `engine="rust"|"python"|"auto"`.
- Multi-agent: `SequentialAgent`, `ParallelAgent`, `MapAgent`, `LoopAgent`,
  `Swarm`, and agent-as-tool.
- Optional DSPy-style `Signature`/`Module`/`Optimizer` (BootstrapFewShot,
  MIPROv2, GEPA) compiling to frozen artifacts.

### Governance (the differentiator)
- Agent registry + model inventory, lifecycle FSM, guardrail/policy engine,
  tamper-evident hash-chained audit log, pre-tool authorization + idempotency,
  fast-path human approval, drift/trust monitoring, and compliance mappers for
  SR 11-7, EU AI Act, NIST AI RMF, ISO/IEC 42001, and SOC 2.

### RAG
- Built-in, provider-neutral pipeline: `Document`/`Chunk`, chunkers, embedders
  (with caching), vector stores (in-memory, pgvector, Chroma, Qdrant),
  rerankers (keyword, LLM, cross-encoder), and a `KnowledgeBase` with per-user
  access control, citations, dedup/incremental indexing, retrieval guardrails,
  and faithfulness evaluation. Document loaders for txt/md/html/pdf/csv/json.

### Platform
- `yaab-core` Rust performance core (PyO3, abi3 — one wheel for CPython 3.11+),
  with a high-performance async-first **pure-Python fallback** so the SDK runs
  everywhere (`YAAB_NO_RUST=1` exercises it in CI).
- Sessions/memory/artifacts + managers, prefix-scoped state (`app:`/`user:`/`temp:`),
  plugins, prompt versioning, skills, pluggable auth, `fastapi_server_app` +
  A2A server/client, MCP client + server, AG-UI middleware, `yaab web` dev UI,
  batch/offline inference, resilience (rate limit + circuit breaker), YAML config.
- Extensible eval layer with RAGAS and DeepEval adapters; OTel GenAI tracing
  with Langfuse/Logfire/OTel audit sinks.
- Component registry + entry points make every concern (models, tools, stores,
  rerankers, embedders, sinks, metrics, compliance mappers) a plug-in.

[Unreleased]: https://github.com/sthitaprajnas/yaab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sthitaprajnas/yaab/releases/tag/v0.1.0
