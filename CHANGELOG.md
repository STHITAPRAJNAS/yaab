# Changelog

All notable changes to YAAB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-06-01

First public release: `pip install yaab-sdk` → `import yaab` / `$ yaab`.

### Core
- Type-safe `Agent[Deps, Output]` with dependency injection, structured-output
  validation + reflection/retry, and an event-driven `Runner`.
- Universal model layer over LiteLLM (fallbacks, retries, cost tracking) plus a
  deterministic `TestModel`/`FunctionModel` for offline use.
- `Agent(model_settings=…)` — forward arbitrary provider kwargs (temperature,
  top_p, seed, max_tokens, reasoning_effort, extra_body, …) on every model call.
- First-class multimodal `Content`/`Part` type; token + semantic-event streaming
  and SSE endpoints.
- **Streaming through the tool loop** — `Agent.stream_events` / `Runner.stream_run`
  yield `TEXT_DELTA` token deltas live AND run tools mid-run across multiple
  steps; `LiteLLMModel.stream` surfaces streamed tool calls.
- `tool_choice` (auto / required / pinned tool), reasoning-trace capture, and a
  tool-arg repair hook.
- **Run lifecycle control** — `UsageLimits` (including `max_wall_seconds`),
  external mid-run cancellation via `CancellationToken`, per-tool timeouts, and
  `Agent.reset()` for reuse.
- **Parallel tool execution** — a turn's multiple tool calls run concurrently
  with deterministic event order; opt out with `Agent(parallel_tools=False)`.
- **Resumable fast-path runs** — `Runner(run_checkpointer=…)` plus a `resume_id`
  makes the model-driven loop fault-tolerant: progress is checkpointed after
  every completed step, a crashed run resumes without re-requesting captured
  model turns, and a finished `resume_id` replays its result idempotently.
- **Context caching (write-side)** — `LiteLLMModel(cache_system_prompt=True,
  cache_tools=True)` injects Anthropic `cache_control` breakpoints (and passes
  Gemini `cached_content` through) so large stable prefixes bill at cached rates.
- **`ModelRouter`** — route each request to a cheap or capable model via a
  built-in length/complexity classifier or any custom callable.

### Orchestration
- Model-driven fast path; durable `StateGraph` with checkpointing, channel
  reducers, BSP supersteps, human-in-the-loop interrupt/resume, per-node
  `RetryPolicy` (exponential backoff), and a selectable
  `engine="rust"|"python"|"auto"`.
- Multi-agent: `SequentialAgent`, `ParallelAgent`, `MapAgent`, `LoopAgent`,
  `Swarm`, and agent-as-tool.
- **Sub-agent delegation** — `Agent(sub_agents=[...], transfer_depth=…)`
  auto-injects a `transfer_to_agent` tool; the LLM routes by each sub-agent's
  description and the chosen sub-agent's answer becomes the run's output.
- Optional `Signature`/`Module`/`Optimizer` programs (BootstrapFewShot,
  BootstrapFewShotWithRandomSearch, minibatched MIPROv2, GEPA) compiling to
  frozen artifacts.

### Governance
- Agent registry (SQLite or remote/central via HTTP) + model inventory,
  lifecycle FSM with evidence gates, guardrail/policy engine, tamper-evident
  hash-chained audit log, pre-tool authorization + idempotency, fast-path human
  approval, drift/trust monitoring, and compliance mappers for SR 11-7,
  EU AI Act, NIST AI RMF, ISO/IEC 42001, and SOC 2.
- **Industry guardrail adapters** — `PresidioPIIScanner`, `LLMGuardScanner`, and
  `NeMoGuardrailsScanner` behind the same `GuardrailScanner` protocol (optional
  extras `yaab-sdk[presidio]` / `yaab-sdk[llm-guard]` / `yaab-sdk[nemo]`).
- **Tool-level auth** — `ToolAuth`/`ToolCredential` on any tool (`@tool(auth=…)`):
  api-key/bearer/OAuth2/basic credentials resolved per call, injected as hidden
  params; missing credentials surface a consent-URL error the agent can relay.

### Evaluation
- Built-in metrics + RAGAS and DeepEval adapters; `Dataset`/`Experiment` runner.
- **Portable evalsets** — `EvalSet`/`EvalCase` with `.evalset.json` save/load;
  `ToolTrajectoryMatch` scores the agent's actual tool-call sequence against an
  expected trajectory.
- **`yaab eval` CLI** — score an agent against an evalset with auto-selected or
  explicit metrics, JSON reports, and a `--fail-under` CI gate.
- **User-simulation evals** — `UserSimulator` / `simulate` / `simulate_evalset`
  drive a multi-turn conversation against the agent under test with an LLM
  persona pursuing a goal, then score goal achievement.

### Memory & RAG
- Sessions/memory/artifacts + scoped managers, prefix-scoped state
  (`app:`/`user:`/`temp:`), namespace-aware long-term memory isolated per user.
- **Memory intelligence** — `MemoryManager.add_session_to_memory(extract=True)`
  distills durable memories from a session via one LLM call, with
  cosine-similarity consolidation; `KnowledgeBaseMemory` makes long-term memory
  durable on any vector-store backend.
- Built-in, provider-neutral RAG: `Document`/`Chunk`, chunkers, embedders (with
  caching + auto-upgrade to `LiteLLMEmbedder` when a provider key is present),
  8 vector-store backends (in-memory, pgvector/Aurora, Chroma, Qdrant,
  OpenSearch, Oracle, Pinecone, Weaviate), rerankers, and a `KnowledgeBase` with
  per-user access control, citations, dedup/incremental indexing, retrieval
  guardrails, and faithfulness evaluation.

### Tools & interop
- **Built-in tool catalog** — calculator, time, HTTP fetch, sandboxed Python
  exec, sandboxed file read/write/list, `fetch_url` (URL → readable page text),
  web search with a keyless DuckDuckGo provider, and provider-native
  search-grounding settings; all registered in the component registry.
- **OpenAPI toolset** — `openapi_toolset(spec)` turns any OpenAPI 3.x spec into
  agent tools, one per operation.
- MCP client + MCP server (tools, resources, prompts); A2A server + outbound
  `RemoteAgent` client with task polling; AG-UI middleware.

### Serving & deployment
- `fastapi_server_app` — native, A2A, and discovery endpoints with pluggable
  auth; **run-management API** (background runs, status polling, remote
  cancellation, run listing).
- **Dev console** — `yaab web` serves a single-page playground (chat, live event
  stream, agent info).
- **`yaab deploy` CLI** — generate (and optionally execute) Dockerfile /
  Cloud Run / Fly.io deployment artifacts from an agent spec; plan-by-default,
  with secrets always operator-supplied, never read from the local environment.
- **Turn-based voice agents** — `VoiceAgent` (speech-to-text → agent loop →
  text-to-speech) with streaming transcripts and injectable
  `Transcriber`/`Speaker` protocols.
- Batch/offline inference, resilience (rate limit + circuit breaker), YAML
  agent/runner specs (`agent_from_yaml`, `runner_from_dict`).

### Platform & quality
- `yaab-core` Rust performance core (PyO3, abi3 — one wheel for CPython 3.11+),
  with a high-performance async-first **pure-Python fallback** so the SDK runs
  everywhere (`YAAB_NO_RUST=1` exercises it in CI).
- Component registry + entry points make every concern (models, tools, stores,
  rerankers, embedders, sinks, metrics, compliance mappers) a plug-in.
- OTel GenAI tracing with Langfuse/Logfire/OTel audit sinks.
- Seven runnable sample apps, nine example scripts, and a documentation site —
  all tested in CI (examples run as subprocesses *and* in-process; doc snippets
  are import-checked) on Linux and Windows.
- Live end-to-end harnesses (`scripts/live_e2e.py`, `scripts/live_wave3_check.py`)
  for verification against real models.

### Packaging
- Publishes to PyPI as **`yaab-sdk`** (the bare name `yaab` is held by an
  unrelated project). The import package and CLI are `yaab`.
- The optional Rust accelerator publishes separately as **`yaab-core`**
  (`pip install 'yaab-sdk[rust]'`); the SDK falls back to its pure-Python core
  when absent.
- Releases are cut from the `main` branch via version tags; `release.yml`
  publishes through PyPI Trusted Publishing (OIDC) with a tag↔version gate and
  a built-wheel smoke test.

[Unreleased]: https://github.com/sthitaprajnas/yaab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sthitaprajnas/yaab/releases/tag/v0.1.0
