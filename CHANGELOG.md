# Changelog

All notable changes to YAAB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-06-09

A major feature release on top of 0.1.0: one coherent orchestration model with
seven native patterns, durable multi-replica execution, a unified
human-in-the-loop surface, and a broad set of capability additions. Backward
compatible — existing 0.1.0 code keeps working.

### Added — Orchestration: one model, seven native patterns
- **One shared `State` per run** — every agent, tool, and step in a run reads and
  writes the same prefix-scoped `State` (`app:`/`user:`/`temp:`). The Runner
  builds it once (from the session when present) and every child in
  Sequential/Parallel/Map/Loop/Swarm inherits the same object, so a value written
  in one step is read by the next.
- **`writes=` output capture & `{key}` instructions** — `Agent(writes="key")`
  lands an agent's *typed* output into shared state; string instructions
  substitute `{key}` / `{key?}` from that state before the model call. The
  declarative inter-agent handoff, with no prompt-string piping.
- **Unified conditions** — one `Condition` concept (a callable **or** a safe,
  sandboxed expression string) over a read-only state view, with `&`/`|`/`~`
  combinators; `when=` (input guard), `stop=` (output guard), and `else=`
  (fallback) on every pattern; decision events carry the resolved operands so
  *why* a step was skipped is answerable from the trace.
- **`RouterAgent`** — a seventh workflow pattern for exclusive choice: run exactly
  one of N branches by input guard, first-match-wins with a mandatory default,
  zero model calls to route, `from_picker` with build-time typo-safety.
- **`Flow`** — explicit, durable, branchable control flow (`.step/.then/.route/
  .loop/.fan_out/.start_at/.returns`) that lowers onto the durable graph engine.
  It owns no state/checkpoint/pause of its own: it threads the one `State`, routes
  on the one `Condition`, and pauses into the one `Pending`. `RunHistory` adds
  time-travel: list a run's checkpoints, inspect any (including a paused) one, and
  fork-from-checkpoint into a new thread.

### Added — Durable runtime, multi-pod, and human-in-the-loop
- **Durable background runs that survive restarts and span replicas** — a run is
  now a durable record in a swappable `RunStore` (in-memory, SQLite, Postgres,
  Redis) instead of an in-process task: poll it, cancel it from any replica, and
  resume it from its last completed step after a crash. `RunWorker` drains the
  queue with bounded concurrency, heartbeat leases, and crash recovery (an
  abandoned run is re-queued and picked up by another replica), so background
  work no longer dies on restart or a rolling deploy. Cross-replica cancel flows
  through `StoreCancellationToken`.
- **One human-in-the-loop idiom: pause → decide → resume** — a guarded tool, an
  `ask_user` question, or a `Flow` pause all surface one typed `Pending`; a human
  decides with `approvals.approve`/`deny`/`edit`/`respond` (returning a
  self-correlating `Decision`); `agent.run(resume=decision)` continues — needing
  only the `approval_id`, so it resumes durably from a *fresh process or replica*.
  Decisions are validated before they mutate and are first-write-wins
  (double-approve resumes exactly once).
- **Run any number of replicas behind a load balancer, safely** —
  `durable_backends(dsn=…, redis_url=…)` wires every stateful concern to shared
  storage in one call; a startup guardrail warns loudly if any backend is still
  in-memory while running more than one replica; a shared `RedisRateLimiter`
  keeps a `rate=N` budget global across replicas.
- **A debugger that replays a run with per-step model/tool/token/cost/latency** —
  an opt-in `TraceStore` persists each run's timeline; `yaab web` renders a span
  waterfall (latency bars, cost badges), an approvals tab, a session-state
  inspector, and run replay. Round-trip AG-UI emits `STATE_SNAPSHOT`/`STATE_DELTA`
  and accepts human input back via `resume_agui`.
- **Durable schedules and durable artifacts** — a `CronStore` materializes due
  schedules into runs; `SQLiteArtifactService` / `PostgresArtifactService` /
  `RedisArtifactService` make artifacts durable across replicas. `resume_id` is
  exposed on the public `Agent.run` API.

### Added — Capability breadth
- **Typed outputs from declarative agents** — a YAML/dict `output_type` resolves
  by name to a built-in scalar or a registered Pydantic model.
- **Reuse foreign tools** — `from_langchain_tool` / `from_crewai_tool` /
  `adapt_tool` wrap a tool from another ecosystem as a native tool (duck-typed,
  no extra install).
- **Relevance-filter context strategy** — `RelevanceFilter` keeps only history
  relevant to the latest message (injectable scorer), alongside truncate/summarize.
- **Per-agent callbacks** — `Agent(before_agent=, after_agent=)` fire around each
  agent's own loop (so they run for every agent in a composition); a declarative
  spec can wire `callbacks:` and `plugins:` by registered name.
- **Rubric judge & text-overlap metric** — `RubricJudge` scores against named
  criteria with a per-criterion breakdown; `ResponseMatch` is a deterministic
  ROUGE-style overlap metric.
- **Session rewind & migration** — roll a conversation back to a prior turn
  (`rewind`/`rewind_last`) or copy a session across backends (`migrate_session`).
- **Hybrid retrieval** — `KnowledgeBase(hybrid=True)` fuses a dense (embedding)
  and a sparse (BM25) recall by reciprocal rank, so exact rare terms surface.

### Verification
- Full suite green (1094 offline tests, ruff/format/mypy clean, 21/21 smoke);
  the v2/durability surface is additionally live-verified against a real model
  (`scripts/live_v2_check.py`, `live_durable_check.py`, `live_wave3_check.py`).

### Added — durable runtime (detail)
- **Durable background runs that survive restarts and span replicas** — a run is
  now a durable record in a swappable `RunStore` (in-memory, SQLite, Postgres,
  Redis) instead of an in-process task: poll it, cancel it from any replica, and
  resume it from its last completed step after a crash. `RunWorker` drains the
  queue with bounded concurrency, heartbeat leases, and crash recovery (an
  abandoned run is re-queued and picked up by another replica), so background
  work no longer dies on restart or a rolling deploy. Cross-replica cancel flows
  through `StoreCancellationToken`: a cancel issued anywhere stops the run on the
  replica executing it.
- **Out-of-band human sign-off for sensitive actions** — `ToolApprovalPlugin`
  gains a `queue` mode backed by a durable `ApprovalStore` (in-memory, SQLite,
  Postgres, Redis). A guarded tool call parks the run as a pending approval +
  checkpoint and pauses it — consuming zero compute while it waits — instead of
  blocking a thread. A reviewer approves or denies from any replica (HTTP
  `GET /approvals`, `POST /approvals/{id}/approve|deny`, `POST /runs/{id}/resume`)
  and the run resumes from exactly where it stopped, running the approved tool or
  feeding the denial back to the model. Pauses survive a restart.
- **Run any number of replicas behind a load balancer, safely** — `durable_backends(dsn=…, redis_url=…)`
  builds one coherent set of shared backends (sessions, artifacts, run store,
  approval store, trace store, checkpointer, audit sink, registry, rate limiter)
  all pointed at the same database, so making a deployment replica-safe is a
  single call you splat into `Runner` and the server. A startup guardrail
  (`warn_if_ephemeral`, wired into the server via `YAAB_REPLICAS`) shouts at boot
  if any backend is still in-memory while running more than one replica, turning
  silent data loss into a loud warning. A shared `RedisRateLimiter` keeps a
  `rate=N` budget *global* across replicas instead of per-replica.
- **A debugger that replays a run with per-step model/tool/token/cost/latency
  detail** — an opt-in `TraceStore` (in-memory, SQLite, Postgres, Redis) persists
  each run's timeline so it survives the run and a restart. New endpoints expose
  the full event trace (`GET /runs/{id}/events`), a computed span waterfall with
  durations, tokens, and cost (`GET /runs/{id}/trace`), and a session/run state
  inspector (`GET /runs/{id}/state`, `GET /sessions/{id}/state`). The web console
  gains Trace and State tabs and surfaces total tokens/cost/latency per run.
- **Durable schedules and join-on-the-fly** — a durable `CronStore` materializes
  due schedules into queued runs (`POST/GET/DELETE /crons`), per-run completion
  webhooks notify callers without polling, `GET /runs/{id}/stream` re-attaches to
  an in-flight or finished run (replay then tail), and `multitask_strategy`
  (`reject|enqueue|cancel`) controls overlapping runs on the same session.
- **Durable artifacts and `resume_id` on the public API** — `SQLiteArtifactService`
  / `PostgresArtifactService` / `RedisArtifactService` persist artifact bytes and
  version history across replicas, and `Agent.run`/`Agent.run_sync` now accept
  `resume_id` so a fault-tolerant run is resumable straight from the agent
  surface. `Runner(checkpoint_mode="step"|"final")` tunes how often progress is
  checkpointed.

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

[Unreleased]: https://github.com/sthitaprajnas/yaab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sthitaprajnas/yaab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sthitaprajnas/yaab/releases/tag/v0.1.0
