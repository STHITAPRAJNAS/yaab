# Changelog

All notable changes to YAAB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
