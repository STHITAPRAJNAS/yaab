# YAAB vs Google ADK 2.0 — verified gap analysis

**Date:** 2026-06-01 · **Method:** 14-agent audit — 7 capability analyzers reading the YAAB
codebase against current ADK 2.0 documentation, then 7 adversarial verifiers attempting to refute
every claimed gap against the code. **77 gaps claimed, 77 confirmed, 0 refuted.**

**Severity distribution (after dedup): 19 high · 36 medium · 18 low**

Severity: *high* = production adopters will hit this; *medium* = parity/marketing matters; *low* = niche.
Status: *missing* = not present at all; *partial* = building blocks exist, not assembled; *weaker* = exists but shallower than ADK.

---

## Where YAAB is ahead of ADK (verified strengths)

### Knowledge base / RAG, Memory, Sessions, Artifacts
- Vector store breadth far exceeds ADK: yaab/rag/store.py + stores_external.py ship 8 swappable backends behind one Protocol (InMemory, pgvector/Aurora, Chroma, Qdrant, OpenSearch/Serverless, Oracle 23ai, Pinecone, Weaviate), all with metadata `where` push-down. ADK 2.0's first-party retrieval is Vertex-centric (VertexAiRagRetrieval / VertexAiSearch / Vertex RAG memory); YAAB is fully provider-neutral and self-hostable.
- Built-in, open RAG pipeline that ADK delegates to a managed cloud. yaab/rag/ ships chunkers (Character/Sentence/Paragraph, chunking.py), document loaders for txt/md/html/pdf/csv/json + directory + bytes (loaders.py), and dependency-free fallbacks (regex HTML stripper). ADK has no comparable open ingestion/chunking layer.
- Reranking layer ADK lacks entirely: rerank.py provides KeywordReranker (lexical+vector blend), LLMReranker (model-as-judge), and CrossEncoderReranker (sentence-transformers cross-encoder).
- RAG governance features ADK does not surface: retrieval-time per-tenant access control via metadata `where` (KnowledgeBase.as_tool scope_from_deps in knowledge.py), source citations (RetrievedChunk.citation in types.py), a context_guard hook + min_score retrieval guardrail (knowledge.py), incremental dedup indexing by content hash and reindex()/delete-by-source (knowledge.py), embedding caching (CachingEmbedder in embedders.py), and RAG faithfulness/groundedness + context-relevance evaluators both deterministic and LLM-judge (rag/eval.py) — ADK leaves groundedness eval to external tools.
- Session backend breadth meets or beats ADK's: InMemory + SQLite + Postgres(/Aurora) + Redis(/ElastiCache/MemoryDB/Azure Cache) (sessions/), each one-line swappable behind SessionService; ADK ships InMemory + DatabaseSessionService + VertexAiSessionService.
- Prefix-scoped state (app:/user:/temp:/session) is fully implemented as a routed MutableMapping in yaab/state.py with persisted()/temp separation and wired through SessionManager.resolve_state/save_state — true parity with ADK's scoped-state model, backend-agnostic.
- Pluggable embedders with auto-upgrade: default_embedder() in memory/__init__.py auto-selects a real LiteLLM provider embedding model when an API key is present (OpenAI/Gemini/Cohere/Mistral/Voyage) and warns once when falling back to the deterministic hashing stub — a fool-proofing nicety ADK doesn't have.
- Rust-accelerated top-k similarity (yaab._core.top_k) for in-memory vector recall and memory search, with a pure-Python fallback — addresses the 'fast' bar for the default path.

### Evals, Guardrails, Audit, Governance
- Tamper-evident hash-chained audit log (yaab/governance/audit.py): every event folds the prior event's hash via the Rust core (_core.hash_event), AuditLog.verify() detects retroactive edits, with InMemory/SQLite/Protocol-based sinks. ADK 2.0 has NO tamper-evident audit log at all.
- Agent registry as system-of-record (yaab/governance/registry.py): AgentCard superset of A2A card with 30+ governance fields (risk_tier, eu_act_category, decision_authority, approval_status, lineage, incident_history), InMemory/SQLite/Remote(HTTP) backends, inventory() view, /.well-known/agent.json export. ADK has no agent registry.
- SR 11-7-aligned lifecycle FSM (yaab/governance/lifecycle.py): 9-state finite-state machine with enforced legal transitions, per-state REQUIRED_EVIDENCE gating (validation_report, effective_challenge_signoff, etc.), every transition audited. ADK has nothing comparable.
- Five compliance mappers (yaab/governance/compliance/): SR 11-7, EU AI Act, NIST AI RMF, ISO 42001, SOC 2 — each projects registry+audit+lifecycle onto regime controls and emits ComplianceReport with coverage %, gaps, and markdown, pluggable via yaab.compliance entry point. ADK has zero compliance mapping.
- Behavioral drift detection + trust scoring (yaab/governance/monitor.py): DriftMonitor flags eval-score regressions vs a baseline window; TrustScorer folds eval performance + guardrail-block rate + error rate from the audit log into a 0-1 weighted trust score. ADK has no drift monitoring.
- Guardrail engine far richer than ADK (yaab/governance/policy.py + guardrails/): dependency-free built-in scanners (PromptInjection, PII, Secrets, Topics, SystemPromptLeak) with Action ladder (allow/redact/flag/block) and redaction-chaining, PLUS first-class adapters for Microsoft Presidio, Protect AI LLM-Guard, and NVIDIA NeMo Guardrails behind one GuardrailScanner protocol, all registry-discoverable. ADK relies on user-written callbacks + Google Cloud services with no built-in PII/injection scanners.
- Three-mode governance enforcement wired into the Runner (yaab/governance/service.py + runner.py): OFF/OBSERVE/ENFORCING; ENFORCING refuses unregistered/unapproved agents (check_registered) and raises PolicyViolation on BLOCK; input+output guardrails scan every run. ADK guardrails are purely opt-in callbacks with no registry gate.
- Pre-tool authorization + idempotency + HITL approval as composable Runner plugins (yaab/governance/authorization.py, approval.py): RBACAuthorizer (allow/deny/capability), CallableAuthorizer, IdempotencyPlugin (dedupe side-effecting tools), ToolApprovalPlugin (inline approver or block-and-surface ApprovalRequired) — all audited.
- Extensible metric registry (yaab/eval/): 7 deterministic metrics + LLMJudge + RAG groundedness (faithfulness, context_relevance) + RAGAS and DeepEval adapters (lazy-imported), unified async score() shim, third-party metrics via yaab.metrics entry point. Broader out-of-the-box metric catalog than ADK's built-in criteria.

### Context management & Model usage
- Pluggable context-window strategies in yaab/context.py: KeepAll, TruncateMessages (deterministic, no model call), and SummarizeHistory which does real ADK-style compaction (folds oldest turns into a model-generated summary, preserves system prompt + last keep_recent turns, only triggers above a token budget, falls back to truncation when no model). Token counting is pluggable via TokenCounter (default approx chars/4). Strategy is applied by the runner before every model call (yaab/runner.py lines 195-196, 404-405) in both blocking and streaming paths — verified by tests/test_context_window.py.
- Provider-agnostic model layer over LiteLLM (yaab/models/litellm_provider.py) reaching 100+ providers/thousands of models from one ModelProvider protocol — covers Gemini, Claude, Gemma, Ollama, vLLM, Bedrock, etc. via model-id strings, which is broader reach than ADK's hand-curated integration list. Includes ordered fallback chains, Retry-After-aware exponential-backoff retries, and per-call cost tracking.
- Cached-token accounting is implemented end to end: Usage.cached_input_tokens (yaab/types.py) is populated from both OpenAI (prompt_tokens_details.cached_tokens) and Anthropic (cache_read_input_tokens) shapes in litellm_provider._normalize, aggregated in Usage.add, and surfaced for cost attribution. This is a token-efficiency observability win ADK does not explicitly document.
- Reasoning/thinking trace capture: ModelResponse.reasoning (yaab/models/base.py) is populated from reasoning_content/reasoning, emitted as a MODEL_DELTA event by the runner, and modeled as a THOUGHT Part in yaab/content.py — a provider-neutral way to surface extended-thinking output across o-series/R1/Anthropic.
- Strong usage-governance primitives ADK does not bundle: UsageLimits (yaab/limits.py) caps requests, input/output/total tokens, overall and per-tool call counts, and wall-clock seconds, enforced by the runner between steps and before each tool call; CancellationToken with deadline-based timeouts; CostBudgetPlugin enforcing a hard USD ceiling (yaab/plugins/builtins.py).
- Exact-match response caching via CachingPlugin (yaab/plugins/builtins.py): caches terminal model responses keyed by the conversation and zeroes out usage/cost on a hit. This is a whole-response cache (orthogonal to, and in some repeat-query cases cheaper than, Gemini context caching).
- Resilience wrappers (yaab/models/resilient.py): async token-bucket RateLimiter and a CircuitBreaker (closed/open/half-open) that compose transparently with any ModelProvider — protects against rate-limited/failing providers, which aids cost/latency efficiency.
- OpenTelemetry GenAI-convention instrumentation per model call (yaab/models/instrumented.py) emitting gen_ai.usage.input_tokens/output_tokens/cost_usd attributes — first-class per-call token+cost observability.
- Provider-neutral multimodal Content/Part model (yaab/content.py) with TEXT/DATA(blob)/FILE/THOUGHT/TOOL parts that lowers to the OpenAI/LiteLLM multimodal array — image input works today, independent of any single vendor.

### Orchestration, Agent flows, Run lifecycle
- Durable graph runtime (yaab/graph/state.py) executes in BSP supersteps planned by the Rust core, with channel reducers (last_value/append/add) run natively, an opt-in whole-superstep Rust fold (engine='auto'|'rust'|'python'), checkpointing at every step, and HITL interrupt/resume by thread_id. This is deeper than ADK's workflow agents on the determinism/durability axis and rivals LangGraph.
- Pluggable durable checkpointers out of the box (yaab/graph/checkpoint.py): MemorySaver, SQLiteSaver, PostgresSaver/Aurora, RedisSaver, all using a Rust-accelerated framed encoder, plus full time-travel history() per thread. ADK's workflow checkpoint store options are narrower in the OSS layer.
- UsageLimits (yaab/limits.py) enforces hard caps on requests, input/output/total tokens, overall tool calls AND per-tool call counts (e.g. {'charge': 1}), and wall-clock seconds, checked between steps and before each tool call. ADK has no comparably granular per-tool-call cap primitive in the run loop.
- Swarm (yaab/multiagent.py) provides autonomous peer-to-peer handoff via auto-injected handoff_to_<peer> tools threaded through SwarmState DI - a Strands-style topology ADK's transfer model does not directly offer.
- MapAgent (yaab/multiagent.py) fans one agent across many inputs with max_concurrency bounding - a first-class map/fan-out workflow agent ADK lacks (ADK has ParallelAgent over distinct sub-agents, not map-over-inputs).
- Workflow agents share the exact run/run_sync/as_tool surface as Agent and roll up Usage across all sub-agents, so Sequential/Parallel/Loop/Map/Swarm nest arbitrarily and drop into tools/graphs/servers with whole cost accounting.
- batch.py provides bounded-concurrency offline/batch execution (batch_run/batch_map/batch_embed) with partial-failure tolerance, order preservation, and progress callbacks - a high-throughput run mode ADK does not ship as a primitive.
- Plugin callback chain (yaab/plugins/__init__.py) covers before/after run, before/after model (with short-circuit), before/after tool (with short-circuit), plus repair_tool_args and on_user_message - all async and able to observe/intervene/amend, wired into both run_stream and stream_run paths in the Runner.
- Governance runs inside the run lifecycle: registry gate (check_registered), input/output guardrail scanning per stage, and tamper-evident audit of run start/end/error - lifecycle hooks no incumbent ships inside the loop.

### YAML specs, Callbacks, Plugins, Skills, Extensibility
- Plugin system is genuinely rich and arguably ahead of ADK in breadth: yaab/plugins/builtins.py ships AuditPlugin, CostBudgetPlugin, and a real response CachingPlugin (before_model short-circuit + after_model cache-fill), and yaab/governance adds production-grade lifecycle-hook plugins ADK has no direct equivalent for — ToolApprovalPlugin (human-in-the-loop mid-run approval, yaab/governance/approval.py), ToolAuthorizationPlugin (authorizer chain, hard/soft deny, audited, yaab/governance/authorization.py), and IdempotencyPlugin (dedupes side-effecting tool calls by key, optionally across runs).
- Extra plugin hook ADK lacks: repair_tool_args (yaab/plugins/__init__.py + runner._run_tool) runs before validation to coerce malformed model tool-call args — a fool-proofing seam ADK's before_tool_callback does not cleanly provide.
- Plugin hooks are all async by default (Plugin base class) so they can do real I/O (audit sinks, remote policy checks) without the sync/async split friction.
- Extensibility backbone (yaab/extensions.py) is a unified (kind,name) component registry covering 13 kinds (model, tool, session, memory, artifact, checkpointer, guardrail, embedder, vectorstore, reranker, plugin, compliance, skill) with both in-process register/@component and lazy entry-point discovery per kind, plus runtime-checkable Protocols for every swappable concern (documented table in docs/extending.md) — broader and more uniform than ADK's per-service custom-class story.
- First-class MCP interop as both client and server (yaab/tools/mcp.py, mcp_client.py, mcp_server.py): import an MCP server's tools as native Tools, and expose an agent's tools as an MCP server — covers the dynamic-tool-import use case OpenAPI toolsets serve in ADK, via an open standard.
- Skills are governance-aware in a way ADK's are not: Skill.permissions feed the registry action-scope and Skill.card_skill() renders an A2A agent-card skills[] entry (yaab/skills.py), tying the capability bundle to compliance and discovery.
- Prompt management/versioning (yaab/prompts.py: PromptTemplate/PromptVersion/PromptRegistry with immutable hash-stamped versions and an active pointer) is a first-class auditable artifact system ADK has no equivalent for in this cluster.
- Declarative YAML config fails loud: unknown tool/skill names in agent.yaml raise ValueError (config._resolve_tools/_resolve_skills) instead of silently producing a broken agent — a fool-proofing edge over permissive loaders.

### Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth
- In-SDK RBAC tool authorization that ADK lacks entirely: yaab/governance/authorization.py ships RBACAuthorizer (allow/deny lists + per-tool required-capability checks against ctx.state['capabilities']), CallableAuthorizer, and a ToolAuthorizationPlugin that enforces a chain of authorizers in the before_tool hook, audits every denial, and supports hard (raise PolicyViolation) vs soft (return error string to the model) modes. ADK defers RBAC to Google Cloud IAM with no in-SDK equivalent.
- IdempotencyPlugin (yaab/governance/authorization.py) dedupes side-effecting tool calls by an idempotency key (hash of tool+args or a custom key_fn), per-run or shared across runs, returning the cached result instead of re-executing and refusing to cache error results. ADK has no built-in tool idempotency.
- Bidirectional MCP sampling on BOTH sides: MCPClient.sampler_from_model / MCPServer(request_sampling=...) let a server delegate completions back to the client's model (yaab/tools/mcp_client.py, mcp_server.py). This server->client sampling round-trip is a depth feature beyond plain MCP tool calling.
- MCP server side is first-class: MCPServer exposes YAAB tools, resources (static text or sync/async loader), and prompt templates over JSON-RPC, with MCPServer.from_agent(agent) to publish an agent's whole toolset; round-trip tested against the in-process MCPClient (tests/test_interop_depth.py).
- Governance-enriched A2A agent card: AgentCard.to_a2a_card (yaab/governance/registry.py) embeds an x-yaab-governance block (risk_tier, EU AI Act category, approval status, decision authority, lifecycle state) into /.well-known/agent.json, and serve.py injects securitySchemes from the active auth scheme - richer than a vanilla ADK card.
- A2A long-running task polling client-side: RemoteAgent.poll_task polls /a2a/tasks/{id} until a terminal state with configurable interval/timeout/terminal-state set (yaab/a2a/client.py), plus a token_provider hook for fresh per-request OAuth bearer tokens.
- Pluggable serving auth as a clean Protocol (yaab/auth.py): NoAuth, BearerTokenAuth, APIKeyAuth, and OAuth2 (delegated validator) all map request headers to an identity that flows into the run context and audit log, and each self-describes for the agent card's securitySchemes.

### Deployment, Runtime, Observability, Built-in tools
- Provider-neutral serving: `fastapi_server_app(agent)` (yaab/serve.py) exposes a single ASGI app with native (/run, /run/stream, /chat/stream), A2A (/a2a/tasks + polling), discovery (/.well-known/agent.json) and /health endpoints — works on Cloud Run, Fargate, Lambda, GKE, any ASGI host. ADK's HTTP serving is Gemini/Vertex-centric; YAAB's is vendor-neutral with pluggable auth (Bearer/APIKey/OAuth2 via yaab/auth.py) advertised in the agent card's securitySchemes.
- Pluggable auth baked into serving (yaab/auth.py, surfaced in serve.py via _identify) with identity flowing into the run context AND the tamper-evident audit log — stronger first-class auth-to-audit wiring than ADK's local servers.
- Runtime-controllable tracing in yaab/observability/__init__.py: global on/off switch (set_tracing_enabled / YAAB_DISABLE_TRACING=1) and a PII redactor (set_trace_redactor) that scrubs every span attribute on the way in and post-hoc via _RedactingSpan — directly addresses ecosystem asks (ADK #2792, Strands #1292) that ADK does not natively cover.
- Multiple observability sinks beyond OTel: Langfuse, Logfire, OTelSpanSink, and CallbackSink (yaab/observability/sinks.py) all fed by the hash-chained governance audit log — Langfuse/Logfire are integrations ADK 2.0 does not ship.
- DockerSandbox (yaab/tools/sandbox.py) is a genuinely hardened local container executor: --network none, --read-only, --cap-drop ALL, --pids-limit 64, memory/CPU/time caps — comparable to ADK's ContainerCodeExecutor and configurable as a pluggable Sandbox protocol with set_default_sandbox.
- Provider-neutral multimodal Content/Part model (yaab/content.py): TEXT/DATA(blob)/FILE/THOUGHT/TOOL_CALL/TOOL_RESULT parts that round-trip through sessions, checkpoints and SSE and render to the OpenAI multimodal content array — so image input is not tied to one vendor (ADK's Content is google.genai-specific).
- Human-in-the-loop tool approval on the fast path (yaab/governance/approval.py ToolApprovalPlugin) with inline-approver and block (ApprovalRequired) modes, plus graph-level interrupt()/resume (yaab/graph/state.py) and durable checkpointers (Memory/SQLite/Postgres/Redis Saver with history()) — covers the pause-for-human case across both runtime paradigms.
- A2A remote agent client (yaab/a2a/client.py RemoteAgent) supports long-running task polling (get_task/poll_task with terminal-state detection and timeout) and token_provider for per-request OAuth bearer refresh — a remote-agent runtime feature ADK addresses differently.

---

## HIGH — build these next

### YAML agent spec coverage (sub_agents, callbacks, guardrails, model settings)
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **partial** · effort: **medium***

**ADK provides:** agent.yaml is a full declarative definition: model, instructions, tools, sub_agents, and callbacks referenced by name — you can build real multi-agent, callback-guarded agents with zero code.

**The gap (verified against code):** yaab/config.py agent_from_dict/agent_from_yaml only honor a fixed _AGENT_KEYS set = {model, instructions, registry_id, max_steps, output_retries, tool_choice, instrument} plus tools and skills lists. It silently drops/ignores everything else: there is NO support for sub_agents (despite SequentialAgent/ParallelAgent/LoopAgent/Swarm existing in yaab/multiagent.py), no callbacks/plugins key, no guardrails, no model_settings, no deps_type, no parallel_tools, no context_strategy. output_type is hard-coded: line 79-80 maps any output_type name to str ('only "str" is supported declaratively'), so typed outputs are impossible from YAML. So YAAB's declarative path can only express a single flat tool-using agent — far short of ADK's code-free multi-agent + callback story.

**Verifier note:** Confirmed real. yaab/config.py:91-99 _AGENT_KEYS = {model, instructions, registry_id, max_steps, output_retries, tool_choice, instrument}; agent_from_dict (lines 68-88) pops only name/tools/skills/output_type and forwards the rest filtered by _AGENT_KEYS, silently dropping anything else. output_type is hard-pinned to str at line 80 (`output_type = str if output_type_name in ('str', None) else str` — both branches return str), so typed outputs are impossible from YAML. The Agent class (yaab/agent.py:29-51) DOES accept guardrails, model_settings, deps_type, parallel_tools, max_parallel_tools, context_strategy, output_type in code, but none of these are in _AGENT_KEYS so the declarative path cannot reach them. There is no sub_agents support despite SequentialAgent/ParallelAgent/LoopAgent/Swarm/MapAgent in yaab/multiagent.py, and no callbacks/plugins/guardrails keys. The declarative path can only express a single flat tool+skill agent. Severity high is fair.

### OpenAPI / Swagger toolset auto-generation
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **missing** · effort: **medium***

**ADK provides:** ADK auto-generates a callable toolset from an OpenAPI/Swagger spec (one tool per operation, params/auth wired) so any REST API becomes agent tools without hand-written wrappers.

**The gap (verified against code):** Grep across the repo for openapi/swagger/from_spec/from_openapi finds only doc/RAG mentions, none in yaab/tools. yaab/tools/__init__.py exports only FunctionTool, AgentTool, MCPTool, mcp_toolset. There is no spec->tools generator. MCP partially covers dynamic tool import but does not consume OpenAPI specs, so wrapping arbitrary REST APIs still requires writing each FunctionTool by hand.

**Verifier note:** Confirmed real. Repo-wide grep for openapi/swagger/from_spec/from_openapi finds nothing in yaab/tools (and nothing anywhere in the package). yaab/tools/__init__.py exports only FunctionTool, AgentTool, MCPTool, mcp_toolset, tool, coerce_tools, Tool. yaab/tools/mcp.py wraps already-discovered MCP tool descriptors (name/description/inputSchema) — it does not parse an OpenAPI spec. Wrapping a REST API still requires hand-written FunctionTools. Severity high is fair for a no-code-tool-reach feature.

### Dynamic model routing per request
*Context management & Model usage · status: **missing** · effort: **medium***

**ADK provides:** ADK routes per request — send simple/cheap queries to a small model and hard ones to a flagship — for major cost reduction.

**The gap (verified against code):** No router/dispatch model exists. yaab/models/ has LiteLLMModel (single primary + ordered FALLBACKS that only trigger on failure, not on query complexity/cost), TestModel, FunctionModel, InstrumentedModel, ResilientModel — none select a model based on the request. Grep for Rout/Dispatch/Selector/Tiered/Cascade/complexity model classes returns nothing; multiagent.py 'routing' is agent handoff/delegation, not per-request model selection. An agent is bound to one model at construction (yaab/agent.py). Adopters wanting cost-aware tiering must hand-roll it.

**Verifier note:** Confirmed real. The only ModelProvider implementations are LiteLLMModel (yaab/models/litellm_provider.py), TestModel/FunctionModel (yaab/models/test_model.py), InstrumentedModel (instrumented.py), ResilientModel (resilient.py). LiteLLMModel.complete() (lines 102-145) tries self.model then self.fallbacks in order, triggered ONLY on exception — failure-driven, not complexity/cost-driven. No Router/Dispatch/Selector/Tiered/Cascade/complexity model class, no route()/select_model()/choose_model()/pick_model() function (grep returned none). yaab/optimize/optimizer.py is DSPy-style build-time prompt compilation, not a runtime per-request router. multiagent.py 'routing' (samples handoff) is agent delegation. An agent binds one model via resolve_model at construction (agent.py model property). Adopters must hand-roll tiering.

### Bidirectional / live audio runtime (run_live)
*Deployment, Runtime, Observability, Built-in tools · status: **missing** · effort: **large***

**ADK provides:** ADK has run_live() bidi streaming runtime with Gemini Live API voice agents, audio transcription in/out, and video streaming.

**The gap (verified against code):** No run_live, no bidi audio, no voice agent, no transcription anywhere. Grep for run_live/bidi/transcription returns nothing in the runtime. agent.stream is text-token-only (single turn, no tool loop); agent.stream_events drives the tool loop but emits text/tool events only. The Content model defines DATA/FILE parts (could carry audio bytes) but there is no live audio session, no Live API integration, no input/output transcription pipeline.

**Verifier note:** Confirmed real. No run_live/bidi audio/voice/transcription anywhere in the runtime. agent.stream (agent.py:184) is documented 'single turn, no tool loop' text deltas; agent.stream_events (agent.py:227) drives the tool loop but yields only TEXT_DELTA/TOOL_CALL/TOOL_RESULT/FINAL_OUTPUT/RUN_END events. The two grep hits for 'bidi'/'run_live' are unrelated: mcp_client.py:82 is a comment about bidirectional MCP transports, and live_e2e.py is a live-API HTTP smoke test, not a live audio runtime. content.py defines DATA/FILE parts but there is no Live API integration or in/out transcription pipeline.

### Visual dev UI (adk web equivalent)
*Deployment, Runtime, Observability, Built-in tools · status: **weaker** · effort: **large***

**ADK provides:** ADK's `adk web` is a full visual dev UI: chat, live event/trace/state inspector, an eval tab, a session browser, and audio/video streaming UI.

**The gap (verified against code):** yaab/web.py serves a single self-contained HTML page (_PAGE) that is a bare chat box streaming /chat/stream tokens — no event/trace inspection, no state viewer, no session browser, no eval tab, no audio/video. It is explicitly described in docs/COMPARISON.md gap list as just a 'zero-build playground'. Far below adk web's depth.

**Verifier note:** Confirmed real. yaab/web.py serves a single self-contained _PAGE HTML that is a bare chat box streaming /chat/stream SSE tokens — no event/trace inspector, no state viewer, no session browser, no eval tab, no audio/video (grep of agui.py for trace/inspector/state-view/audio/video found nothing; AG-UI is a streaming protocol layer, not a visual dev UI). web.py's own docstring calls it a 'zero-build local dev playground'. Minor correction to the claim's sourcing: the 'zero-build playground' phrasing lives in web.py and docs/platform.md, not in COMPARISON.md's gap list — but the substance (far below adk web's depth) is fully confirmed by the code itself.

### Managed / one-command cloud deployment
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **large***

**ADK provides:** ADK deploys to Vertex AI Agent Engine (fully managed, auto-scaling, managed sessions/memory) and Cloud Run/GKE via a single CLI command (`adk deploy cloud_run`), plus an agents-cli.

**The gap (verified against code):** YAAB ships a generic Dockerfile (compiles Rust core, runs `yaab serve $YAAB_AGENT`) and docs/DEPLOYMENT.md gives manual recipes for Cloud Run/Fargate/K8s/Lambda. There is NO `yaab deploy` command — yaab/cli.py only has info/init/registry/compliance/serve/web. No managed control plane, no auto-scaling integration, no one-command push. Deployment is bring-your-own-container with hand-written steps.

**Verifier note:** Confirmed real. yaab/cli.py defines only info/init/registry/compliance/serve/web subparsers — there is no `deploy` command (grep for 'yaab deploy'/'_deploy'/'AgentEngine'/'agent_engine' returns nothing across the repo). The Dockerfile is a generic build-Rust-then-`yaab serve` image, and docs/DEPLOYMENT.md gives only hand-written Cloud Run/Fargate/K8s/Lambda recipes ('deploy the image; set the port to 8000'). No managed control plane, auto-scaling integration, or one-command push. docs/COMPARISON.md item 6 and the net assessment explicitly concede ADK has managed deployment and YAAB ships 'docs + a Dockerfile rather than turnkey templates'. Severity 'high' is reasonable for an enterprise/ops-facing gap.

### Curated managed built-in tools (google_search, VertexAiSearch, BigQuery, Spanner, url_context, computer_use)
*Deployment, Runtime, Observability, Built-in tools · status: **weaker** · effort: **large***

**ADK provides:** ADK ships google_search (grounding), VertexAiSearchTool, BigQuery toolset, Spanner toolset, url_context, and computer_use (browser control).

**The gap (verified against code):** YAAB's built-in toolset (yaab/tools/builtin/) is calculator, current_time, http_get, web_search, python_exec only. web_search is just a provider shim — set_search_provider must be wired to Tavily/Brave/SerpAPI; with none configured it returns a config-hint string (search.py). There is NO grounding/google_search, NO managed-data-source tools (BigQuery/Spanner/VertexAiSearch), NO url_context tool, and NO computer_use/browser-control tool. Grep confirms none exist. Much thinner built-in tool surface than ADK.

**Verifier note:** Confirmed real. yaab/tools/builtin/__init__.py exports exactly calculator, current_time, http_get, web_search, python_exec. search.py's web_search is a provider shim: with no provider set via set_search_provider it returns a 'no web search provider configured' hint string. Grep for google_search/grounding/VertexAiSearch/BigQuery/Spanner/url_context/computer_use/browser_use across the whole repo returns zero matches. No grounding, no managed-data-source toolsets, no url_context, no computer-use/browser-control tool. Much thinner built-in surface than ADK; 'high' severity is defensible given how central built-in tools are to ADK's value prop.

### EvalSet / EvalCase file format
*Evals, Guardrails, Audit, Governance · status: **missing** · effort: **medium***

**ADK provides:** ADK persists evals as versioned .evalset.json files (EvalSet containing EvalCases with multi-turn conversation + expected tool-use), editable in the web UI and replayable.

**The gap (verified against code):** yaab/governance/eval.py defines Case/Dataset/Experiment as pure in-memory pydantic objects. There is no file schema, no load/save, no from_json/from_jsonl/from_file — I grepped the whole tree for evalset/EvalSet/EvalCase/from_json/load_dataset and found zero matches. Datasets must be hand-built in Python each run; there is no portable, versioned, UI-editable eval artifact.

**Verifier note:** Real gap. yaab/governance/eval.py defines Case/Dataset/Experiment as pure in-memory pydantic models with no load/save/from_json/from_jsonl/from_file. Repo-wide grep for evalset/EvalSet/EvalCase/from_json/load_dataset returns zero eval-related matches (only unrelated session/graph/artifact 'savers'). No portable, versioned, UI-editable eval artifact exists.

### 'adk eval' CLI command
*Evals, Guardrails, Audit, Governance · status: **missing** · effort: **medium***

**ADK provides:** ADK ships an 'adk eval' CLI that runs an agent against an eval set file and reports pass/fail per criteria.

**The gap (verified against code):** yaab/cli.py only registers subcommands info/init/registry list/compliance report/serve/web (confirmed by reading cli.py and its argparse setup). There is no 'yaab eval' command. Running an Experiment requires writing async Python and calling exp.run() yourself. No CLI path to run a dataset against an agent and get scored output.

**Verifier note:** Real gap. yaab/cli.py registers only info, init, registry list, compliance report, serve, web (read directly). No 'yaab eval' subcommand; running an Experiment requires hand-written async Python calling exp.run().

### Tool-trajectory evaluation metric
*Evals, Guardrails, Audit, Governance · status: **missing** · effort: **medium***

**ADK provides:** ADK's tool_trajectory_avg_score compares the agent's actual sequence of tool calls (names + args) against an expected trajectory, a core ADK eval criterion.

**The gap (verified against code):** The Experiment.run() loop (eval.py lines 213-234) only feeds case.inputs to the task and captures the single final output; evaluators receive (case, output) and score only the final string. No evaluator inspects the tool-call sequence. The AuditPlugin records TOOL_CALL events (plugins/builtins.py) but the eval framework never consumes them, and tool args are not even recorded (only the tool name). There is no expected-trajectory field on Case and no trajectory comparator metric.

**Verifier note:** Real gap. Experiment.run() (eval.py lines 213-234) feeds case.inputs to the task and captures only the single final output; every evaluator receives (case, output) and scores the final value. No expected-trajectory field on Case, no tool-call-sequence comparator anywhere (grep for trajectory/expected_tool: zero matches).

### User simulation for multi-turn eval
*Evals, Guardrails, Audit, Governance · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 introduces simulated users that autonomously drive multi-turn conversations against an agent to evaluate end-to-end dialogue behavior.

**The gap (verified against code):** No simulated-user capability exists. Grep for simulat/UserSimulator across the repo returns only unrelated comments ('simulate an external cancel', 'simulated restart') in tests/samples. Experiment.run() is strictly single-turn: task(case.inputs) -> output. There is no driver that role-plays a user across turns, no goal-conditioned conversation generator, and no multi-turn eval harness.

**Verifier note:** Real gap. Grep for simulat/UserSimulator/role.?play returns only unrelated test/sample comments ('simulate an external cancel', 'simulated restart'). Experiment.run() is strictly single-turn task(case.inputs)->output. No user-simulating driver or multi-turn eval harness exists.

### Managed long-term memory with automatic memory extraction
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **weaker** · effort: **medium***

**ADK provides:** VertexAiMemoryBankService: managed long-term memory that automatically extracts durable 'memories' (facts, preferences) from finished sessions via an LLM, deduplicates/consolidates them, and serves them back — not just raw turn storage.

**The gap (verified against code):** yaab/memory/manager.py MemoryManager.add_session_to_memory() only copies raw user/assistant message text verbatim into the vector store (one record per message), scoped by app_name/user_id. There is no extraction, summarization, salience filtering, or consolidation — grep for extract/consolidate/summar/salient/distill across yaab/ returns no matches. So recall surfaces raw chat lines, not distilled memories, and the store grows unbounded with duplicate/low-value content. This is the single biggest memory-quality gap vs ADK's MemoryBank.

**Verifier note:** Confirmed real. yaab/memory/manager.py MemoryManager.add_session_to_memory() (lines 66-86) iterates session.messages and stores each user/assistant message's raw .content verbatim via self.add(), one MemoryRecord per message, with metadata {session_id, role}. There is zero LLM-based extraction, summarization, salience filtering, deduplication, or consolidation. Grep for extract/consolidat/summar/salien/distill across yaab/memory/ returns no matches. The only concrete MemoryService (InMemoryVectorMemory in memory/__init__.py) is append-only. Severity 'high' is fair: this is the central memory-quality differentiator vs ADK's VertexAiMemoryBankService.

### RAG-backed managed memory service
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **partial** · effort: **medium***

**ADK provides:** VertexAiRagMemoryService — a first-class MemoryService whose recall is backed by a managed RAG corpus.

**The gap (verified against code):** YAAB has the building blocks (InMemoryVectorMemory in memory/__init__.py; KnowledgeBase in rag/knowledge.py) but ships no adapter that exposes a KnowledgeBase/VectorStore *as* a MemoryService. The only concrete MemoryService is InMemoryVectorMemory (process-local, lost on restart). docs/storage-backends.md says you can 'back it with any vector store via a KnowledgeBase' but no such class exists in the code — it is a do-it-yourself instruction, not a shipped durable MemoryService. So durable, cross-restart long-term memory requires the user to write a MemoryService themselves.

**Verifier note:** Confirmed real. The only class implementing the MemoryService protocol is InMemoryVectorMemory (memory/__init__.py:118, process-local list self._records, lost on restart). No PgVectorMemory, no KnowledgeBase-backed or VectorStore-backed MemoryService adapter exists anywhere (grep 'class .*Memory' yields only MemoryRecord, MemoryService protocol, InMemoryVectorMemory). docs/storage-backends.md:48-54 explicitly frames durable memory as DIY: 'for durable memory, back it with any vector store below via a KnowledgeBase, or implement MemoryService against your store of choice' — an instruction, not a shipped class. Note the extensions.py registry reserves a 'memory' entry-point group (line 39) but no concrete durable memory backend is registered via register('memory', ...). Claim accurate.

### Context caching for long prompts
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **missing** · effort: **large***

**ADK provides:** Gemini context-caching integration: register a large reusable prompt/context once and reference it cheaply across calls.

**The gap (verified against code):** No context/prompt caching anywhere in the cluster or model layer — grep for cache_control/cachePoint/prompt_cache/ephemeral/context.cach finds nothing; types.py only *reports* provider cached-prompt token counts (a usage field at line ~81), it does not *create or manage* a cache. There is no API to register reusable context, no Gemini cachedContent integration, and no Anthropic cache_control breakpoint emission. RAG-heavy/long-system-prompt workloads pay full token cost every call.

**Verifier note:** Confirmed real. Grep for cache_control/cachePoint/prompt_cache/ephemeral/cachedContent/cached_content/context.cach across yaab/ returns nothing relevant (only 'ephemeral' as a state-prefix doc note in state.py/sessions/manager.py, unrelated to prompt caching). types.py:81-83 only *reports* a provider usage field cached_input_tokens ('Prompt tokens served from the provider's prompt cache') and accumulates it — it never *creates or manages* a cache. There is no API to register reusable context, no Gemini cachedContent integration, and no Anthropic cache_control breakpoint emission in the model layer (models/). Severity 'high' is justified for RAG-heavy/long-system-prompt cost, though it is a cost-optimization rather than a correctness gap.

### LLM-driven delegation by sub-agent description (transfer_to_agent)
*Orchestration, Agent flows, Run lifecycle · status: **partial** · effort: **medium***

**ADK provides:** ADK auto-injects a transfer_to_agent function and lets the LLM hand control to a registered sub_agent purely by matching the task to each sub-agent's description; control transfers (the sub-agent takes over the conversation), not just a tool-call-and-return.

**The gap (verified against code):** YAAB has agent-as-tool (yaab/tools/agent_tool.py AgentTool) and Swarm handoff tools (yaab/multiagent.py _make_handoff_tool), but there is no framework-managed sub_agents registry on Agent and no auto-generated transfer_to_agent that lets the LLM pick a sub-agent by description and transfer the run. agent.py has no sub_agents field; delegation is either (a) the developer manually wiring .as_tool() so the parent only gets the sub-agent's output back, or (b) Swarm, which requires the developer to pre-list peers and pass SwarmState deps. The ADK pattern of 'declare sub_agents, framework injects transfer, LLM routes by description, control moves' is absent.

**Verifier note:** Verified in yaab/agent.py: Agent.__init__ (lines 29-99) has no sub_agents parameter or field. Grep for 'sub_agents'/'transfer_to_agent' across yaab/ returns zero code matches (only docstrings in multiagent.py describing workflow composition). Delegation is via AgentTool (.as_tool, tools/agent_tool.py - fire-and-return) or Swarm handoff tools (multiagent.py _make_handoff_tool, requires SwarmState deps and pre-listed peers). The ADK 'declare sub_agents, framework injects transfer, LLM routes by description, control moves' pattern is genuinely absent.

### Per-node retry policies in the workflow runtime
*Orchestration, Agent flows, Run lifecycle · status: **missing** · effort: **medium***

**ADK provides:** ADK 2.0 workflow runtime supports retry policies per node (configurable attempts/backoff on a graph node).

**The gap (verified against code):** yaab/graph/state.py CompiledGraph.ainvoke has no retry handling whatsoever - a node raising any exception other than Interrupt propagates and aborts ainvoke (no try/except around fn execution except for Interrupt). add_node takes only (name, fn); there is no RetryPolicy, max_retries, or backoff parameter anywhere in graph/. Grep for retry/RetryPolicy/max_retries/attempts in yaab/graph returned zero matches. Node-level fault tolerance must be hand-rolled inside each node function.

**Verifier note:** Verified in yaab/graph/state.py: StateGraph.add_node (line 100) takes only (name, fn). CompiledGraph.ainvoke (lines 251-269) only try/excepts Interrupt; any other exception from a node propagates and aborts the invocation. Grep for retry/RetryPolicy/max_retries/backoff/join across yaab/graph returns only unrelated matches ('Rust barrier' comments). No per-node retry/backoff anywhere.

### Resumable / fault-tolerant fast-path (model-driven) runs
*Orchestration, Agent flows, Run lifecycle · status: **partial** · effort: **large***

**ADK provides:** ADK 2.0 RESUME agents: fault-tolerant resumption of interrupted model-driven runs via a resumability config, so a crashed/interrupted agent run can be picked back up.

**The gap (verified against code):** Resumability exists ONLY for the durable graph (yaab/graph/state.py CompiledGraph.ainvoke resumes from checkpointer by thread_id, parking interrupted/not-yet-run nodes). The model-driven fast path (Runner.run_stream in yaab/runner.py) is NOT resumable: it builds messages fresh each call, has no checkpoint of loop progress (step index, partial messages, tool results), and on exception emits a terminal ERROR event and returns - there is no way to resume a fast-path run from where it died. Session history replay (_build_messages) re-feeds prior turns but does not restore mid-loop state (tool calls in flight, retry budget). So an interrupted multi-step tool loop restarts from scratch.

**Verifier note:** Verified: resumability exists only for the durable graph (yaab/graph/state.py ainvoke lines 227-235 resume from checkpointer by thread_id, parking interrupted/not-yet-run nodes). Runner.run_stream (yaab/runner.py lines 110+) builds messages fresh via _build_messages each call, holds no checkpoint of loop progress (step index, in-flight tool calls, retry budget), and on exception emits a terminal ERROR event and returns (lines 322-330). No resume of a fast-path run. Confirmed partial.

### Cancel in-flight runs via API
*Orchestration, Agent flows, Run lifecycle · status: **partial** · effort: **medium***

**ADK provides:** ADK 2.0 lets you terminate in-flight runs through the run API (cancel a running agent remotely).

**The gap (verified against code):** The primitive exists: limits.py CancellationToken (cooperative cancel + deadline) is honored by the Runner between steps and before each tool call, and timeout wires an auto-deadline. BUT there is no API surface to cancel a run you did not start in-process: yaab/serve.py has no /run/{id}/cancel endpoint, no run registry mapping run_id -> CancellationToken, and the run_id is generated per-call inside RunContext with no external handle. So an operator/HTTP caller cannot terminate an in-flight server run; only the code holding the token can. ADK's remote-terminate-by-run-id is not reachable over the served API.

**Verifier note:** Verified: yaab/limits.py CancellationToken (lines 108-141) is honored by the Runner between steps and before each tool call, and timeout wires an auto-deadline. But yaab/serve.py has no cancel endpoint and no run_id->token registry (grep for cancel/registry shows only result.run_id echoed in /run response and event payloads, lines 79/184). run_id is generated inside RunContext per-call with no external handle. An HTTP caller cannot terminate an in-flight server run. Confirmed partial.

### Tool-level OAuth2 / interactive consent flow
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **missing** · effort: **large***

**ADK provides:** ADK supports full OAuth2 FOR TOOLS: auth_scheme + auth_credential attached to a tool, interactive OAuth consent driven mid-conversation, token exchange/refresh, OpenID Connect, service-account credentials, and API-key injection - the agent can walk a user through OAuth consent to call a protected tool.

**The gap (verified against code):** YAAB's OAuth2 (yaab/auth.py) is purely SERVER-SIDE request authentication (validate an inbound bearer token -> identity for /run and /a2a). The Tool protocol (yaab/tools/base.py) and FunctionTool have NO auth_scheme/auth_credential fields, no notion of a tool requiring user authorization, and there is no mid-run consent/redirect handshake, no token exchange, no OIDC, no service-account or API-key injection into tool calls. The only tool-auth seam is RBAC allow/deny (governance/authorization.py), which is access control, not credential acquisition. This is the single biggest gap in the cluster for any adopter building tools that call OAuth-protected SaaS APIs.

**Verifier note:** Confirmed. yaab/tools/base.py Tool protocol and FunctionTool have no auth_scheme/auth_credential fields; grep across yaab/tools for oauth/consent/auth_scheme/token_exchange/refresh_token/api_key returns nothing. yaab/governance/authorization.py is allow/deny RBAC + idempotency only (access control, not credential acquisition). There is no mid-run OAuth consent/redirect handshake, token exchange/refresh, OIDC, service-account, or API-key injection into tool calls. The gap is real. I'm revising severity from critical to high: it is the largest gap in this cluster and blocks turnkey OAuth-protected-SaaS tools, but YAAB tools are arbitrary async Python and an adopter can acquire/refresh tokens inside the tool body or via deps (RunContext deps + token_provider patterns exist, e.g. RemoteAgent.token_provider), so it is a missing first-class convenience/protocol-driven flow rather than an absolute capability wall — high, not critical.

---

## MEDIUM — parity items

### Callbacks referenced by name in declarative config
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **missing** · effort: **medium***

**ADK provides:** Callbacks (before/after agent, before/after model, before/after tool) can be referenced by name inside agent.yaml so guardrails/logging/caching are wired declaratively without code.

**The gap (verified against code):** Grep for callbacks in config.py returns nothing; _AGENT_KEYS has no callbacks/plugins entry. Plugins live only on the Runner (programmatic Runner(plugins=[...])) and cannot be attached to an agent built from YAML, nor referenced by name. There is no name->callback resolver. A YAML-defined agent therefore cannot carry any guardrail/logging/caching behavior declaratively.

**Verifier note:** Confirmed real. yaab/config.py has no 'callbacks' or 'plugins' key and no name->callback resolver; _AGENT_KEYS (lines 91-99) omits both. Plugins are only constructed programmatically and attached to a Runner (Runner(plugins=[...]) in runner.py:46-66; add_plugin at 68-70). A YAML-built Agent carries no runner reference for plugins (Agent only lazily builds a bare Runner() in agent.py:129-134), so a YAML-defined agent cannot declaratively carry guardrail/logging/caching behavior. Severity medium is fair.

### Per-agent before_agent / after_agent callbacks
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **weaker** · effort: **medium***

**ADK provides:** ADK exposes before_agent_callback and after_agent_callback bound to a specific agent (not the whole runner), letting one agent in a multi-agent system have its own entry/exit interception and short-circuit.

**The gap (verified against code):** yaab/plugins/__init__.py Plugin has before_run/after_run, but these fire on the Runner across EVERY agent it drives (runner.run_stream loops self.plugins and passes agent.name as a string). There is no agent-scoped callback: you cannot attach an interceptor to just one sub-agent, and before_run cannot short-circuit the whole run (its return is ignored — only before_model/before_tool returns short-circuit). For composed/multi-agent graphs this is a real expressiveness gap vs ADK's per-agent hooks.

**Verifier note:** Confirmed real. yaab/plugins/__init__.py Plugin defines before_run/after_run (lines 31-33) but they are invoked by the Runner across every agent it drives, with agent passed as a bare string (runner.py:155-156, 171-172, 307-308 and the stream path 375-376, 469-470). There is no agent-scoped hook attach point on the Agent. before_run's return value is ignored (no short-circuit) — only before_model (runner.py:580-583) and before_tool (628-631) returns short-circuit, plus after_model/after_tool amend. So you cannot intercept just one sub-agent's entry/exit nor short-circuit a whole run from a before-agent hook. Severity medium is fair (real expressiveness gap, but plugins cover much of the cross-cutting need).

### LangChain / CrewAI tool wrappers
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **missing** · effort: **small***

**ADK provides:** ADK ships adapters to wrap LangChain and CrewAI tools so the large existing tool ecosystems drop into an ADK agent.

**The gap (verified against code):** Grep for langchain/crewai in the package returns only docs/ROADMAP and RESEARCH_FEATURE_DEMAND references — no adapter code. coerce_tools (yaab/tools/base.py) accepts only callables or objects already implementing the Tool protocol (schema()+execute()); a LangChain BaseTool or CrewAI tool would not satisfy it without a user-written shim. This blocks easy reuse of those ecosystems' hundreds of prebuilt tools.

**Verifier note:** Confirmed real. The only langchain/crewai matches in yaab/ are prose/issue references: governance/authorization.py (CrewAI issue numbers in a docstring), governance/monitor.py (docstring), rag/knowledge.py and rag/types.py (LangChain/LlamaIndex naming nods). No adapter code. coerce_tools (yaab/tools/base.py:136-146) accepts only FunctionTool, objects with schema()+execute(), or plain callables — a LangChain BaseTool / CrewAI tool would raise ToolError without a user shim. Severity medium is fair.

### Context filtering (drop irrelevant events)
*Context management & Model usage · status: **weaker** · effort: **medium***

**ADK provides:** ADK 2.0 'every token earns its place': automatically FILTERS the event/message history to drop irrelevant items before a model call, not just truncate by recency.

**The gap (verified against code):** yaab/context.py offers only recency truncation (TruncateMessages keeps last N) and summarization (SummarizeHistory). There is no relevance-based filtering — no scoring/dropping of individual irrelevant events while keeping relevant older ones. The ContextStrategy protocol could host one, but no such strategy ships and the runner only ever calls strategy.apply on the flat message list.

**Verifier note:** Confirmed real. yaab/context.py ships exactly three ContextStrategy implementations: KeepAll (no-op), TruncateMessages (keeps system + last N by recency, lines 52-62), SummarizeHistory (folds oldest into a model-generated summary above a token budget, lines 65-110). Neither scores/drops individual irrelevant events while retaining relevant older ones — both are recency-based. The ContextStrategy protocol (line 34) could host a relevance filter but none ships. Severity medium is appropriate (truncation+summarization is a reasonable partial substitute).

### Lazy-loading of artifacts / large tool outputs into context
*Context management & Model usage · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 lazy-loads artifacts and tool outputs: large blobs/results are kept by reference and only materialized into the prompt when actually referenced, keeping tokens down.

**The gap (verified against code):** No reference-and-lazy-load path exists. ArtifactManager (yaab/artifacts/manager.py) load() returns full bytes eagerly; nothing inserts an artifact/tool-output PLACEHOLDER into the message list that is hydrated on demand. Tool results are appended verbatim to messages in the runner. Content.Part has FILE (uri) and DATA (inline base64) kinds but to_provider() inlines them rather than deferring; there is no token-budget-aware 'load only if the model references it' mechanism. Grep for lazy/placeholder/by-reference in the context/artifact/tool paths finds nothing relevant. Large tool outputs therefore consume context every turn.

**Verifier note:** Gap is real but severity 'high' is overstated; medium is fairer. Evidence: ArtifactManager.load() (yaab/artifacts/manager.py line 45-62) returns full bytes eagerly via service.get(); ArtifactService.get() returns bytes|None (artifacts/__init__.py line 27/47). The runner appends tool results verbatim as full text into messages (yaab/runner.py lines 239-261, 439-444: content=_to_text(result_value)) — no placeholder Part inserted, no token-budget-aware 'load only if referenced' hydration. Part.to_provider() (content.py lines 83-92) inlines FILE uri and DATA base64 rather than deferring. Grep for lazy/placeholder/by-reference/hydrate/materialize in the context/artifact/tool paths finds nothing relevant. Reason for downgrade to medium: this is an optimization, not a correctness gap; truncation/summarization strategies (context.py) plus the artifact-by-name store partially mitigate runaway context, so it is not as severe as a hard-missing capability.

### Per-component token usage tracking
*Context management & Model usage · status: **weaker** · effort: **medium***

**ADK provides:** ADK 2.0 tracks token usage per component (per sub-agent / per tool / per pipeline stage) to attribute and optimize cost.

**The gap (verified against code):** Usage (yaab/types.py) is aggregated only at the run level via Usage.add into ctx.usage; RunResult exposes one Usage. There is no per-agent or per-tool token/cost breakdown surfaced. In multi-agent runs (yaab/multiagent.py) all sub-agent calls accumulate into one counter with no attribution. governance/monitor.py has no per-component token accounting (grep confirms). The InstrumentedModel span carries per-CALL tokens to OTel, but YAAB provides no in-process per-component rollup.

**Verifier note:** Confirmed real. Usage (yaab/types.py lines 74-92) is a single flat counter; Usage.add accumulates everything into ctx.usage; RunContext holds one usage (line 117); RunResult exposes one Usage (line 157). _call_model does ctx.usage.add(response.usage) per call (runner.py line 591) with no per-agent/per-tool key. governance/monitor.py 'components' (lines 75,121-131) is a drift/trust SCORE blend (accuracy/safety weights), not token accounting — confirms claim. InstrumentedModel emits per-CALL token attributes to OTel spans but YAAB provides no in-process per-component rollup. Severity medium is fair.

### Gemini Live API: bidirectional audio/video streaming (voice agents)
*Context management & Model usage · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 integrates the Gemini Live API for bidirectional streaming with AUDIO and VIDEO input/output, enabling real-time voice/video agents.

**The gap (verified against code):** No bidirectional/duplex/realtime path anywhere. ModelProvider.stream is one-directional text/tool deltas (yaab/models/base.py); there is no audio/video input ingestion or audio output. Grep for audio/video/bidirectional/duplex/realtime/Live API/WebSocket in yaab/ finds only a content.py comment ('image/audio/...') and an MCP transport note — no Live session, no PCM/audio frames, no WebSocket serving. docs/ROADMAP.md explicitly lists 'Realtime / voice API' as 'out of scope near-term'. Voice agents are not buildable on YAAB today.

**Verifier note:** Confirmed real. ModelProvider.stream (yaab/models/base.py lines 67-75) is one-directional StreamChunk text/tool deltas only. Grep for audio/video/bidirectional/duplex/realtime/websocket/Live API/PCM across yaab/ finds only a content.py comment ('image/audio/...', line 28), an mcp_client.py 'bidirectional transports' note (line 82, about MCP callbacks not media), and serve.py SSE/HTTP. No Live session, no audio/PCM frames, no WebSocket media serving. docs/ROADMAP.md line 31 explicitly lists 'Realtime / voice API' as 'out of scope near-term'. Severity medium is fair (niche real-time voice/video capability).

### Native Anthropic / Gemini SDK integration and on-device runtimes
*Context management & Model usage · status: **partial** · effort: **large***

**ADK provides:** ADK 2.0 ships native Anthropic integration plus Gemma, LiteRT-LM (on-device), Apigee AI Gateway, and Vertex AI Model Garden as first-class model backends in addition to a LiteLLM wrapper.

**The gap (verified against code):** YAAB reaches Claude, Gemini, Gemma, Ollama, vLLM, and an arbitrary LiteLLM proxy/gateway (api_base) only THROUGH LiteLLM (yaab/models/, docs/models.md) — there is no native Anthropic or google-genai client, so Anthropic/Gemini-specific features (e.g. cache_control authoring, Gemini cachedContent, Live API) cannot be expressed. There is no LiteRT-LM / on-device runtime and no first-class Apigee AI Gateway or Vertex AI Model Garden backend (only a generic LiteLLM-proxy api_base). For pure text+tools this parity is fine; for vendor-native cost/latency features it is weaker.

**Verifier note:** Confirmed real (partial as claimed). All providers are reached only THROUGH LiteLLM: docs/models.md states the model layer is LiteLLMModel over LiteLLM's unified interface, with string ids like anthropic/claude-sonnet-4-6, gemini/gemini-2.0-flash, ollama/llama3, bedrock/...; the 'gateway' story is just api_base pointed at a LiteLLM proxy (models.md 'Pointing at a LiteLLM proxy', lines 79-83). Grep for native anthropic./google-genai/LiteRT/on-device/Apigee/Model Garden in yaab/*.py finds no native client (the 'native' hits are AG-UI event stream, the Rust graph engine, and DB-native vector types — unrelated). So Anthropic/Gemini-specific authoring features (cache_control, cachedContent, Live API) cannot be expressed, and there is no LiteRT-LM/on-device runtime nor a first-class Apigee/Vertex Model Garden backend. Severity medium is fair: text+tools parity is fine via LiteLLM; only vendor-native cost/latency features are weaker.

### Multimodal input wired end-to-end (image/audio/video)
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **medium***

**ADK provides:** ADK natively accepts image/audio/video input to the model and streams video.

**The gap (verified against code):** yaab/content.py models multimodal parts and Content.to_provider_content() renders DATA/FILE parts as OpenAI image_url items, and runner.py preserves multimodal parts (lines 668-671). But every part — image, audio, video alike — is rendered as type 'image_url' (content.py to_provider), so audio/video are not actually expressed in the provider payload; only image input round-trips correctly. No video streaming. This is image-input partial, not full multimodal.

**Verifier note:** Confirmed real. content.py Part.to_provider() (lines 83-92) renders EVERY non-text part as OpenAI type 'image_url': DATA parts become a data: URL image_url (lines 87-89) and FILE parts become a uri image_url (lines 90-91). There is no 'input_audio'/audio/video rendering branch, so audio/video bytes are mislabeled as images in the provider payload; only image input round-trips. No video streaming. Severity 'medium' is correct — image input genuinely works, so this is partial, not missing.

### Model-native code execution (BuiltInCodeExecutor / VertexAiCodeExecutor)
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **medium***

**ADK provides:** ADK offers BuiltInCodeExecutor (Gemini-native code execution) and VertexAiCodeExecutor (managed cloud sandbox) in addition to Container/UnsafeLocal executors.

**The gap (verified against code):** YAAB's code execution is a tool (yaab/tools/builtin/code.py python_exec) backed by SubprocessSandbox (default; explicitly 'not a security boundary') or DockerSandbox (real isolation). It does NOT use the model's native code-execution capability (no Gemini/Anthropic built-in code-execution path) and has no managed-cloud sandbox executor. So it matches ADK's ContainerCodeExecutor/UnsafeLocalCodeExecutor but lacks the two model/cloud-native executors.

**Verifier note:** Confirmed real. yaab/tools/builtin/code.py exposes python_exec as a tool backed by yaab/tools/sandbox.py, which ships exactly two backends: SubprocessSandbox (default, docstring says 'not a security boundary') and DockerSandbox (real container isolation). No model-native code-execution path (no Gemini/Anthropic built-in code execution) and no managed-cloud sandbox executor. Matches ADK's Container/UnsafeLocal executors but lacks BuiltInCodeExecutor and VertexAiCodeExecutor. Severity 'medium' fits (the two missing executors are convenience/managed paths).

### Long-running function tools (pause agent while external work completes)
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **medium***

**ADK provides:** ADK long-running function tools let a tool return control and pause the agent while external work completes, resuming when the result arrives.

**The gap (verified against code):** There is no first-class long-running tool primitive on the fast-path runner (grep for long_running/long.running/defer in runner.py returns nothing). The closest mechanisms are: (a) ToolApprovalPlugin's block mode raising ApprovalRequired for out-of-band resume (governance/approval.py), and (b) graph interrupt()/resume with durable checkpointers. These cover human-approval pauses but not the general 'tool kicks off external async work, agent suspends and is resumed with the result' pattern as a built-in tool contract. The A2A client polls remote tasks but that is a different layer.

**Verifier note:** Confirmed real. Grep for long_running/long-running/defer/suspend in runner.py returns nothing; the fast-path runner has no first-class long-running-tool contract. The only related mechanisms are governance/approval.py (ApprovalRequired for out-of-band human-approval resume) and graph interrupt()/resume with durable checkpointers — both human-pause patterns, not a 'tool kicks off async work, agent suspends, resumes with the result' tool primitive. The a2a/client.py poll_task() long-running-task support is a different (remote-agent) layer, as the claim states. Severity 'medium' is appropriate.

### OpenTelemetry metrics + token-usage metrics instruments
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **medium***

**ADK provides:** ADK emits token usage metrics and integrates OpenTelemetry metrics (counters/histograms) alongside traces.

**The gap (verified against code):** Token/cost usage IS captured — on spans as gen_ai.usage.* attributes (models/instrumented.py lines 51-53) and in the Usage object — but only as trace span attributes. There is NO OTel metrics pipeline: grep finds no Meter/MeterProvider/Counter/Histogram/Gauge and no metric_reader. So there are no aggregatable metrics instruments, no token-usage counters/histograms, and no Prometheus /metrics endpoint for dashboards/alerting.

**Verifier note:** Confirmed real. Token/cost usage is captured only as span attributes (models/instrumented.py:50-53 sets gen_ai.usage.input_tokens/output_tokens/cost_usd) and in the Usage object. Grep for Meter/MeterProvider/Counter/Histogram/Gauge/metric_reader/create_counter/create_histogram/PrometheusMetricReader/'/metrics' returns no real hits (the lone context.py match is a 'TokenCounter' type alias; README/ragas/deepeval 'metrics' hits are eval metrics, unrelated to OTel instruments). No metrics pipeline, no token counters/histograms, no Prometheus /metrics endpoint. Severity 'medium' is correct.

### Managed sessions/memory in deployment
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **large***

**ADK provides:** Vertex AI Agent Engine provides fully managed sessions and memory as part of the managed deployment.

**The gap (verified against code):** YAAB has durable SessionService/MemoryService/Saver backends (SQLite/Postgres/Redis per checkpoint.py and DEPLOYMENT.md) that the operator wires up themselves, but there is no managed hosting tier that provisions and operates them. It is self-managed durability vs ADK's managed-service durability — a real delta for teams who want zero-ops state.

**Verifier note:** Confirmed real. YAAB ships durable self-hosted backends (SQLiteSessionService/PostgresSessionService per DEPLOYMENT.md, SQLiteSaver checkpointers, SQLite/Postgres/Redis audit/registry) that the operator wires and runs themselves. Grep for 'managed session'/'Agent Engine'/managed deploy returns nothing — there is no managed hosting tier that provisions/operates state. Self-managed durability vs ADK's managed-service durability. Severity 'medium' fits.

### LLM-judged final-response match with reference rubric (final_response_match_v2)
*Evals, Guardrails, Audit, Governance · status: **partial** · effort: **medium***

**ADK provides:** ADK's final_response_match_v2 uses an LLM judge with a structured rubric to grade the final response against a reference, more robust than literal match.

**The gap (verified against code):** yaab has LLMJudge (governance/eval.py) which prompts a model to 'Rate the OUTPUT from 0 to 1 on {criteria}' and regex-extracts a single number, plus FaithfulnessEvaluator. This is a single-prompt freeform judge: no structured rubric, no per-criterion breakdown, no self-consistency/multi-sample, and brittle float parsing that returns 0.0 on any parse failure. It is weaker and less reliable than ADK's versioned rubric-based judge.

**Verifier note:** Real partial gap. LLMJudge.ascore (eval.py lines 143-156) sends one freeform 'Rate the OUTPUT from 0 to 1' prompt and regex-extracts a single float, returning 0.0 on any parse/exception. No structured rubric, no per-criterion breakdown, no self-consistency/multi-sample, no versioning. Weaker and more brittle than ADK's rubric judge, as claimed.

### Environment / tool simulation for eval
*Evals, Guardrails, Audit, Governance · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 adds simulated tool environments so eval can run deterministically without hitting real tools/services.

**The gap (verified against code):** There is no simulated tool-environment layer for evaluation. Nothing in yaab/eval, yaab/governance/eval.py, or tests provides mock/stubbed tool backends that an eval harness can swap in deterministically. Evaluators see only the final output; tools either run for real or the user must hand-wire fakes. No environment-simulation abstraction exists.

**Verifier note:** Real gap. Nothing in yaab/eval, yaab/governance/eval.py, or tests provides a swappable mock/stub tool-backend layer for deterministic eval. Evaluators only ever see the final output; there is no environment-simulation abstraction. (Note: a general TestModel exists for the model layer, but that stubs the LLM, not tool/service backends, so the eval-time tool-simulation gap stands.)

### Eval tab in web/dev UI
*Evals, Guardrails, Audit, Governance · status: **missing** · effort: **large***

**ADK provides:** ADK's adk web UI has an interactive Eval tab to create eval cases from sessions, run evals, and view pass/fail visually.

**The gap (verified against code):** yaab/web.py (the browser dev playground) has no eval/experiment/drift/compliance surface — grep across web.py for eval/Experiment/drift/compliance/guardrail/registry returned zero matches. The playground is a chat/run harness only; there is no UI to author eval cases from a session, run a dataset, or visualize scores/regressions.

**Verifier note:** Real gap. yaab/web.py serves a single self-contained chat playground over /chat/stream; the embedded HTML and Python contain no eval/experiment/drift/compliance/guardrail/registry surface (read in full). No UI to author eval cases from a session, run a dataset, or visualize scores.

### pytest integration / AgentEvaluator helper
*Evals, Guardrails, Audit, Governance · status: **partial** · effort: **small***

**ADK provides:** ADK provides AgentEvaluator.evaluate(...) for one-line pytest integration that runs an eval set and asserts criteria thresholds in CI.

**The gap (verified against code):** There is no AgentEvaluator-style assert helper. The docs (docs/evaluation.md) show using Experiment in CI, but the user must manually call await exp.run(...) and then assert on report.aggregate themselves; ExperimentResult exposes aggregate/mean_score but no threshold-checking, no pass/fail per criterion, no pytest fixture or assert_eval(...) convenience. test_eval_adapters.py confirms tests hand-roll the assertions. So CI integration exists but is lower-level and more boilerplate than ADK's one-call helper.

**Verifier note:** Real partial gap. No assert_eval/AgentEvaluator helper anywhere (grep: zero). ExperimentResult exposes aggregate/mean_score but no threshold check or per-criterion pass/fail. docs/evaluation.md shows manual 'await exp.run(...)' then '.aggregate', and test_eval_adapters.py hand-rolls all assertions. CI works but is lower-level than ADK's one-call helper, as claimed.

### Outcomes-analysis / back-testing regression workflow
*Evals, Guardrails, Audit, Governance · status: **partial** · effort: **medium***

**ADK provides:** (ADK side: covered by eval sets + CLI replay enabling regression/back-testing across versions.)

**The gap (verified against code):** DriftMonitor (monitor.py) tracks a rolling score series and flags regressions, and the SR 11-7 mapper explicitly marks SR11-7.2c outcomes-analysis as PARTIAL ('attach Evaluator ExperimentResults to the registry entry'). But there is no mechanism to actually attach an ExperimentResult to an AgentCard, no version-to-version eval comparison/back-test runner, and DriftMonitor scores are fed manually via record_score() with no automatic pipeline from Experiment results. The regression story is present in pieces but not an end-to-end workflow.

**Verifier note:** Real partial gap. DriftMonitor (monitor.py) tracks a rolling score series with manual record_score() and flags regressions; sr_11_7.py explicitly marks SR11-7.2c 'Outcomes analysis / back-testing' as PARTIAL with note 'attach Evaluator ExperimentResults to the registry entry'. But AgentCard (registry.py) has no typed ExperimentResult/eval field (only extra='allow' free-form), there is no version-to-version eval comparison/back-test runner, and no automatic pipeline from Experiment results into DriftMonitor. End-to-end workflow is absent, as claimed.

### Memory recall exposed as an agent tool
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **partial** · effort: **small***

**ADK provides:** ADK provides load_memory / preload_memory tools so an agent can query long-term memory on demand (or have it auto-injected) inside a run.

**The gap (verified against code):** KnowledgeBase has as_tool() (rag/knowledge.py) but MemoryManager/MemoryService have no as_tool() or load_memory tool — grep shows as_tool only on Agent, Multiagent, and KnowledgeBase. Memory recall is only auto-injected by the Runner as a 'Relevant memory:' system message (runner.py ~540-568); the agent cannot deliberately query memory mid-turn the way ADK's load_memory tool allows.

**Verifier note:** Confirmed real. Grep for as_tool across yaab/ shows it only on Agent (agent.py:123), Multiagent (multiagent.py:44), and KnowledgeBase (rag/knowledge.py:131). Neither MemoryManager nor MemoryService nor InMemoryVectorMemory exposes as_tool() or a load_memory/preload_memory tool. Memory is only auto-injected by the Runner as a single SYSTEM message 'Relevant memory:\n{recalled}' (runner.py:540-544, via _memory_search at 549-567). The agent cannot deliberately query long-term memory mid-turn the way ADK's load_memory tool allows. Severity 'medium' reasonable — KnowledgeBase.as_tool() gives on-demand RAG retrieval, so the gap is specifically the long-term MemoryService surface, not retrieval-as-tool in general.

### Cloud/durable artifact storage
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **missing** · effort: **medium***

**ADK provides:** GcsArtifactService — versioned artifacts persisted to cloud blob storage (GCS), alongside InMemoryArtifactService.

**The gap (verified against code):** yaab/artifacts/__init__.py ships ONLY InMemoryArtifactService (a process-local dict); there is no GCS/S3/filesystem/DB artifact backend (grep for gcs/s3/blob/boto3/cloud.storage/FileArtifact finds none under artifacts/). Artifacts are lost on restart and cannot be shared across processes, and unlike sessions/vector-stores/checkpoints there is no registered 'artifact' component family in the registry. This is the weakest backend story in the whole cluster.

**Verifier note:** Gap is real but one sub-claim is imprecise. artifacts/__init__.py ships ONLY InMemoryArtifactService (process-local dicts _data/_meta); no GCS/S3/filesystem/DB artifact backend exists (grep gcs/s3/blob/boto3/cloud.storage/FileArtifact finds only checkpoint/content 'blob' uses, nothing under artifacts/). And crucially, unlike sessions/vectorstores/checkpointers which each call register(...) with concrete factories, NO register('artifact', ...) is called anywhere in the repo, so zero artifact backends are selectable by name. IMPRECISION: the claim says 'no registered artifact component family' — but extensions.py:40 DOES reserve the 'artifact' -> 'yaab.artifacts' entry-point group (the family namespace exists; it is just unpopulated with backends). Materially the conclusion holds. I lower severity to 'medium': durable artifacts matter, but artifacts are a comparatively niche surface (binary blobs) vs sessions/memory, and a durable backend is a straightforward Protocol impl; calling it 'high' overstates it next to the memory gaps.

### Artifacts attached to / persisted with sessions
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **weaker** · effort: **medium***

**ADK provides:** ADK artifacts are versioned and attached to a session (and user) and persist with the session backend.

**The gap (verified against code):** ArtifactManager (artifacts/manager.py) tracks versions in a plain in-process dict (self._versions) and scopes by app:user:session string keys, but neither the version index nor the bytes are persisted by any durable backend, and the Session model (sessions/base.py) has no artifact linkage. Even if a durable ArtifactService existed, the version index that maps name->version->id lives only in process memory, so versioning does not survive a restart.

**Verifier note:** Confirmed real. ArtifactManager (artifacts/manager.py) holds the name->version->id index in a plain in-process dict self._versions (line 23), keyed by 'app:user:session:name' strings; nothing persists it. The Session model (sessions/base.py:18-23) has only id, messages, state — no artifact field or linkage (grep 'artifact' across yaab/sessions/ returns nothing). Even with a durable ArtifactService, the version index lives only in process memory, so versioning does not survive restart. Claim accurate.

### Session rewind (roll a session back to an earlier event)
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **missing** · effort: **medium***

**ADK provides:** ADK 2.0 session REWIND: roll a session back to a prior event, discarding later events.

**The gap (verified against code):** The Session model (sessions/base.py) stores a flat messages list + state dict with no per-event version/id sequence the session layer can roll back to, and no SessionService/SessionManager method does rewind/rollback/revert/truncate (grep across yaab/ finds rewind only absent here; checkpoint/time-travel exists only in the separate graph engine, graph/checkpoint.py, which operates on graph state per (thread_id, step), not on session conversation history). Sessions cannot be rolled back to an earlier turn.

**Verifier note:** Confirmed real. Session (sessions/base.py) stores a flat messages list + state dict with no per-event id/version sequence. SessionService protocol (base.py:26-38) exposes only get/get_or_create/save/append/delete; SessionManager (sessions/manager.py) adds scoping/listing/state but no rewind/rollback/revert/truncate (grep across yaab/sessions/ finds none). Time-travel/checkpoint exists only in the separate graph engine (graph/checkpoint.py) keyed by (thread_id, step) on graph state, not on session conversation history. Sessions cannot be rolled back to an earlier turn. Claim accurate.

### Session migration between services / schema versions
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **missing** · effort: **medium***

**ADK provides:** ADK 2.0 session MIGRATION: move sessions between session services or upgrade schema versions.

**The gap (verified against code):** No migration/export-import utility exists for sessions. There is no schema version field on Session (sessions/base.py) and no helper to copy sessions between, e.g., InMemory and Postgres backends or to upgrade an on-disk schema. Sessions are JSON-blob'd per backend (postgres.py data JSONB, sqlite.py data TEXT, redis.py JSON) with no version tag, so cross-backend or cross-version migration is entirely manual.

**Verifier note:** Confirmed real. No migration/export-import/version-upgrade utility for sessions anywhere (grep migrat/rewind/rollback/export-import across yaab/ returns nothing). Session model (sessions/base.py) has no schema version field. Backends serialize the session as an opaque JSON blob with no version tag (postgres.py/redis.py/sqlite.py), so cross-backend or cross-schema migration is entirely manual. Claim accurate.

### Native search grounding tools (google_search, VertexAiSearchTool)
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **weaker** · effort: **medium***

**ADK provides:** First-class grounding tools: google_search grounding and VertexAiSearchTool for enterprise document/datastore search, returning grounded results with citations.

**The gap (verified against code):** yaab/tools/builtin/search.py provides only a generic web_search tool that does nothing until the app registers an async provider via set_search_provider() (Tavily/Brave/SerpAPI); with no provider it returns a config hint. There is no built-in google_search grounding, no VertexAiSearch/enterprise-datastore connector, and no grounding-metadata/citation plumbing from search results (grep for google_search/VertexAiSearch/grounding finds only this generic hook + a config reference). So enterprise document-search grounding is bring-your-own-everything.

**Verifier note:** Confirmed real. tools/builtin/search.py provides only a generic web_search tool gated on set_search_provider() (returns a config hint when no provider is set, lines 34-39). Grep for google_search/VertexAiSearch/grounding/grounding_metadata across the whole repo returns NO matches. There is no built-in google_search grounding, no Vertex AI Search / enterprise-datastore connector, and no grounding-metadata/citation plumbing from search results. Enterprise document-search grounding is bring-your-own. Claim accurate.

### True hybrid (sparse + dense) retrieval
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **weaker** · effort: **medium***

**ADK provides:** Vertex RAG / Vertex Search support hybrid retrieval (dense vectors + sparse/keyword) with server-side fusion.

**The gap (verified against code):** The only 'hybrid' in YAAB is KeywordReranker (rerank.py), which post-hoc blends a lexical-overlap score onto the dense vector score of already-retrieved candidates. There is no sparse/BM25 index, no parallel dense+sparse retrieval, and no reciprocal-rank fusion (grep for bm25/sparse/rrf/reciprocal.rank finds none). Recall is purely dense; lexical signal only re-ranks what dense already surfaced, so exact-term/rare-token queries that dense recall misses are never recovered.

**Verifier note:** Confirmed real. The only 'hybrid' is KeywordReranker (rag/rerank.py:31-52), which post-hoc blends lexical query-term overlap onto the dense score of ALREADY-retrieved candidates. Grep for bm25/sparse/rrf/reciprocal.rank across yaab/ finds none. No sparse/BM25 index, no parallel dense+sparse retrieval, no reciprocal-rank fusion. The OpenSearch external store (stores_external.py:262-288) issues a pure knn vector query, not a hybrid bool query. Recall is purely dense; lexical signal only re-ranks what dense surfaced. Claim accurate.

### Dynamic nodes created at runtime
*Orchestration, Agent flows, Run lifecycle · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 workflow runtime can create graph nodes dynamically at runtime (graph topology expanded during execution).

**The gap (verified against code):** yaab/graph/state.py builds the topology eagerly: StateGraph.compile() snapshots nodes/edges and calls _core.plan_supersteps once; CompiledGraph holds a fixed self.graph.nodes/edges and ainvoke only walks _successors over the pre-declared edges/conditional map. A node function returns state updates (dict), not new nodes/edges; there is no API to add a node mid-run or spawn fan-out tasks whose count is decided at runtime. The closest substitute is MapAgent (data fan-out) which is not graph-topology-dynamic.

**Verifier note:** Verified in yaab/graph/state.py: StateGraph.compile (line 168) calls _core.plan_supersteps once over a fixed node/edge snapshot; CompiledGraph.ainvoke walks pre-declared edges via _successors (lines 209-215). A node returns a state-update dict, not new nodes/edges. No API to add nodes mid-run. MapAgent (multiagent.py) is data fan-out, not graph-topology-dynamic.

### Structured agent-to-agent Task API (multi-turn / single-turn task mode)
*Orchestration, Agent flows, Run lifecycle · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 Task API provides structured A2A delegation: a multi-turn task mode, single-turn controlled-output mode, mixed delegation patterns, HITL within a task, and task agents usable as workflow nodes.

**The gap (verified against code):** No Task abstraction exists. Grep for class Task / multi-turn / single-turn / task_mode / TaskAgent in yaab/ found nothing except docstrings and an A2A client timeout. Delegation is fire-and-return (AgentTool.execute runs the sub-agent once and returns output). The A2A server endpoint (yaab/serve.py /a2a/tasks) runs the agent once and returns a 'completed' task synchronously - _A2A_TASKS is an in-process dict with no multi-turn continuation, no task-state machine (working->input-required->completed), and no resume of a parked task. There is no single-turn-controlled vs multi-turn task distinction and no 'task agent as workflow node' type.

**Verifier note:** Verified: no Task/TaskAgent class anywhere (grep zero code matches). yaab/serve.py /a2a/tasks (lines 132-142) runs agent.run once and returns {'status': {'state': 'completed'}}; _A2A_TASKS (line 22) is an in-process dict with no working->input-required->completed state machine and no resume of a parked task. The a2a client (yaab/a2a/client.py) only POSTs and polls; no multi-turn continuation or single-turn-controlled mode. Gap is real. NOTE on severity: its concrete sub-deltas (multi-turn state machine, single-turn-controlled, task-agent-as-node, HITL-within-task) overlap heavily with already-confirmed gaps #1/#5/#7; as an independent A2A-protocol-completeness gap its standalone severity is closer to medium than high.

### Ambient / long-running background agents
*Orchestration, Agent flows, Run lifecycle · status: **missing** · effort: **large***

**ADK provides:** ADK 2.0 run lifecycle includes ambient agents - long-running background agents that run detached from a request/response.

**The gap (verified against code):** YAAB has no ambient/background agent runtime. Grep for ambient/background/run_live across the package found only doc/spec references, no implementation. Runner.run/run_stream are request-scoped coroutines; there is no scheduler, no detached run registry, no lifecycle to start/observe/stop a persistent agent. batch.py is bounded one-shot concurrency, not a persistent ambient loop. serve.py exposes only synchronous /run, /run/stream, /a2a/tasks - no endpoint to launch or manage a detached run.

**Verifier note:** Verified: grep for ambient/background/run_live across yaab/ returns zero code matches (only docs/multi-agent.md references). Runner.run/run_stream (yaab/runner.py) are request-scoped coroutines with no scheduler or detached run registry. yaab/batch.py is bounded one-shot concurrency (asyncio.gather over a Semaphore, batch_map lines 47-80), not a persistent loop. serve.py exposes only synchronous /run, /run/stream, /a2a/tasks. No detached-run lifecycle.

### Escalation-based loop exit driven by sub-agents
*Orchestration, Agent flows, Run lifecycle · status: **weaker** · effort: **small***

**ADK provides:** ADK LoopAgent exits when a sub-agent signals escalation (e.g. via EventActions/escalate), so a sub-agent inside the loop can decide to stop the loop from within.

**The gap (verified against code):** yaab/multiagent.py LoopAgent stops only via an external until(output)->bool callback evaluated on the agent's final output, or max_iterations. There is no escalation signal a sub-agent (or a tool inside it) can raise to break the loop; the loop body cannot self-terminate except by producing output the external predicate happens to match. Likewise SequentialAgent.stop_when inspects output text, not an agent-raised escalation. This is the simplified/weaker form of ADK's escalation-driven exit.

**Verifier note:** Verified in yaab/multiagent.py: LoopAgent.run (lines 187-199) stops only via external until(output)->bool or max_iterations; SequentialAgent.stop_when (lines 71-84) inspects output. Grep for escalate/EventActions/actions in exceptions.py and types.py returns zero matches - no in-loop escalation signal a sub-agent/tool can raise to break the loop. Confirmed weaker form.

### Distinct agent-lifecycle callbacks (before/after agent vs run) and sync callbacks
*Orchestration, Agent flows, Run lifecycle · status: **partial** · effort: **medium***

**ADK provides:** ADK provides callbacks at every lifecycle point - before/after agent, before/after model, before/after tool - in both sync and async forms.

**The gap (verified against code):** yaab/plugins/__init__.py covers before/after run (== agent invocation), before/after model, before/after tool, plus repair_tool_args and on_user_message - good coverage. Two deltas: (1) all hooks are async-only (no sync variant), so a purely-sync callback must be wrapped; minor. (2) There is no separate before/after AGENT distinct from before/after RUN for nested/sub-agent invocations - because workflow agents (multiagent.py) call child agent.run() directly without routing through the parent Runner's plugin chain, a plugin registered on the Runner does NOT fire before/after each sub-agent in a Sequential/Parallel/Loop/Swarm composition (each child constructs or reuses its own Runner via Agent._get_runner). So per-sub-agent lifecycle callbacks in a composed workflow are not guaranteed.

**Verifier note:** Verified in yaab/plugins/__init__.py: all hooks (before_run/after_run, before_model/after_model, before_tool/after_tool, repair_tool_args, on_user_message) are declared 'async def' (lines 31-68) - no sync variant. Workflow agents (yaab/multiagent.py SequentialAgent/ParallelAgent/LoopAgent/Swarm) call child agent.run() directly, and Agent._get_runner (agent.py lines 129-134) constructs/reuses a per-agent Runner, so a plugin on the parent Runner does not fire per-sub-agent in a composition. Both deltas confirmed; gap is the absence of (a) sync hook forms and (b) guaranteed per-sub-agent (vs per-run) callbacks.

### A2A push notifications
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **missing** · effort: **medium***

**ADK provides:** ADK implements the A2A push-notification spec: clients register a webhook and receive task-completion callbacks for long-running tasks instead of polling.

**The gap (verified against code):** No push-notification or webhook support anywhere. serve.py implements polling (GET /a2a/tasks/{id}) and SSE streaming only; grep for push/webhook in serve.py returns nothing. Long-running A2A clients must poll (RemoteAgent.poll_task). The A2A card hard-codes capabilities={'streaming': True} with no pushNotifications capability.

**Verifier note:** Confirmed. Grep for push/webhook/pushNotification across the repo returns only CI yaml and unrelated Rust Vec.push/JSONB 'pushes down' prose — nothing in serve.py. serve.py implements polling (GET /a2a/tasks/{id}) and SSE (POST /a2a/tasks/stream) only; client poll_task in a2a/client.py. AgentCard.to_a2a_card hard-codes capabilities={'streaming': True} with no pushNotifications. Severity medium is appropriate.

### A2A client streaming consumption
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **partial** · effort: **medium***

**ADK provides:** ADK's RemoteA2aAgent consumes the remote agent's streamed task updates (SSE) and surfaces incremental events to the caller.

**The gap (verified against code):** The YAAB server emits SSE task events at POST /a2a/tasks/stream (serve.py), but the RemoteAgent client only does a single blocking POST to /a2a/tasks and extracts final artifact text (yaab/a2a/client.py run()); there is no client method to consume the streaming endpoint. Client gets only final output or must poll.

**Verifier note:** Confirmed. yaab/a2a/client.py RemoteAgent.run() does a single blocking POST to /a2a/tasks and pulls final artifact text via _extract_text; the only other methods are get_task/poll_task (polling). There is NO client method that opens/consumes the server's POST /a2a/tasks/stream SSE endpoint. The server SSE endpoint exists (serve.py) and is exercised only by server-side tests (test_a2a_depth.py / test_serve_endpoints.py drive client.stream directly), not via RemoteAgent. Severity medium is appropriate.

### MCP SSE / streamable-HTTP transports (client)
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **partial** · effort: **medium***

**ADK provides:** ADK's MCPToolset connects to MCP servers over stdio, SSE, and streamable-HTTP transports out of the box.

**The gap (verified against code):** MCPClient (yaab/tools/mcp_client.py) ships a real stdio transport (subprocess + line-delimited JSON-RPC) and from_transport(callable) for 'HTTP/SSE servers or tests', but there is NO built-in SSE or streamable-HTTP transport implementation - the user must hand-write the async RPCTransport callable (no httpx-based HTTP transport, no SSE event-stream reader, no session/Mcp-Session-Id handling). Only stdio works without bespoke glue.

**Verifier note:** Confirmed. yaab/tools/mcp_client.py ships only _stdio_transport (subprocess + line-delimited JSON-RPC). from_transport(callable) and the RPCTransport type require the adopter to hand-write the async transport; there is no httpx-based HTTP transport, no SSE event-stream reader, and no Mcp-Session-Id handling (only stdio works turnkey). Gap is real. However, I'm lowering severity from high to medium: writing an httpx POST RPCTransport for streamable-HTTP is a small amount of glue, the seam is explicitly provided and documented, and stdio (the most common local-server transport) works out of the box — this is a convenience/batteries-included gap, not a capability the architecture precludes.

### A2UI (Agent-to-User-Interface protocol)
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **missing** · effort: **large***

**ADK provides:** ADK (Dec 2025) integrates A2UI: agents return declarative UI component trees (Card, Button, TextField from a trusted catalog) rendered progressively by clients; powers Gemini Enterprise, Opal, Flutter GenUI.

**The gap (verified against code):** No A2UI support of any kind. grep for A2UI/GenUI/component-tree/Card/render across yaab returns nothing relevant (extensions.py 'component' is an unrelated plugin-factory registry, not UI components). AG-UI (agui.py) streams text/tool/thinking events but emits no declarative UI component trees. This is a brand-new protocol; absence is a parity-marketing gap.

**Verifier note:** Confirmed. Grep for A2UI/GenUI/component-tree/UIComponent/Card(/Button/TextField/declarative-UI returns nothing relevant (extensions.py 'component' is a plugin-factory registry; web.py 'button' is the playground HTML). agui.py emits only text/tool/thinking events, no declarative UI component trees. Brand-new protocol, absent in YAAB. Severity medium is reasonable for a parity-marketing gap.

### AG-UI bidirectional state & human-in-the-loop
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **partial** · effort: **medium***

**ADK provides:** ADK's CopilotKit/AG-UI integration supports the full AG-UI event set including STATE_SNAPSHOT/STATE_DELTA shared state, frontend-executed tools, and human-in-the-loop input requests round-tripped from the UI.

**The gap (verified against code):** agui.py translates run_stream into a solid subset: RUN_STARTED/FINISHED/ERROR, TEXT_MESSAGE_*, TOOL_CALL_*, TOOL_CALL_RESULT, THINKING. But STATE_SNAPSHOT is declared in AGUIEventType and STATE_DELTA is named in the module docstring yet NEITHER is ever emitted (no state-sync), and the SSE app (agui_sse_app) is one-directional output only - it accepts an initial prompt/messages but has no channel for frontend tool results or human-in-the-loop responses to flow back into the run. Tests confirm only the output-event subset.

**Verifier note:** Confirmed. yaab/agui.py declares AGUIEventType.STATE_SNAPSHOT and mentions STATE_DELTA in the docstring, but neither is ever emitted by run_agui (no state-sync code path). agui_sse_app's POST /agui is output-only: it reads an initial prompt/messages and streams events out, with no channel for frontend tool results or human-in-the-loop responses to flow back into the run. tests/test_agui.py covers only the output-event subset (RUN_STARTED/FINISHED, TEXT_MESSAGE_*, TOOL_CALL_*, THINKING). Note: YAAB does have HITL via the graph (ctx.interrupt) and fast-path tool approval, but NOT round-tripped through the AG-UI channel as ADK/CopilotKit does. Severity medium is appropriate.

### OAuth2 token introspection/JWKS implementation
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **weaker** · effort: **medium***

**ADK provides:** ADK provides working OIDC/JWKS token verification and Google service-account credential handling for serving auth.

**The gap (verified against code):** yaab/auth.py OAuth2 only delegates to a user-supplied `validator(token)->identity` callback and emits a static authorizationCode flow descriptor; it ships NO actual JWKS verification, token introspection, expiry/audience/issuer validation, or scope enforcement (describe() returns empty scopes={}). It is a correct seam but every real validation step is left to the adopter. BearerTokenAuth/APIKeyAuth are static dict lookups. Production-grade IdP integration is BYO.

**Verifier note:** Confirmed. yaab/auth.py OAuth2 only delegates to a user-supplied validator(token)->identity and emits a static authorizationCode descriptor with scopes={}. No JWKS verification, token introspection, or expiry/audience/issuer/scope validation ships (the only 'introspect'/'jwks' references are docstring guidance in auth.py and example snippets in docs/serving.md and docs/DEPLOYMENT.md). BearerTokenAuth/APIKeyAuth are static dict lookups. Production IdP integration is BYO, as claimed. Severity medium is appropriate.

---

## LOW — niche / nice-to-have

### Plugins auto-applied across all agents from a packaged bundle
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **partial** · effort: **small***

**ADK provides:** ADK plugins are packaged lifecycle-hook bundles registered on a runner and applied across all agents, with a plugin registry; install a plugin (e.g. save-files-as-artifacts) and it applies globally.

**The gap (verified against code):** Runner-level application across agents IS present (runner.plugins loop). But there is no plugin auto-discovery/auto-application: extensions.py declares the entry-point group 'yaab.plugins' and get('plugin', name) can instantiate one, yet pyproject.toml declares NO yaab.plugins entry points and nothing wires discovered plugins onto a Runner automatically — you must manually pass plugins=[...]. There's also no built-in 'save artifacts as files' style plugin (artifacts exist separately in yaab/artifacts but aren't exposed as a drop-in plugin). So the 'pip install a plugin and it just applies' experience is absent.

**Verifier note:** Confirmed real but over-rated. Runner-level application across agents IS present (runner.py:61, 68-70, and the plugin loops). The entry-point group 'yaab.plugins' is declared in yaab/extensions.py:46 and get('plugin', name) can instantiate one, BUT pyproject.toml declares only [project.entry-points."yaab.compliance"] (lines 90-95) and no yaab.plugins entries; nothing in the codebase auto-discovers plugins and wires them onto a Runner (every usage is a manual Runner(plugins=[...]) — confirmed across serve.py (none), docs, examples, tests). plugins/builtins.py ships only AuditPlugin/CostBudgetPlugin/CachingPlugin — no 'save artifacts as files' drop-in (artifacts live in yaab/artifacts, not exposed as a plugin). The 'pip install a plugin and it just applies' experience is absent. But manual wiring is one line and the discovery seam exists, so this is a low-severity polish gap, not medium.

### Enterprise Integration Connectors (500+ connectors)
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **missing** · effort: **large***

**ADK provides:** ADK exposes Google Application Integration connectors — 500+ enterprise systems (Salesforce, ServiceNow, SAP, Jira, databases, etc.) as ready-made tools.

**The gap (verified against code):** No connector framework anywhere: grep for connector/enterprise/integration in the package finds only governance/monitor.py and authorization.py (unrelated uses) and docs. Enterprise system access requires hand-writing FunctionTools or going through MCP servers. This is the single biggest enterprise-tool-reach gap vs ADK, though it is largely a managed-cloud offering ADK gets from Google Cloud.

**Verifier note:** Confirmed real (no connector framework: 'connector' matches nothing in the package; the prior 'connector' hits were only docs). However this is fundamentally a Google-Cloud-managed Application Integration offering ADK gets for free from GCP, not a library capability YAAB could ship as code — MCP servers are the realistic substitute. Marking revised_severity low (the audit already noted it is 'largely a managed-cloud offering'; for an OSS SDK comparison this should not read as a medium code gap).

### Anthropic/Claude Skills format support
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **missing** · effort: **medium***

**ADK provides:** Requested ADK 2.0 capability: load reusable agent skills including the Anthropic/Claude Skills format (SKILL.md + frontmatter packaged skill folders).

**The gap (verified against code):** yaab/skills.py defines only YAAB's own in-code Skill class and a load_skills() that reads the 'yaab.skills' Python entry point. There is no loader for SKILL.md / frontmatter / Claude Skills directory layout (grep for SKILL.md/frontmatter/claude/anthropic in skills.py returns nothing). A skill authored in the portable Claude Skills format cannot be consumed by YAAB.

**Verifier note:** Confirmed real. yaab/skills.py defines an in-code Skill class (lines 29-62) and load_skills() (65-81) that only reads the 'yaab.skills' Python entry-point group. Grep for SKILL.md/frontmatter/claude/anthropic across the repo returns nothing. No loader for the portable Claude Skills folder layout (SKILL.md + YAML frontmatter). Severity low is fair (emerging/requested format).

### 'adk create' style YAML scaffolding
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **weaker** · effort: **small***

**ADK provides:** ADK CLI scaffolds a declarative agent (agent.yaml) so non-coders start from a data file, reinforcing the no-code path.

**The gap (verified against code):** yaab/cli.py _init / `yaab init <name>` writes a Python file (_STARTER template instantiates Agent in code), not a YAML spec. There is no command to scaffold an agent.yaml, and no `yaab run agent.yaml` command (serve/web take module:agent specs via _load_attr, not YAML paths). So even though agent_from_yaml exists, the CLI does not surface a YAML-first workflow.

**Verifier note:** Confirmed real. yaab/cli.py _init (lines 55-60) writes a Python file from _STARTER (lines 38-52) that instantiates Agent in code — not an agent.yaml. There is no scaffold command for YAML and no 'yaab run agent.yaml'; serve/web resolve module:agent specs via _load_attr (lines 63-66, 101-112), not YAML paths. No agent.yaml/.yml samples exist in the repo. agent_from_yaml exists in config.py but the CLI never surfaces a YAML-first workflow. Severity low is fair.

### Declarative skill definition (skill body in YAML)
*YAML specs, Callbacks, Plugins, Skills, Extensibility · status: **partial** · effort: **small***

**ADK provides:** ADK skills are reusable instruction+tools bundles that can be defined/declared as part of the agent config.

**The gap (verified against code):** In yaab/config.py _resolve_skills, the YAML 'skills' list can only reference skills already registered via the yaab.skills entry point by name (unknown names raise). You cannot define a skill inline in the YAML (instructions/tools/permissions) — skills must be authored in Python and installed as a package first. So the declarative surface references skills but cannot author them.

**Verifier note:** Confirmed real. yaab/config.py _resolve_skills (lines 56-65) calls load_skills() (from skills.py entry points) and raises ValueError for any name not already registered — you can only reference pre-installed skills by name, never define instructions/tools/permissions inline in the YAML. Skills must be authored in Python and installed as a 'yaab.skills' entry point first. Severity low is fair.

### Structured thinking/reasoning config passthrough (thinking_budget)
*Context management & Model usage · status: **partial** · effort: **small***

**ADK provides:** ADK 2.0 passes thinking/reasoning config through to the model, e.g. thinking_budget for Gemini 2.5 (and a structured thinking_config), as a first-class, validated control.

**The gap (verified against code):** YAAB has only a generic untyped escape hatch: Agent(model_settings={...}) forwards arbitrary kwargs to model.complete (yaab/runner.py line 589). A user can hand-pass reasoning_effort or a raw extra_body thinking_config, but there is no first-class, typed thinking_budget/thinking_config field, no per-provider normalization (OpenAI reasoning_effort vs Gemini thinking_budget vs Anthropic thinking.budget_tokens), and no validation. The agent docstring lists reasoning_effort only as an example kwarg; tests/test_model_settings.py confirms it is plain passthrough. So it works for experts but is not the discoverable, normalized, fool-proof control ADK provides.

**Verifier note:** Confirmed real (partial as claimed). Only a generic untyped escape hatch exists: Agent(model_settings={...}) (agent.py lines 73-76) forwards arbitrary kwargs to model.complete via **getattr(agent,'model_settings',{}) (runner.py line 589). tests/test_model_settings.py confirms plain passthrough (lines 1-7 docstring lists reasoning_effort among arbitrary kwargs; tests only assert temperature/seed/top_p forwarding). No typed thinking_budget/thinking_config field, no per-provider normalization (OpenAI reasoning_effort vs Gemini thinking_budget vs Anthropic thinking.budget_tokens), no validation. ModelResponse.reasoning only READS a thinking trace back (base.py line 29; litellm_provider.py line 178). Severity low is fair — works for experts, just not normalized/discoverable.

### Cloud Trace integration
*Deployment, Runtime, Observability, Built-in tools · status: **partial** · effort: **small***

**ADK provides:** ADK has first-class Google Cloud Trace export integration.

**The gap (verified against code):** YAAB emits standard OTel GenAI-convention spans (observability/__init__.py, instrumented.py), so a user CAN export to Cloud Trace by configuring an OTLP/Cloud Trace exporter themselves — but there is no built-in Cloud Trace exporter wiring or helper; DEPLOYMENT.md just says 'configure an exporter'. Functional via generic OTel, but not the turnkey Cloud Trace integration ADK ships.

**Verifier note:** Confirmed real. YAAB emits generic OTel GenAI-convention spans (observability/__init__.py, models/instrumented.py), so a user can self-configure an OTLP/Cloud Trace exporter, but there is no built-in Cloud Trace exporter or helper — grep for CloudTrace/CloudTraceSpanExporter/cloud_trace finds nothing, and DEPLOYMENT.md only says 'configure an exporter'. Functional via generic OTel but not the turnkey integration ADK ships. Severity 'low' is correct — this is a GCP-specific convenience over already-present standard OTel.

### Structured logging
*Deployment, Runtime, Observability, Built-in tools · status: **missing** · effort: **medium***

**ADK provides:** ADK provides structured logging across the runtime.

**The gap (verified against code):** There is no structured logging framework in the runtime — grep across yaab/ for logging.getLogger/structlog/import logging finds a single ad-hoc logging.getLogger('yaab').warning call in memory/__init__.py and nothing else. No per-run structured log records, no JSON log formatter, no log correlation with trace IDs. Observability is span/audit-based only.

**Verifier note:** Confirmed real. Grep across yaab/ for getLogger/structlog/import logging/JsonFormatter finds a single ad-hoc call: memory/__init__.py:95-97 (`import logging; logging.getLogger('yaab').warning(...)`). No structured-logging framework, no JSON formatter, no per-run log records, no trace-ID/log correlation. Observability is span- and audit-based. Note 'missing' is slightly strong since one logger call exists, but as a *framework* it is effectively absent; severity 'low' is appropriate (audit log + spans cover most needs).

### ROUGE / response_match_score metrics
*Evals, Guardrails, Audit, Governance · status: **partial** · effort: **small***

**ADK provides:** ADK provides response_match_score and rouge as built-in text-overlap criteria for comparing response to a reference.

**The gap (verified against code):** yaab/governance/eval.py ships ExactMatch, Contains, Regex, JSONMatch, NumericTolerance, and Levenshtein (normalized edit-distance). Levenshtein is a reasonable similarity proxy, but there is no ROUGE-N/ROUGE-L (n-gram recall) and no named response_match_score; rag/eval.py's lexical faithfulness/context_relevance are groundedness metrics, not reference-overlap scores. So the specific ADK text-overlap criteria are only loosely approximated.

**Verifier note:** Real partial gap. eval.py ships ExactMatch/Contains/Regex/JSONMatch/NumericTolerance/Levenshtein; rag/eval.py adds lexical faithfulness/context_relevance (groundedness, not reference-overlap). Grep for rouge/response_match/final_response_match: zero matches, including across the RAGAS/DeepEval adapters. No ROUGE-N/ROUGE-L and no named response_match_score; Levenshtein is the only similarity proxy. Low severity is appropriate.

### Custom-metrics plugin contract for eval suite
*Evals, Guardrails, Audit, Governance · status: **partial** · effort: **small***

**ADK provides:** ADK 2.0 documents a custom-metrics plugin mechanism so teams add their own eval criteria into the standard eval pipeline/CLI/UI.

**The gap (verified against code):** YAAB's metric registry (yaab/eval/__init__.py) already lets anyone register_metric(...) or ship metrics via the yaab.metrics entry point, and the Evaluator/ascore protocol is clean and extensible — arguably better than ADK at the library level. But because YAAB has no eval CLI, no evalset file format, and no eval UI, a custom metric cannot be wired into a standard pipeline/CLI/UI run the way ADK's plugin metrics can; extensibility is library-only.

**Verifier note:** Real partial gap. yaab/eval/__init__.py provides register_metric/get_metric/available_metrics plus a 'yaab.metrics' entry point and a clean ascore/evaluate protocol (library-level extensibility is genuinely strong). But with no eval CLI, no evalset file format, and no eval UI, a custom metric cannot be wired into a standard pipeline/CLI/UI run the way ADK plugin metrics can. Library-only, as claimed; low severity fits.

### Gemini/provider built-in safety-settings passthrough
*Evals, Guardrails, Audit, Governance · status: **partial** · effort: **small***

**ADK provides:** ADK passes through Gemini's native safety settings (harm-category thresholds) and documents Model Armor integration as a provider-level guardrail.

**The gap (verified against code):** YAAB's guardrails are framework-side scanners (regex + Presidio/LLM-Guard/NeMo). I found no first-class plumbing to pass provider-native content-safety settings (e.g. Gemini harm_category thresholds, OpenAI moderation) through the model layer as a configured guardrail; safety is handled entirely by YAAB's own scanners. For Gemini-on-Vertex shops wanting native safety filters + Model Armor, YAAB has no documented passthrough, so it is weaker on provider-native safety even though its own scanner suite is stronger overall.

**Verifier note:** Partial gap stands, with a nuance worth recording: LiteLLMModel.__init__ accepts **default_params and complete()/_params() forward arbitrary kwargs straight into litellm.completion, and LiteLLM itself forwards Gemini safety_settings / OpenAI moderation params. So a user CAN pass safety_settings as a default param via the generic kwargs passthrough. What is genuinely missing is FIRST-CLASS, documented plumbing: grep for safety_settings/harm_category/HarmCategory/moderation/model_armor returns zero across the whole repo (no typed field, no docs, no Model Armor integration). The claim's wording ('no first-class plumbing... no documented passthrough') is accurate; low severity is correct.

### Memory namespace scoping efficiency / correctness for the default backend
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **weaker** · effort: **small***

**ADK provides:** ADK MemoryService scopes recall to (app, user) at the backend.

**The gap (verified against code):** MemoryManager.search (memory/manager.py) does NOT push the app_name/user_id filter into the backend; it over-fetches k*4 globally from InMemoryVectorMemory and then filters by namespace in Python. With many users this both leaks ranking budget across tenants (a heavy user can crowd out another's top-k before filtering) and degrades recall quality for any single user. Scoping is best-effort post-filtering, not a backend-level constraint.

**Verifier note:** Confirmed real. MemoryManager.search (memory/manager.py:48-64) calls self.service.search(query, k=k*4) with NO namespace filter pushed into the backend, then filters by app_name/user_id in Python and slices [:k]. InMemoryVectorMemory.search (memory/__init__.py:130-136) accepts no metadata filter, so scoping is necessarily post-hoc. With many tenants, a heavy user's records can crowd out another's within the global k*4 budget before filtering. Severity 'low' is appropriate — it only bites the in-memory default; the pgvector/external paths used in production push metadata filters into the DB.

### Default in-memory vector recall scalability
*Knowledge base / RAG, Memory, Sessions, Artifacts · status: **weaker** · effort: **medium***

**ADK provides:** Vertex-backed RAG/memory use ANN indexes that scale to large corpora.

**The gap (verified against code):** InMemoryVectorStore.query (rag/store.py) and InMemoryVectorMemory.search (memory/__init__.py) do an exact brute-force scan over all chunk embeddings (building the full matrix every query) before the Rust top-k. There is no ANN index (HNSW/IVF) for the default path, so the in-memory default degrades linearly and is unsuitable for large corpora without moving to an external store. Acceptable as a dev default but counts against the 'fast at scale' bar for the built-in path.

**Verifier note:** Confirmed real. InMemoryVectorStore.query (rag/store.py:56-64) and InMemoryVectorMemory.search (memory/__init__.py:130-136) both build the full embedding matrix every query and call _core.top_k, which is an EXACT brute-force scan (_core.py:69-74: scores every row, sorts, slices; the Rust path is likewise a full top_k, no ANN). No HNSW/IVF index on the default path, so it degrades linearly. Severity 'low' correct — explicitly a dev default, with pgvector/Chroma/Qdrant/OpenSearch/Pinecone ANN-capable stores available for scale.

### Explicit fan-in / join synchronization node
*Orchestration, Agent flows, Run lifecycle · status: **partial** · effort: **medium***

**ADK provides:** ADK 2.0 workflow runtime supports fan-out/fan-in with explicit join semantics.

**The gap (verified against code):** Fan-out works (multiple successors added to next_frontier; ParallelAgent). Fan-in is only implicit: when several nodes in a superstep write the same channel, the reducer (append/add/last_value) merges them (yaab/graph/state.py _advance/_apply). There is no explicit join/barrier node that waits for N named predecessors before firing with their combined outputs - a node simply runs when it appears in the frontier. For diamond patterns this works via reducers, but there is no first-class 'join' construct with per-branch result addressing, and a conditional edge returns exactly one target (list(self._successors) yields a single mapped target), limiting some fan patterns.

**Verifier note:** Verified in yaab/graph/state.py: fan-out works (multiple successors appended to next_frontier, lines 279-283; ParallelAgent in multiagent.py). Fan-in is implicit only via channel reducers (_apply/_advance, lines 195-207). A conditional edge returns exactly one target (_successors line 214: 'return [target]'). No first-class join/barrier node that waits for N named predecessors with per-branch result addressing. Confirmed partial.

### A2A one-liner exposure (to_a2a)
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **weaker** · effort: **small***

**ADK provides:** ADK exposes any agent as an A2A server with a single to_a2a(agent) call (or A2aAgentExecutor), wiring the full A2A spec automatically.

**The gap (verified against code):** YAAB has no to_a2a()/serve() convenience on the Agent (yaab/agent.py has run/stream/run_sync only; no serve/to_a2a method found). To expose an agent you must import yaab.serve.fastapi_server_app(agent) or serve(agent). None of serve/fastapi_server_app/RemoteAgent/agui/MCP symbols are re-exported from yaab/__init__.py, so discoverability is poor. Functionally close but more boilerplate than ADK's one-liner.

**Verifier note:** Confirmed. yaab/agent.py exposes only run/stream/stream_structured/stream_events/run_sync/as_tool/tool/reset — no serve()/to_a2a() method. Exposure requires yaab.serve.fastapi_server_app(agent) or serve(agent) (serve.py __all__=['fastapi_server_app','serve']). yaab/__init__.py re-exports none of fastapi_server_app/serve/RemoteAgent/agui/MCPClient/MCPServer. Functionally close (serve(agent) is one line) but no Agent-method one-liner and poor discoverability, as claimed. Severity low is correct.

### MCP Toolbox for Databases / managed MCP servers
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **missing** · effort: **large***

**ADK provides:** ADK ships MCP Toolbox for Databases - managed MCP servers for BigQuery, Spanner, Postgres, etc. with prebuilt DB tools.

**The gap (verified against code):** No managed/prebuilt MCP servers for databases or any data source. YAAB only provides the generic MCPServer/MCPClient plumbing; there is no equivalent toolbox of ready-made connectors. This is largely an ecosystem/content gap rather than a core-SDK gap.

**Verifier note:** Confirmed. Built-in tools are only yaab/tools/builtin/{calculator,code,datetime_tool,http,search}.py. The 'toolbox/bigquery/postgres' grep hits are all storage backends (sessions/postgres.py, rag stores) and docs, not prebuilt MCP DB connectors. No managed/prebuilt MCP servers exist. As the claim itself notes, this is an ecosystem/content gap, not a core-SDK gap; severity low is correct.

### Agent card completeness & scope-based security
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **partial** · effort: **small***

**ADK provides:** ADK generates full A2A AgentCards including provider, documentationUrl, defaultInputModes/defaultOutputModes, per-skill input/output modes, and security requirements (scopes) consistent with the A2A spec.

**The gap (verified against code):** AgentCard.to_a2a_card (registry.py) emits name/description/url/version, capabilities={'streaming':True} (hard-coded, no pushNotifications/stateTransitionHistory), skills (id+name only, no description/inputModes/outputModes/examples), and a governance block. serve.py adds securitySchemes but never a top-level `security` requirements array or scopes, so clients can't discover required OAuth scopes. Missing defaultInputModes/defaultOutputModes/provider/documentationUrl.

**Verifier note:** Confirmed. registry.py AgentCard.to_a2a_card emits name/description/url/version, hard-coded capabilities={'streaming': True} (no pushNotifications/stateTransitionHistory), skills as [{'id','name'}] only (no description/inputModes/outputModes/examples), plus an x-yaab-governance block. serve.py adds body['securitySchemes'] but never a top-level 'security' requirements array or scopes (grep for a 'security': [ array confirms none). No defaultInputModes/defaultOutputModes/provider/documentationUrl. Severity low is correct.

### ACP (Agentic Commerce Protocol)
*Protocols (A2A/AG-UI/MCP/ACP/A2UI), Serving, Auth · status: **missing** · effort: **large***

**ADK provides:** ACP (OpenAI/Stripe) checkout & payment flows for agents - per the brief this is NOT in ADK either; a shared market gap.

**The gap (verified against code):** Confirmed absent in YAAB (no checkout/payment/commerce protocol code). Per instructions this is a market gap for both ADK and YAAB, so it is not a competitive disadvantage vs ADK - listed here only for completeness at the lowest severity.

**Verifier note:** Confirmed absent. Grep for checkout/payment/commerce/stripe/ACP/agentic-commerce hits only CI yaml, docs/tools.md, and the 'approval_pipeline' sample name — no commerce-protocol code. Per the brief this is a shared market gap (also absent in ADK), so not a competitive disadvantage; severity low is correct.

---
