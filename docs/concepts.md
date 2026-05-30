# Concepts — what each component is for

YAAB has several components that *look* similar because many of them persist or
retrieve state. This page explains what each one is, when to reach for it, and —
importantly — **how to tell the confusable ones apart**.

## The mental model in one breath

> An **Agent** (model + tools + output type) is executed by a **Runner**, which
> records turns to a **Session**, can recall facts from **Memory** or documents
> from a **KnowledgeBase (RAG)**, and — when you use the durable **Graph** — saves
> its progress to a **Checkpointer**. **Governance** watches all of it.

## The confusing pairs, settled first

### Checkpointer vs. Session — *the* common mix-up

Both persist state, but they answer different questions:

| | **Session** | **Checkpointer** |
|---|---|---|
| Question it answers | "What was *said* in this conversation?" | "Where was this *workflow* when it stopped?" |
| What it stores | conversation **messages** + a KV **state** dict | a **graph's** full execution state at each superstep |
| Keyed by | `session_id` (a conversation/user thread) | `thread_id` (a graph run) |
| Used by | the **fast-path Runner** (`agent.run(session_id=…)`) | the **durable Graph** (`graph.compile(checkpointer=…)`) |
| Purpose | multi-turn chat memory + app state | crash recovery, resume, time-travel, human-in-the-loop pause |
| Granularity | append a message per turn | snapshot the whole state per superstep |
| Typical backends | in-memory, SQLite, Postgres/Aurora, Redis | in-memory, SQLite, Postgres/Aurora, Redis |

Rule of thumb: **Sessions are for conversations; Checkpointers are for resumable
graph workflows.** A chatbot needs a Session. A long approval pipeline that must
survive a restart needs a Checkpointer. They're independent — you can use both,
one, or neither.

### Memory vs. RAG (KnowledgeBase)

Both retrieve relevant text by similarity, but the *source* and *lifecycle*
differ:

| | **Memory** (`MemoryService`) | **RAG** (`KnowledgeBase`) |
|---|---|---|
| Source | things the agent *learned* — past conversations, facts it was told | documents you *ingest* — manuals, PDFs, wikis |
| Lifecycle | grows during use; per-user/app scoped | curated corpus; indexed ahead of time |
| Write path | `memory.add(...)`, `add_session_to_memory(...)` | `kb.add(documents)` (chunk → embed → store) |
| Read path | `memory.search(query)` | `kb.retrieve(query)` / `kb.as_tool()` |
| Typical use | "remember the user prefers email" | "answer from the employee handbook" |

Both can sit on the same vector store; the difference is *who fills it and why*.

### Session state vs. prefix-scoped State

`Session.state` is a plain dict scoped to one session. `State`
(`temp:`/`user:`/`app:` prefixes) routes keys to **different lifetimes** —
ephemeral, per-user, or app-global — on top of sessions. Use `State` when a value
must outlive a single conversation (a user preference) or never be persisted (a
one-turn flag).

### Tool authorization vs. tool approval vs. guardrails

All three gate behavior, at different points:
- **Guardrails** (`PolicyEngine`) — scan *text* (prompts/outputs) for injection,
  PII, secrets.
- **Tool authorization** (`ToolAuthorizationPlugin`) — decide *programmatically*
  whether a tool call may run (RBAC, capabilities) — no human.
- **Tool approval** (`ToolApprovalPlugin`) — pause for a *human* to approve a
  sensitive tool call.

### Optimizer vs. Evaluator

An **Evaluator/metric** *scores* an output (did it pass?). An **Optimizer** *uses*
a metric to improve a module's prompt/demos at build time. Evaluation measures;
optimization improves.

## Every component, by layer

### Core
- **Agent** — the typed unit of work: a model + instructions + tools + an output
  type (`Agent[Deps, Output]`). What you define.
- **Runner** — executes an agent: runs the tool loop, emits the event stream,
  wires in sessions/plugins/governance. The engine.
- **RunContext** — per-run context passed to tools/hooks (deps, identity, usage,
  scratch `state`). Like ADK's `ToolContext`.
- **Tool** — a capability the model can call (typed function, MCP tool,
  agent-as-tool, remote A2A agent).
- **Model (`ModelProvider`)** — the LLM backend; `LiteLLMModel` covers 100+
  providers, `TestModel` runs offline, `ResilientModel` adds rate-limit/breaker.

### State & retrieval
- **Session / SessionService / SessionManager** — conversation history + KV
  state; the manager adds app/user scoping. *(See "vs. Checkpointer" above.)*
- **Memory / MemoryService / MemoryManager** — long-term semantic recall of
  learned facts.
- **KnowledgeBase (RAG)** — ingest + retrieve documents; chunkers, embedders,
  vector stores, rerankers, citations, access control.
- **ArtifactService / ArtifactManager** — versioned binary/file storage (reports,
  images) — *not* text retrieval.
- **State** — prefix-scoped (`temp:`/`user:`/`app:`) values across lifetimes.

### Orchestration
- **Fast path** — `agent.run(...)`: model-driven tool loop. Default; simplest.
- **Graph (`StateGraph`)** — explicit nodes/edges/cycles with durable
  **Checkpointers** and human-in-the-loop. For deterministic, resumable control
  flow.
- **Multi-agent** — `SequentialAgent`, `ParallelAgent`, `MapAgent`, `LoopAgent`,
  `Swarm`; each is itself an agent.
- **Optimize** — DSPy-style `Signature`/`Module`/`Optimizer` to compile prompts.

### Governance
- **AgentRegistry** — system-of-record (Agent Cards, risk tier, approval, model
  inventory).
- **LifecycleManager** — the model-risk FSM with evidence-gated transitions.
- **PolicyEngine / guardrails** — input/output content scanners.
- **Tool authorization / approval** — gate tool calls (programmatic / human).
- **AuditLog** — tamper-evident, hash-chained record of everything.
- **Evaluator + metrics** — score outputs (incl. RAGAS/DeepEval adapters).
- **DriftMonitor / TrustScorer** — track quality over time.
- **ComplianceMapper** — project the above onto SR 11-7 / EU AI Act / etc.

### Platform & interop
- **MCPClient / MCPServer** — consume / expose tools over the MCP standard.
- **RemoteAgent (A2A)** — delegate to remote agents; also usable as a tool.
- **serve / web / AG-UI** — HTTP + A2A endpoints, a dev playground, and the
  AG-UI streaming protocol for frontends.
- **AuditSink** — where audit events go (SQLite, Langfuse, Logfire, OTel).
- **Plugin** — cross-cutting hooks (`before/after_*`) on the Runner.
- **Component registry (`yaab.extensions`)** — register/select any of the above
  by name; the basis of extensibility.

## "Which do I use?" cheat-sheet

| I want to… | Use |
|---|---|
| Build a chatbot that remembers the conversation | **Session** (`session_id=`) |
| Answer questions from my company docs | **RAG / KnowledgeBase** |
| Have the agent remember user facts over time | **Memory** |
| Run a multi-step workflow that survives crashes / pauses for approval | **Graph + Checkpointer** |
| Store a generated file/report | **Artifacts** |
| Keep a value only for this turn / across a user's sessions / app-wide | **State** (`temp:` / `user:` / `app:`) |
| Stop a tool from running without a human | **Tool authorization** |
| Require a human to approve a tool | **Tool approval** |
| Block prompt injection / PII | **Guardrails (PolicyEngine)** |
| Measure answer quality | **Evaluator / metrics** |
| Improve prompts automatically | **Optimizer** |
| Prove compliance | **Registry + Audit + ComplianceMapper** |
