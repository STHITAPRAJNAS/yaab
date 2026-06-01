# YAAB documentation

**Yet Another Agent Builder** — a type-safe, governance-first agent SDK with a
Rust performance core. Type-safe, optimizable, durable, and simple — the best
ideas from across the agent ecosystem on one runtime, on a universal LiteLLM
model layer.

YAAB is designed as a **drop-in upgrade path for teams on other agent frameworks**:
the same building blocks you expect — agents, runners, sessions,
memory, artifacts, tools, multi-agent workflows, MCP/A2A interop, streaming — plus
a first-class governance/registry/compliance layer that none of them ship.

## Start here

| Guide | What it covers |
|---|---|
| [**Get started**](get-started.md) | The guided path: install → agent → tools → RAG → memory → multi-agent → govern → deploy |
| [**Concepts**](concepts.md) | What each component is for, and how to tell the confusable ones apart (Checkpointer vs Session, Memory vs RAG, …) |
| [**Samples**](https://github.com/STHITAPRAJNAS/yaab/tree/main/samples) | End-to-end sample apps & patterns (support bot, RAG QA, approval pipeline, swarm, …), tested offline |
| [Quickstart](quickstart.md) | Three-line agent, tools, typed output, offline testing |
| [Agents](agents.md) | `Agent[Deps, Output]`, dependency injection, instructions, capabilities |
| [Tools](tools.md) | Typed function tools, agent-as-tool, MCP tools |
| [Models](models.md) | LiteLLM layer, fallbacks, retries, cost, structured output, TestModel |
| [State: sessions, memory, artifacts](state.md) | Managers + services, scoping, session→memory ingestion |
| [Storage & backends](storage-backends.md) | In-memory defaults + Postgres/Aurora, Redis, pgvector, OpenSearch, Chroma, Qdrant, Oracle — all extendible |
| [State scoping & AG-UI](state-and-agui.md) | `temp:`/`user:`/`app:` state prefixes; AG-UI streaming middleware |
| [RAG](rag.md) | Built-in retrieval: knowledge base, chunking, vector stores, rerank, citations, faithfulness |
| [Multi-agent](multi-agent.md) | Sequential, Parallel, Loop, Swarm, agent-as-tool |
| [Streaming & events](streaming-events.md) | Token streaming, the semantic event stream, SSE endpoints |
| [Usage limits & run control](limits.md) | Token/request/tool caps, cancellation, timeouts |
| [Robustness](robustness.md) | Built-in tools, context-window mgmt, HITL approval, resilience, YAML config |
| [Graph orchestration](graph.md) | Durable `StateGraph`, checkpoints, HITL, channels, time-travel |
| [Interop: MCP & A2A](interop.md) | MCP client/tools, A2A server + client (RemoteAgent) |
| [Governance & compliance](governance.md) | Registry, lifecycle, guardrails, audit, evals, compliance mappers |
| [Evaluation](evaluation.md) | Metric registry, RAGAS/DeepEval adapters, experiments, custom metrics |
| [Optimization](optimization.md) | Signature/Module/Optimizer, compiled artifacts |
| [Prompts & skills](prompts-skills.md) | Versioned prompt management, reusable skill bundles |
| [Serving & auth](serving.md) | `fastapi_server_app`, A2A server, bearer/API-key/OAuth2 |
| [Platform extensions](platform.md) | Doc loaders, Chroma/Qdrant, sandbox, structured streaming, batch, `yaab web`, sinks |
| [Extending YAAB](extending.md) | The component registry, protocols, entry points |
| [Deployment](DEPLOYMENT.md) | Local → Cloud Run / Fargate / Lambda / K8s, durable backends |

## The mental model

```
Agent  ── the typed unit of work (model + instructions + tools + output type)
Runner ── executes agents: event stream, sessions, plugins, governance
Graph  ── durable, checkpointed orchestration when you need explicit control
Governance ── registry + lifecycle + guardrails + audit + compliance (opt-in by mode)
yaab-core ── a Rust performance core for the compute-bound primitives (pure-Python fallback)
```

Three orchestration paths compose over **one runtime**:

1. **Fast path** — `agent.run(prompt)`: a model-driven tool loop.
2. **Graph path** — `StateGraph`: durable, checkpointed, HITL.
3. **Optimizable path** — `Module.compile(...)`: tune at build time, freeze for prod.

### Python vs Rust — the honest split

YAAB is **Python-first**. The whole developer API, the agent loop, the model
layer, governance, and orchestration *logic* are Python (~95% of the code). The
Rust core (`yaab-core`, ~325 lines) accelerates only the compute-bound
primitives — vector search, checkpoint serialization, channel reducers, BSP
superstep planning + the opt-in whole-superstep fold, and audit hashing — each
with a pure-Python fallback. The I/O-bound agent loop stays in Python on purpose
(the model/tool network calls dominate, not the loop). The durable graph also
exposes an explicit `compile(engine="rust"|"python"|"auto")` switch. See
[Graph › Choosing the engine](graph.md#choosing-the-engine-python-vs-rust).

## Install

```bash
pip install yaab-sdk                 # core; pure-Python performance core works everywhere
pip install 'yaab-sdk[litellm]'      # universal model layer
pip install 'yaab-sdk[otel]'         # OpenTelemetry tracing
pip install 'yaab-sdk[all]'          # everything

# Build the Rust accelerator (optional; auto-falls back if absent):
pip install maturin && maturin develop -m yaab-core/Cargo.toml --release
```

Check the active core:

```python
import yaab
print(yaab.BACKEND)   # "rust" or "python"
```
