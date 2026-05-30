# YAAB — Yet Another Agent Builder

**YAAB is the agent SDK for teams that have to ship agents into production AND prove it to a regulator.**
Type-safe like Pydantic AI, optimizable like DSPy, durable like LangGraph, clean
like Google ADK, simple like Strands — on a universal LiteLLM model layer, with a
Rust core that accelerates the compute-bound hot paths (and an opt-in native
graph engine). Governance, an agent registry, audit lineage, and policy
guardrails are **first-class, not bolted on**.

```python
from yaab import Agent

agent = Agent("assistant", model="openai/gpt-4o", instructions="Be concise.")
print(agent.run_sync("Say hello").output)
```

No API key? Everything runs offline with `TestModel`:

```python
from yaab import Agent
from yaab.testing import TestModel

agent = Agent("assistant", model=TestModel("hi!"))
assert agent.run_sync("hello").output == "hi!"
```

---

## Why YAAB

No single existing framework covers the regulated-enterprise agent use case.
Each excels at one layer and is weak elsewhere. YAAB fuses the best ideas from
each onto **one runtime** and adds the layer none of them ship: enterprise
governance.

| Capability | ADK | DSPy | Pydantic AI | Strands | LangGraph | **YAAB** |
|---|---|---|---|---|---|---|
| Type-safe `Agent[Deps, Output]` | ◑ | ✕ | ✓ | ◑ | ◑ | **✓** |
| Model-driven fast path | ◑ | ✕ | ✓ | ✓ | ✕ | **✓** |
| Durable graph + checkpoints + HITL | ◑ | ✕ | ◑ | ✕ | ✓ | **✓** |
| Optimizable programs (compile) | ✕ | ✓ | ✕ | ✕ | ✕ | **✓** |
| Universal models (LiteLLM) | ◑ | ◑ | ✓ | ✓ | ◑ | **✓** |
| MCP + A2A interop | ✓ | ✕ | ◑ | ◑ | ✕ | **✓** |
| `fastapi_server_app` / serve as A2A server | ✕ | ✕ | ✕ | ◑ | ✕ | **✓** |
| Pluggable auth (bearer / API key / OAuth2)| ◑ | ✕ | ✕ | ◑ | ✕ | **✓** |
| OTel GenAI-convention tracing | ✓ | ◑ | ✓ | ✓ | ◑ | **✓** |
| Prompt management + versioning | ✕ | ◑ | ✕ | ✕ | ✕ | **✓** |
| Built-in RAG (provider-neutral) | ◑ cloud | ◑ cloud | ✕ | ◑ cloud | ◑ | **✓** |
| RAG access control + citations + faithfulness | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** |
| Built-in tools + sandboxed code exec | ◑ | ✕ | ◑ | ◑ | ◑ | **✓** |
| Eval (RAGAS / DeepEval adapters) | ◑ | ◑ | ◑ | ✕ | ◑ | **✓** |
| AG-UI streaming + structured-output streaming | ✕ | ◑ | ◑ | ✕ | ◑ | **✓** |
| Cloud backends (Aurora/pgvector/OpenSearch/Oracle/Chroma/Qdrant/Pinecone/Weaviate/Redis) | ◑ | ✕ | ◑ | ◑ | ◑ | **✓** |
| **Agent registry + lifecycle FSM** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** |
| **Guardrail / policy engine** | ◑ | ✕ | ◑ | ✕ | ◑ | **✓** |
| **Tamper-evident audit + lineage** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** |
| **Compliance mappers (SR 11-7 / EU AI Act / NIST / ISO 42001 / SOC 2)** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** |
| **Rust performance core** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** |

✓ first-class · ◑ partial / via add-on · ✕ not provided.
See [`docs/COMPARISON.md`](docs/COMPARISON.md) for the full analysis and the
honest gap list.

---

## Architecture

YAAB is **progressive disclosure**: three lines to a working agent, but every
layer is openable.

**Python is the brain and the entire developer surface** — the `Agent`,
`Runner`, model layer, tools, governance, orchestration logic, and every
extension point are Python (~95% of the code). **Rust (`yaab-core`) is a small
performance core** (~325 lines) holding the compute-bound primitives where
native speed pays off: vector similarity, checkpoint serialization, channel
reducers, BSP superstep planning + whole-superstep state folding, and
tamper-evident audit hashing. Every Rust primitive has a **pure-Python
fallback**, so YAAB installs and runs anywhere — Rust is an accelerator, never a
hard dependency. This is the proven pydantic-core / Polars / Ruff pattern: a
thin native core under a rich Python API.

```
L5  Developer API (Python)   — Agent, tools, signatures, DI, skills
L4  Orchestration (Python)   — fast path · durable graph · optimizable programs
L3  Governance & Registry    — registry, lifecycle, policy, audit, evals, compliance
L2  Model Layer (LiteLLM)    — streaming, tools, structured output, fallbacks, cost
L1  yaab-core (RUST/PyO3)    — vectors · channels · checkpoints · scheduler · hashing
        ↑ accelerates the hot paths called by L1–L4; pure-Python fallback always present
```

What runs **where**, precisely:

| Concern | Runs in |
|---|---|
| Agent loop, tool dispatch, model calls (I/O-bound) | **Python** (network is the bottleneck, not the loop) |
| Governance, registry, lifecycle, compliance | **Python** |
| Graph control flow (routing, HITL, checkp-orchestration) | **Python** |
| Vector top-k, checkpoint (de)serialize, channel reduce, superstep fold, audit hash | **Rust** (Python fallback) |

Check which core is active:

```python
import yaab
print(yaab.BACKEND)   # "rust" or "python"
```

### Opt-in Rust graph engine

The durable graph lets you choose how each superstep's state is advanced — your
call, per compiled graph:

```python
app = graph.compile(engine="auto")     # rust if the extension is built, else python (default)
app = graph.compile(engine="rust")     # force the native whole-superstep fold (raises if unbuilt)
app = graph.compile(engine="python")   # force the pure-Python engine
print(app.engine)                       # "rust" | "python"
```

Both engines produce **identical results**; `rust` folds an entire superstep's
state in one native call instead of one cross-language hop per key. The Python
developer API is unchanged either way.

---

## The three orchestration paths (one runtime)

**1. Fast path — model-driven (Strands-style).** Just call the agent.

```python
from yaab import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"It's sunny in {city}."

agent = Agent("weather", model="openai/gpt-4o", tools=[get_weather])
print(agent.run_sync("What's the weather in Paris?").output)
```

**2. Graph path — durable (LangGraph-style)** with checkpoints and HITL.

```python
from yaab.graph import StateGraph, Channel, START, END, MemorySaver

g = StateGraph(channels={"count": Channel("add", default=0)})
g.add_node("inc", lambda s: {"count": 1})
g.add_edge(START, "inc")
g.add_conditional_edges("inc", lambda s: "inc" if s["count"] < 3 else END,
                        {"inc": "inc", END: END})

app = g.compile(checkpointer=MemorySaver())
print(app.invoke({}).state)   # {'count': 3}
```

Human-in-the-loop pauses and resumes by `thread_id`:

```python
def approve(state, ctx):
    decision = ctx.interrupt({"need": "approval"})   # pauses on first pass
    return {"approved": decision}

# first call returns interrupted=True; resume with the human's answer:
app.invoke({}, thread_id="t1")
app.invoke(thread_id="t1", resume=True)
```

**3. Optimizable path — compiled (DSPy-style).** Tune at build time, freeze for prod.

```python
from yaab.optimize import Predict, BootstrapFewShot
from yaab.governance.eval import Case

qa = Predict("question -> answer", model="openai/gpt-4o")
artifact = await BootstrapFewShot().compile(qa, trainset, metric)   # frozen, versioned
qa.load(artifact)   # deterministic in production
```

---

## Governance

Governance is opt-in by **mode** (`off` / `observe` / `enforcing`) so YAAB is
frictionless for prototyping but enforces registry, approval, and guardrails in
production.

```python
from yaab import Agent, Runner
from yaab.governance import (
    GovernanceService, GovernanceMode, AgentCard, RiskTier, LifecycleState,
    EvidenceArtifact,
)

gov = GovernanceService(mode=GovernanceMode.ENFORCING)

# Register the agent and walk it through the model-risk lifecycle.
gov.registry.register(AgentCard(agent_id="kyc-bot", name="KYC Bot", risk_tier=RiskTier.HIGH))
gov.lifecycle.transition("kyc-bot", LifecycleState.IN_DEVELOPMENT,
    evidence=[EvidenceArtifact(kind="development_docs"),
              EvidenceArtifact(kind="conceptual_soundness")])
# ... validation → approval ...

agent = Agent("KYC Bot", model="openai/gpt-4o", registry_id="kyc-bot")
runner = Runner(governance=gov)        # refuses unregistered/unapproved agents
```

You get, out of the box:

- **Agent registry** — A2A-compatible cards with ownership, risk tier, decision
  authority, data lineage, and approval status; produces the SR 11-7 / EU AI Act
  **model inventory**.
- **Lifecycle FSM** — `DRAFT → IN_DEVELOPMENT → IN_VALIDATION → APPROVED →
  DEPLOYED → MONITORED → DECOMMISSIONED`, each transition evidence-gated and audited.
- **Policy / guardrail engine** — prompt-injection, PII (redact), secret, topic,
  and system-prompt-leak scanners; pluggable (LLM Guard / NeMo / custom).
- **Tamper-evident audit log** — append-only, **hash-chained in Rust**; every
  run, model call, tool call, guard decision, and lifecycle change. `audit.verify()`
  detects any retroactive edit.
- **Evaluation** — code-first datasets/metrics that double as optimizer metrics
  and drift monitoring.
- **Compliance mappers** — project the data model onto **SR 11-7, EU AI Act,
  NIST AI RMF, ISO/IEC 42001, SOC 2** and emit audit-ready reports.

```bash
yaab compliance report sr_11_7
```

> Compliance mappers produce *evidence*, not legal sign-off. Effective challenge
> and conformity assessment still require qualified human reviewers — YAAB
> produces the evidence; humans attest to it.

---

## Serve anywhere — local to cloud

Local one-liner, or mount the ASGI app in any cloud (Lambda, Cloud Run, Fargate, K8s):

```python
from yaab import Agent
from yaab.serve import fastapi_server_app
from yaab.auth import BearerTokenAuth

agent = Agent("assistant", model="openai/gpt-4o")
app = fastapi_server_app(agent, auth=BearerTokenAuth({"secret-token": "alice"}))
# uvicorn module:app  →  exposes /run, /a2a/tasks, /.well-known/agent.json, /health
```

`yaab serve mymodule:agent` runs it directly. The app speaks **A2A** (agent
card + task endpoint), so other agents can discover and delegate to it. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and the [`Dockerfile`](Dockerfile).

---

## Multi-agent & managers (ADK-style)

Compose agents as Sequential / Parallel / Loop / Swarm workflows — each is itself
an agent, so they nest and drop into tools, graphs, and servers:

```python
from yaab import SequentialAgent, ParallelAgent, Swarm
from yaab.multiagent import SwarmState

pipeline = SequentialAgent("etl", [extractor, transformer, loader])
board    = ParallelAgent("review", [legal, finance, risk])
support  = Swarm("support", [triage, billing, tech], entry="triage")
await support.run("I was double charged", deps=SwarmState())
```

Scoped session / memory / artifact **managers** mirror ADK's services:

```python
from yaab import SessionManager, MemoryManager, ArtifactManager

sessions = SessionManager()
s = await sessions.create_session(app_name="bank", user_id="alice", state={"tier": "gold"})

memory = MemoryManager()
await memory.add("Alice prefers email", app_name="bank", user_id="alice")
await memory.add_session_to_memory(s, app_name="bank", user_id="alice")

artifacts = ArtifactManager()
await artifacts.save("report.pdf", data, session_id=s.id)   # auto-versioned
```

Interop is first-class: import an **MCP** server's tools with `MCPClient` (or
expose yours with `MCPServer`), and discover/delegate to remote agents over
**A2A** with `RemoteAgent` (which is also a tool). See the docs below.

## Built-in RAG (provider-neutral)

A whole retrieval pipeline ships in the box — not delegated to a cloud service —
with the governance pieces other SDKs lack:

```python
from yaab import Agent, KnowledgeBase
from yaab.rag import load_directory

kb = KnowledgeBase()                       # default: in-memory + Rust top-k
kb.add(load_directory("./docs", glob="**/*.md"))   # pdf/html/csv/json loaders too
agent = Agent("support", model="openai/gpt-4o", tools=[kb.as_tool()])
```

Per-user/document access control, source citations, dedup/incremental indexing,
retrieval guardrails (context-poisoning defense), and faithfulness eval are all
first-class. Swap the store for a cloud backend with one line — see below.

## Batteries included

Everything below is built in, extensible by `Protocol`, and selectable by name
through the component registry — each integration is an optional extra:

- **Built-in tools** — calculator, time, HTTP fetch, web search, and sandboxed
  Python exec (subprocess by default; `DockerSandbox` for real isolation).
- **Cloud backends** — sessions on Postgres/**Aurora**/Redis; vector stores on
  **pgvector/Aurora**, **OpenSearch**, **Oracle 23ai**, Chroma, Qdrant,
  Pinecone, Weaviate; graph checkpointers on Postgres/Aurora/Redis. All ship an
  in-memory default and honor metadata filters for per-tenant isolation.
- **Evaluation** — deterministic + LLM-judge metrics, plus **RAGAS** and
  **DeepEval** adapters, behind one registry.
- **Frontends & ops** — `yaab web` dev playground, **AG-UI** streaming
  middleware, token + structured-output streaming, batch/offline inference,
  resilience (rate limit + circuit breaker), YAML-config agents, OTel tracing
  with Langfuse/Logfire/OTel audit sinks.

```python
from yaab import get_component, available_components
available_components("vectorstore")   # memory, pgvector, aurora, chroma, qdrant, opensearch, oracle, pinecone, weaviate
store = get_component("vectorstore", "opensearch", index="kb", hosts=[...])
```

See [Storage & backends](docs/storage-backends.md) and
[Extending YAAB](docs/extending.md).

## Documentation

Full guides live in [`docs/`](docs/index.md):

[**Get started**](docs/get-started.md) ·
[Concepts](docs/concepts.md) ·
[Samples](samples/README.md) ·
[Quickstart](docs/quickstart.md) ·
[Agents](docs/agents.md) ·
[Tools](docs/tools.md) ·
[Models](docs/models.md) ·
[State (sessions/memory/artifacts)](docs/state.md) ·
[Storage & backends](docs/storage-backends.md) ·
[RAG](docs/rag.md) ·
[Multi-agent](docs/multi-agent.md) ·
[Streaming & events](docs/streaming-events.md) ·
[Graph](docs/graph.md) ·
[MCP & A2A](docs/interop.md) ·
[Governance](docs/governance.md) ·
[Evaluation](docs/evaluation.md) ·
[Optimization](docs/optimization.md) ·
[Prompts & skills](docs/prompts-skills.md) ·
[Serving & auth](docs/serving.md) ·
[Platform extensions](docs/platform.md) ·
[Extending](docs/extending.md) ·
[Deployment](docs/DEPLOYMENT.md) ·
[Comparison & gaps](docs/COMPARISON.md)

## Install

```bash
pip install yaab                 # SDK + high-performance async-first Python core
pip install 'yaab[rust]'         # + the prebuilt Rust performance core (yaab-core)
pip install 'yaab[litellm]'      # universal model layer
pip install 'yaab[all]'          # everything (rust, litellm, otel, rag, serve, …)
```

**Two cores, one API.** `pip install yaab` ships a high-performance,
async-first **pure-Python core** that works on every platform with zero build
tooling. `pip install 'yaab[rust]'` adds **`yaab-core`** — a prebuilt
`abi3` wheel (one wheel for CPython 3.11+, including future versions) that
transparently accelerates the hot paths (vector search, checkpoint
serialization, channel reducers, audit hashing, the graph engine). YAAB
auto-selects Rust when present and falls back to Python otherwise — your code
never changes. Check which is active:

```python
import yaab; print(yaab.BACKEND)   # "rust" or "python"
```

Building the Rust core from source (for development) needs only `maturin`:

```bash
maturin develop -m yaab-core/Cargo.toml --release
```

---

## CLI

```bash
yaab info                         # environment + active performance backend
yaab init my_agent                # scaffold a starter agent
yaab registry list                # the model inventory
yaab compliance report eu_ai_act  # audit-ready compliance report
yaab serve my_module:agent        # serve over HTTP / A2A
```

---

## Project layout

```
yaab/            Python SDK (thin API layer)
  agent.py runner.py types.py        core abstractions + event-driven runner
  models/                            LiteLLM provider, instrumentation, TestModel
  tools/ sessions/ memory/ artifacts/ services
  graph/                             durable StateGraph + checkpointers
  optimize/                          Signature / Module / Optimizer (DSPy-style)
  governance/                        registry, lifecycle, policy, audit, eval, compliance
  plugins/ prompts.py skills.py auth.py serve.py cli.py
yaab-core/       Rust crate (PyO3) — the performance core
examples/        runnable examples
tests/           test suite (runs offline)
```

---

## Status & caveats

YAAB is **alpha**. The core runtime, governance layer, graph engine, and Rust
core are implemented and tested offline; some borrowed capabilities are
simplified relative to their source frameworks (noted in `docs/COMPARISON.md`).
Framework APIs evolve fast — pin versions and re-verify. Verify EU AI Act
dates/fines against EUR-Lex and SR 11-7 language against the Federal Reserve's
official letter.

MIT licensed.
