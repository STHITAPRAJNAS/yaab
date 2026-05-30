# YAAB vs. ADK, DSPy, Pydantic AI, Strands, LangGraph — features & gaps

This is the honest scorecard: what YAAB does today, how it stacks up against the
incumbents, and exactly what is still missing on the road to "better than all
existing frameworks combined." It is meant to be read alongside the code — every
"✓ implemented" row points at a module you can open.

## 1. Capability matrix

Legend: **✓** first-class · **◑** partial / simplified · **✕** absent.

| Capability | ADK | DSPy | Pydantic AI | Strands | LangGraph | **YAAB today** | YAAB module |
|---|---|---|---|---|---|---|---|
| Type-safe `Agent[Deps, Output]` | ◑ | ✕ | ✓ | ◑ | ◑ | **✓** | `agent.py`, `types.py` |
| DI via run context | ◑ | ✕ | ✓ | ✕ | ◑ | **✓** | `types.RunContext` |
| Structured output + validation/retry | ✓ | ◑ | ✓ | ◑ | ◑ | **✓** | `runner.py` |
| Typed function tools | ✓ | ◑ | ✓ | ✓ | ◑ | **✓** | `tools/base.py` |
| Model-driven fast path (ReAct loop) | ◑ | ◑ | ✓ | ✓ | ✕ | **✓** | `runner.py` |
| Durable graph orchestration | ◑ | ✕ | ◑ | ✕ | ✓ | **✓** | `graph/` |
| Checkpointing / crash recovery | ◑ | ✕ | ◑ | ✕ | ✓ | **✓** | `graph/checkpoint.py` |
| Human-in-the-loop (interrupt/resume) | ✓ | ✕ | ✓ | ◑ | ✓ | **✓** | `graph/state.py` |
| Time-travel debugging | ✕ | ✕ | ✕ | ✕ | ✓ | **◑** | `checkpoint.history()` |
| Channel reducers (BSP supersteps) | ✕ | ✕ | ✕ | ✕ | ✓ | **✓** | `_core` + `graph` |
| Optimizable programs (compile) | ✕ | ✓ | ✕ | ✕ | ✕ | **◑** | `optimize/` |
| MIPROv2 / GEPA optimizers | ✕ | ✓ | ✕ | ✕ | ✕ | **◑** (simplified) | `optimize/optimizer.py` |
| Universal model layer (LiteLLM) | ◑ | ◑ | ✓ | ✓ | ◑ | **✓** | `models/litellm_provider.py` |
| Fallbacks / retries / cost tracking | ◑ | ✕ | ◑ | ◑ | ◑ | **✓** | `models/litellm_provider.py` |
| Multi-agent (agent-as-tool) | ✓ | ✕ | ◑ | ✓ | ✓ | **✓** | `tools/agent_tool.py` |
| Multi-agent (Sequential/Parallel/Loop/Swarm) | ✓ | ✕ | ◑ | ✓ | ✓ | **✓** | `multiagent.py` |
| MCP (tools + client) | ✓ | ✕ | ◑ | ✓ | ✕ | **✓** (stdio+transport) | `tools/mcp_client.py` |
| A2A (server + cards + outbound client) | ✓ | ✕ | ◑ | ◑ | ✕ | **✓** | `serve.py`, `a2a/client.py` |
| `fastapi_server_app` / serve | ✕ | ✕ | ✕ | ◑ | ✕ | **✓** | `serve.py` |
| Pluggable auth (bearer/API key/OAuth2) | ◑ | ✕ | ✕ | ◑ | ✕ | **✓** | `auth.py` |
| Event-driven streamed run | ✓ | ✕ | ✓ | ✓ | ✓ | **✓** | `runner.run_stream` |
| Token-level streaming + SSE | ✓ | ✕ | ✓ | ✓ | ◑ | **✓** | `agent.stream`, `serve.py` |
| Session/Memory/Artifact managers | ✓ | ✕ | ◑ | ✓ | ✓ | **✓** | `*/manager.py` |
| Component registry + entry-point extensibility | ◑ | ◑ | ◑ | ◑ | ◑ | **✓** | `extensions.py` |
| OTel GenAI-convention tracing | ✓ | ◑ | ✓ | ✓ | ◑ | **✓** | `observability/`, `models/instrumented.py` |
| Sessions (KV state + history) | ✓ | ✕ | ◑ | ✓ | ✓ | **✓** | `sessions/` |
| Long-term vector memory | ✓ | ✕ | ✕ | ◑ | ✓ | **✓** | `memory/` |
| Artifact storage | ✓ | ✕ | ✕ | ✕ | ✕ | **✓** | `artifacts/` |
| Plugin / callback system | ✓ | ✕ | ◑ | ◑ | ◑ | **✓** | `plugins/` |
| Prompt management + versioning | ✕ | ◑ | ✕ | ✕ | ✕ | **✓** | `prompts.py` |
| Skills (reusable bundles) | ◑ | ✕ | ◑ | ✕ | ✕ | **✓** | `skills.py` |
| Code-first evals | ✓ | ✓ | ✓ | ◑ | ◑ | **✓** | `governance/eval.py` |
| **Agent registry + model inventory** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** | `governance/registry.py` |
| **Lifecycle FSM (model risk)** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** | `governance/lifecycle.py` |
| **Guardrail / policy engine** | ◑ | ✕ | ◑ | ✕ | ◑ | **✓** | `governance/policy.py` |
| **Tamper-evident audit + lineage** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** | `governance/audit.py` |
| **Compliance mappers (5 regimes)** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** | `governance/compliance/` |
| **Rust performance core** | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** | `yaab-core/` |
| Governance modes (off/observe/enforce) | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** | `governance/service.py` |

The bottom block — registry, lifecycle, audit, compliance, Rust core — is where
YAAB is categorically ahead: **no incumbent ships any of it.**

## 2. Where YAAB already wins

- **One runtime, three paradigms.** Fast path (Strands), durable graph
  (LangGraph), and optimizable programs (DSPy) compose over the same Runner and
  Rust scheduler. No competitor spans all three.
- **Governance is a runtime concern, not a doc.** Registry gate + guardrails +
  hash-chained audit run *inside* the loop, toggled by mode. This is the reason a
  regulated bank adopts YAAB.
- **Rust does the heavy lifting.** Scheduling, checkpoint serialization, channel
  reducers, vector similarity, and audit hashing live in `yaab-core`; the Python
  layer is a thin, friendly wrapper. A pure-Python fallback keeps it installable
  anywhere (`YAAB_NO_RUST=1` exercises it in CI).
- **Serve + interop out of the box.** `fastapi_server_app` exposes native, A2A, and
  discovery endpoints with pluggable auth — a single function from local to cloud.

## 3. Honest gaps (the road to "better than all combined")

Several gaps from the first cut are now **closed** (Sequential/Parallel/Loop/Swarm
multi-agent, an MCP client, an outbound A2A client, token-level streaming + SSE,
real LiteLLM embeddings, ADK-style managers, and a component registry). The
remaining items are scoped simplifications or not-yet-built pieces, each tracked
toward parity-or-better.

1. **Optimizers are still simplified.** `MIPROv2` now searches instructions ×
   bootstrapped demo sets and `GEPA` reflectively evolves instructions, but
   neither is the full Bayesian / genetic-Pareto search of DSPy. *Plan:* port the
   real search loops; `governance.eval` already supplies the metric source.
2. **MCP transport coverage.** The client speaks JSON-RPC over stdio and any
   injectable transport (HTTP/SSE via a small adapter); a native streamable-HTTP
   transport and an MCP *server* (exposing YAAB tools to others) are next.
3. **A2A client depth.** `RemoteAgent` discovers cards and submits tasks; task
   *streaming/polling* for long-running remote tasks and full OAuth2 token
   exchange are still to come.
4. **Streaming through the tool loop.** Token streaming works for the answering
   turn and SSE carries both token and semantic streams; streaming *interleaved
   with* tool calls, and LangGraph's full set of stream modes, are partial.
5. **Rust core is minimal by design.** It accelerates the proven hot paths; the
   full Tokio actor message bus / structured-concurrency sub-agent lifecycle
   described in the architecture is scaffolded in Python today. *Plan:* move the
   event bus into Rust once profiling shows it dominates.
6. **No `yaab web` dev UI yet**, and deployment recipes are docs + a Dockerfile
   rather than turnkey templates for every cloud.
7. **Embeddings need a provider.** `LiteLLMEmbedder` ships (OpenAI/Cohere/Bedrock/…)
   and is registered in the component registry; the *default* embedder is still a
   deterministic hashing stub so offline/test runs need no keys.
8. **TypeScript binding** over the Rust core (Strands precedent) is designed-for
   (the crate is `rlib` + `cdylib`) but not built.
9. **Compliance mappers are evidence generators, not legal sign-off** — by
   design. Effective challenge and conformity assessment stay human.

## 4. Net assessment

For the **regulated-enterprise agent** use case, YAAB is already differentiated
beyond any single framework today, because governance/registry/audit/compliance
and a Rust core simply do not exist elsewhere. For **raw orchestration breadth**,
LangGraph still leads on streaming-mode richness and battle-tested scale, DSPy
leads on optimizer depth, and ADK leads on the maturity of its tool/eval
ecosystem and managed deployment. Closing gaps 1–6 above is what turns "uniquely
valuable to a bank" into "the default for everyone."
