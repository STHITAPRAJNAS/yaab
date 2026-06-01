# What agent-SDK users keep asking for — cross-framework demand study

**Date:** 2026-05-31 · **Method:** open issues sorted by reactions across the
eight leading agent SDKs, clustered into recurring themes and mapped to YAAB's
current status. Issue numbers are live references gathered from each repo's
issue tracker (sorted by reactions, open).

Frameworks surveyed: LangGraph/LangChain, Google ADK, Pydantic AI, DSPy, CrewAI,
AutoGen, OpenAI Agents SDK, AWS Strands.

## Headline finding

YAAB already ships a large fraction of the most-requested features across the
ecosystem — prompt management/optimization, agent handoffs, `MapAgent`,
streaming-through-the-tool-loop, output-validation retries, drift detection,
citations, batch, usage limits, and the whole governance layer. The genuine
gaps that map to high-demand requests are a short, concrete list (below).

## Demand themes (ranked), with YAAB status

Legend: ✅ shipped · ◑ partial · ✕ gap

| # | Theme | Who's asking (issue refs) | YAAB status |
|---|-------|---------------------------|-------------|
| 1 | **Full provider-kwarg passthrough + new model APIs** (Responses/Realtime, tool_choice, reasoning params) | ADK #773, #3209; CrewAI #4957; Pydantic #1447, #3365, #5044; AutoGen #3741 | ✅ `Agent(model_settings=…)` (new) + `tool_choice` + LiteLLM = 100+ providers |
| 2 | **Run lifecycle: cancel / interrupt / pause / resume / reset a running agent** | ADK #2425, #2853, #1621; OpenAI #798; Strands #1138, #329; LangGraph #5672 | ◑ `CancellationToken`, timeouts, graph HITL pause/resume + checkpoints; **no `Agent.reset()` / mid-run update** |
| 3 | **Prompt management / versioning / optimization** | Pydantic #921; Strands #609; CrewAI #5818, #5931 | ✅ `prompts.py` + `optimize/` (DSPy-style compile) |
| 4 | **Multi-agent handoffs / delegation / nested conversations** | OpenAI #847; Strands #911; Pydantic #1468; ADK #1828 (MapAgent) | ✅ Swarm handoffs, Sequential/Parallel/Loop/**MapAgent**, agent-as-tool |
| 5 | **Streaming ergonomics / stream through the tool loop** | Pydantic #1452; LangGraph #4653, #5672 | ✅ `stream_events` / `stream_run` (new), token-level + SSE |
| 6 | **MCP depth: resources / prompts / sampling** (beyond tools) | Pydantic #1558; Strands #151, #765; OpenAI #464; AutoGen #6995 | ◑ MCP **tools** client/server; **resources/prompts/sampling are a gap** |
| 7 | **Retries on validation / model errors** | DSPy #7693; OpenAI #325; Pydantic | ✅ output reflection/retry + `ResilientModel` |
| 8 | **Rate-limit handling (honor Retry-After)** | OpenAI #782; DSPy #1263 | ◑ rate limiter + circuit breaker; **doesn't yet honor `Retry-After`** |
| 9 | **Cost / token tracking incl. cached tokens** | AutoGen #4835 | ◑ cost+usage tracked; **cached-token accounting missing** |
| 10 | **Production session services (Firestore/Valkey/IAM)** | ADK #3776, #935; OpenAI #3017; AutoGen #5327 | ✅ Postgres/Aurora/Redis/SQLite sessions (Firestore/Valkey not built) |
| 11 | **Observability control (disable/extend internal tracing)** | ADK #2792; LangGraph #6214 | ✅ OTel + `Agent(instrument=False)` + audit sinks |
| 12 | **Provider built-in tools / "skills" (web search, code exec)** | Pydantic #3365, #5044; ADK #3611 | ◑ built-in tools + skills; **provider-native built-ins not surfaced** |
| 13 | **Structured citations** | Pydantic #3126 | ✅ RAG citations first-class |
| 14 | **Batch processing** | Pydantic #1771 | ✅ `batch.py` |
| 15 | **Configurable limits (cycles/tokens/runtime)** | Strands #191 | ✅ `UsageLimits` + `max_steps` + timeouts |
| 16 | **Behavioral drift detection across sessions** | CrewAI #5155 | ✅ `DriftMonitor` / `TrustScorer` |
| 17 | **Multimodal / video** | DSPy #8507 | ◑ multimodal `Content`; video untested |
| 18 | **Multi-language bindings (Go/Rust/TS)** | Strands #616, #1202; AutoGen #1700, #3858 | ✕ Rust core exists; no TS/Go bindings (Phase F) |
| 19 | **Python 3.14 support** | LangGraph #5253; CrewAI #5109 | ◑ 3.11–3.13; add 3.14 |
| 20 | **Security: no eval() on LLM output, supply-chain** | CrewAI #5056; DSPy #9500 | ✅ sandboxed code exec; guardrails; audit |

## The concrete gap list (high-demand, not yet covered)

In priority order, these map directly to repeated, high-reaction requests and are
achievable:

1. **Run lifecycle control (#2)** — `Agent.reset()`, mid-run cancel surfaced on
   the public API, and pause/resume for the *fast path* (not just graphs). This
   is the single most repeated cross-framework ask.
2. **MCP resources/prompts/sampling (#6)** — extend the MCP client beyond tools.
   (Phase D.)
3. **Rate-limit `Retry-After` (#8)** — honor the provider's retry header in
   `ResilientModel`.
4. **Cached-token accounting (#9)** — capture `cache_read_input_tokens` etc. into
   `Usage`.
5. **Provider-native built-in tools (#12)** — surface Anthropic/OpenAI server-side
   tools (web search, code execution) through the tool layer.
6. **Firestore/Valkey session backends (#10)** and **Python 3.14 (#19)** — small,
   frequently requested additions.

## What's conspicuously missing across the *whole* ecosystem (YAAB's opening)

- **Governance / audit / compliance as first-class** — nobody else ships a
  tamper-evident audit chain, evidence-gated lifecycle, or compliance mappers.
  YAAB's core differentiator; the surveyed repos don't even have issues for it
  because users don't expect it there.
- **One runtime spanning fast-path + durable graph + optimizable program** — each
  competitor owns one of these; users repeatedly bridge frameworks to get two.
- **Drift / trust scoring built in** (CrewAI #5155 is asking for what YAAB ships).

## Sources

Issue references above are from each project's GitHub issue tracker (open,
sorted by reactions), retrieved 2026-05-31:
langchain-ai/langgraph · google/adk-python · pydantic/pydantic-ai ·
stanfordnlp/dspy · crewAIInc/crewAI · microsoft/autogen ·
openai/openai-agents-python · strands-agents/sdk-python.
