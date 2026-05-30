# Roadmap — informed by what the ecosystem is actually requesting

This roadmap is grounded in **open feature requests across the major agent
frameworks** (Google ADK, OpenAI Agents SDK, LangGraph, Microsoft AutoGen /
Agent Framework, Pydantic AI, DSPy, AWS Strands, CrewAI), sorted by 👍 reactions
and comment volume. We extracted the recurring asks, then mapped each to YAAB's
current state so we build what users demonstrably want — and double down where
YAAB is already ahead.

Legend: ✅ **Have** · 🟡 **Partial** · ⬜ **Gap (planned)**

## 1. The demand, ranked by cross-framework signal

| # | What users are asking for | Who's asking (issues) | YAAB today |
|---|---|---|---|
| 1 | **Run lifecycle control** — cancel / interrupt / pause / resume in-flight runs; don't re-run interrupted nodes; reset state | ADK #2425/#2853/#1621, OpenAI #798, Strands #329/#1138, LangGraph #5672/#6208 | 🟡 graph has interrupt/resume + checkpoints; **fast-path Runner has no cancel/timeout** |
| 2 | **Tool-call authorization & guardrails** — authorize a tool *before* it runs; governance middleware; sanitize memory/prompt injection | CrewAI #4877 (56 comments — biggest single ask), #5888, #5556, #5057; Strands trace redaction #1292 | ✅ PolicyEngine + `before_tool` hook + modes — **core strength** |
| 3 | **Execution limits & budgeting** — caps on steps, tokens, wall-clock, **per-tool** usage; graceful fallback | Strands #191/#1503, Pydantic AI #3352/#2605 | 🟡 `max_steps` + `CostBudgetPlugin`; no unified `UsageLimits` |
| 4 | **Robust tool-arg validation & retries** — repair/pre-process malformed tool-call args; retry on model-behavior errors | Pydantic AI #3008, OpenAI #325, DSPy #7693 | 🟡 output reflection/retry + `ToolError` feedback; no arg-repair hook |
| 5 | **Tool idempotency on retry** — don't double-charge/email/trade when a step retries | CrewAI #5802 (40 comments) | ⬜ planned (governance-aligned) |
| 6 | **Production session/memory backends** — Postgres / Redis / Firestore / Valkey, pagination, schema migrations | ADK #3776/#2524/#4621/#3343, OpenAI #3017, LangGraph #3716 (top 👍), CrewAI memory | 🟡 SQLite + in-memory + protocols; no Postgres/Redis yet |
| 7 | **Prompt management, versioning & optimization**; inspect the *compiled* prompt | Pydantic AI #921 (21 👍), Strands #609, CrewAI #5818, DSPy #7830 | ✅ `PromptRegistry` + DSPy-style optimizers — **strength**; add prompt-inspect |
| 8 | **Multi-language SDKs** — Go / Rust / Java / TypeScript | Strands #616 (77 👍, #1 ask), #1202, AutoGen #1700/#1045 | 🟡 Rust core ships; `rlib` ready for bindings; no TS/Go SDK yet |
| 9 | **Configurable tracing + PII redaction in traces** | ADK #2792, OpenAI #1844/#2393, Strands #1059/#1292 | 🟡 per-agent `instrument` toggle; no global switch or trace redaction |
| 10 | **`tool_choice` (auto/required/none) & force tool use before answering** | ADK #773, Pydantic AI #1820 | ⬜ planned |
| 11 | **Reasoning-model support** — capture/stream `<thinking>` traces | DSPy #7813, Agent Framework #5538, OpenAI #825 | 🟡 `Part.thought` type exists; not captured from providers yet |
| 12 | **MCP beyond tools** — prompts, resources, sampling; native HTTP transport; MCP *server* | Strands #151 (20 👍)/#765, DSPy #7799, OpenAI #3477 | 🟡 MCP client for tools (stdio + transport); rest planned |
| 13 | **A2A protocol depth** — A2A 1.0, streaming/long-running tasks, back-handoff | OpenAI #472 (top, 33 👍), #847, ADK #5056, Strands #911 | 🟡 A2A server + cards + outbound client; deepen to spec |
| 14 | **Parallel/Map fan-out + conditional early-stop in workflows** | ADK #1828 (MapAgent), #1947/#3405, DSPy #8947 | 🟡 `ParallelAgent`; add `MapAgent` + early-stop |
| 15 | **Async everywhere + parallel tool calls** | DSPy #1975/#8947, Pydantic AI #1771 (batch) | ✅ async-first; parallel tool calls within a turn planned |
| 16 | **Realtime / voice API** | Pydantic AI #1447 (22 👍), Agent Framework #728 | ⬜ out of scope near-term |
| 17 | **Behavioral-drift detection / trust scoring** | CrewAI #5155/#5789 | 🟡 evals + audit are the substrate; no drift monitor yet |
| 18 | **Pydantic models as tool params/returns** | ADK #1066 | ✅ supported via type hints (document + test) |

## 2. The signal is loud where YAAB is strong

The single most-discussed request in the entire scan — CrewAI #4877
**"pre-tool-call authorization"** (56 comments) — plus governance middleware,
prompt-injection sanitization, trace PII redaction, idempotency, and
audit/trust-scoring are **exactly YAAB's thesis**. The same governance demand is
surfacing piecemeal across every framework; YAAB already ships the registry,
guardrail engine, hash-chained audit, and enforcing mode that these issues ask
for. The market is validating the bet.

Likewise, **multi-language SDKs** (Strands' #1 ask, 77 👍) and **durable
checkpointing** (LangGraph's top-👍 issue) play directly to YAAB's Rust core and
checkpointer design.

## 3. What we'll incorporate, prioritized

> **Delivery status (feature/tiered-enhancements):** Tier 1 ✅ · Tier 2 ✅ ·
> Tier 3 ✅ except the TypeScript/Go binding (item 14), which is tracked as a
> separate effort needing a JS/Go toolchain. Each item below links to the
> upstream request; see git history for the commit that implemented it.

**Tier 1 — high demand × strong fit × low effort — ✅ DONE:**

1. **`UsageLimits`** — unified caps (requests, input/output tokens, wall-clock,
   per-tool call counts) enforced in the Runner, with a graceful-fallback hook.
   *(Covers ADK/Strands/Pydantic AI #191/#1503/#3352.)*
2. **Run cancellation & timeout** — a `CancellationToken` + `timeout` on
   `Runner.run`/`run_stream`, checked between steps and tool calls.
   *(OpenAI #798, ADK #2425, Strands #1138.)*
3. **`tool_choice`** on `Agent`/model layer (`auto` / `required` / `none` /
   force-a-named-tool). *(ADK #773, Pydantic AI #1820.)*
4. **Tool-arg repair hook + model-behavior retry policy** — pre-validate/coerce
   raw tool-call args and retry malformed model output with feedback.
   *(Pydantic AI #3008, OpenAI #325, DSPy #7693.)*
5. **Capture & stream reasoning traces** into `Part.thought` (we already have the
   type). *(DSPy #7813, OpenAI #825.)*
6. **Inspect the compiled prompt** — expose the exact rendered prompt/demos from
   an optimized `Module`. *(DSPy #7830.)*

**Tier 2 — high demand, more effort — ✅ DONE:**

7. **Tool idempotency keys** — dedupe side-effecting tool calls across retries.
   *(CrewAI #5802 — governance-aligned.)*
8. **Explicit pre-tool authorization API** — a first-class `authorize(tool, args,
   ctx) -> Decision` seam alongside guardrails. *(CrewAI #4877/#5888.)*
9. **Postgres + Redis backends** for sessions / checkpoints / registry / audit,
   with `list_*` pagination. *(ADK #2524/#4621, OpenAI #3017, LangGraph #3716.)*
10. **Trace PII redaction** + a global tracing on/off switch. *(Strands
    #1059/#1292, OpenAI #2393.)*
11. **`MapAgent`** (fan-out one sub-agent N times) and **early-stop** conditions
    for workflow agents. *(ADK #1828/#3405.)*

**Tier 3 — strategic, larger — ✅ DONE (except 14):**

12. ✅ **MCP resources/prompts + an MCP *server*** exposing YAAB tools, plus
    client resources/read & prompts/get. *(Strands #151/#765, DSPy #7799.)*
13. ✅ **A2A depth** — streaming/long-running tasks (poll + SSE), OAuth2 token
    provider, back-to-orchestrator handoff. *(OpenAI #472/#847, ADK #5056.)*
14. ⬜ **TypeScript binding** over `yaab-core` (then Go) — needs a JS/Go
    toolchain; tracked separately. *(Strands #616, AutoGen #1700.)*
15. ✅ **Drift / trust-scoring monitor** built on evals + audit. *(CrewAI
    #5155/#5789.)*

## 4. Non-goals (for now)

Realtime/voice (Pydantic AI #1447, Agent Framework #728) and a visual workflow
designer (LangGraph Studio, Agent Framework DevUI) are valuable but orthogonal to
YAAB's governance-first, Rust-accelerated focus; revisit after Tier 1–2.

---

*Sources: open GitHub issues for each framework, queried via the GitHub API and
sorted by reactions/comments at research time. Issue numbers are pointers, not
guarantees of current status — re-verify before implementation.*
