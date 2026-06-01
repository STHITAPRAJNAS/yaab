# scripts/

Validation harnesses for the SDK.

## `smoke_all.py` — full feature smoke test (offline, no key)

Exercises every major feature end to end and prints a PASS/FAIL report. Runs
fully offline on deterministic models, so it works anywhere (including
restricted CI):

```bash
python scripts/smoke_all.py
```

Covers: LLM fast-path run · tool-calling loop · structured output · token
streaming · semantic event stream · structured-output streaming · multi-agent
(sequential/parallel) · **swarm hand-off** · graph + HITL + cycles · RAG
retrieve/as-tool · RAG per-user access control · **A2A server+client** ·
**MCP client↔server** · governance lifecycle + audit · guardrails · HITL tool
approval · eval metrics + RAGAS/DeepEval adapters · resilience · usage limits ·
optimizer compile · cloud-backend registration.

Current status: **21/21 pass** (Rust backend; also passes on `YAAB_NO_RUST=1`).

## `live_llm_check.py` — real LLM integration (needs a key + network)

Makes real API calls to verify the LiteLLM integration, real token streaming,
tool calling, and structured output against a live provider. Run it where the
provider is reachable (your laptop/CI) — **not** inside a restricted sandbox.

```bash
pip install 'yaab-sdk[litellm]'

# Groq free tier — https://console.groq.com
export GROQ_API_KEY=...                       # keep secret; never commit
export YAAB_LIVE_MODEL=groq/llama-3.3-70b-versatile
python scripts/live_llm_check.py

# or Gemini free tier
export GEMINI_API_KEY=...
export YAAB_LIVE_MODEL=gemini/gemini-2.0-flash

# or local Ollama (no key)
export YAAB_LIVE_MODEL=ollama/llama3
```

Checks: basic completion · real token streaming · tool calling (model chooses
the tool) · validated structured output · a multi-agent pipeline.
