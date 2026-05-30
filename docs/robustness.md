# Robustness — built-in tools, context, approval, resilience, config

The features that make YAAB usable out of the box without assembling them
yourself.

## Built-in tool library

A starter toolbox so a new agent is useful immediately:

```python
from yaab import Agent
from yaab.tools.builtin import calculator, current_time, http_get, web_search, python_exec, default_toolset

agent = Agent("a", model="openai/gpt-4o", tools=default_toolset())   # safe read-only set
```

| Tool | What it does | Notes |
|---|---|---|
| `calculator` | arithmetic | AST-parsed, never `eval` |
| `current_time` | ISO date/time | UTC, optional tz offset |
| `http_get` | fetch a URL's text | http/https only; lazy `httpx` |
| `web_search` | web results | configure a provider via `set_search_provider` |
| `python_exec` | run a snippet | isolated subprocess + timeout; gate behind approval |

`default_toolset()` excludes `python_exec` (code execution) — opt in explicitly.

## Context-window management

Stop long conversations from silently overflowing the model window:

```python
from yaab import Agent, TruncateMessages, SummarizeHistory

# Keep system + the most recent N messages:
Agent("a", model="openai/gpt-4o", context_strategy=TruncateMessages(max_messages=20))

# Summarize old history into a running summary once a token budget is hit:
Agent("a", model="openai/gpt-4o", context_strategy=SummarizeHistory(max_tokens=6000, keep_recent=6))
```

Strategies run before each model call. `SummarizeHistory` falls back to
truncation if no model is available. Implement `ContextStrategy` for custom
policies (token-aware, semantic compaction).

## Human-in-the-loop tool approval (fast path)

The graph engine pauses with `interrupt()`; for the model-driven loop, require a
human's sign-off on sensitive tools via a plugin:

```python
from yaab import Runner
from yaab.governance import ToolApprovalPlugin

# Inline: await an approver before the tool runs (CLI, Slack bot, queue):
async def approve(tool, args, ctx):
    return args.get("amount", 0) < 10_000

runner = Runner(plugins=[ToolApprovalPlugin(tools=["wire_transfer"], approver=approve, audit=gov.audit)])
```

A rejection short-circuits the tool with a message the model can adapt to. With
**no approver**, a guarded tool raises `ApprovalRequired` (carrying the pending
call) so an out-of-band flow can approve and re-run. Guard by tool name or a
`needs_approval(tool, args, ctx)` predicate. Every decision is audited.

## Resilience: rate limiting & circuit breaking

Retries/fallbacks live in the model layer; add protection for a failing or
rate-limited provider:

```python
from yaab.models.resilient import ResilientModel, RateLimiter, CircuitBreaker

model = ResilientModel(
    "openai/gpt-4o",  # or any ModelProvider
    rate_limiter=RateLimiter(rate=500, per=60),          # token bucket
    circuit_breaker=CircuitBreaker(threshold=5, cooldown=30),
)
agent = Agent("a", model=model)
```

The circuit breaker opens after consecutive failures, fails fast for the
cooldown, then half-opens to probe recovery.

## Declarative agents (YAML)

Define an agent as data — an auditable artifact ops and non-coders can review:

```yaml
# support_bot.yaml
name: support-bot
model: openai/gpt-4o
instructions: You are a helpful support agent.
registry_id: support-bot
max_steps: 8
tools: [calculator, current_time, http_get]
```

```python
from yaab import agent_from_yaml, agent_from_dict

agent = agent_from_yaml("support_bot.yaml")     # path or YAML string
agent = agent_from_dict({...})                  # no yaml dependency needed
```

Unknown tool/skill names fail loudly so typos surface immediately.
