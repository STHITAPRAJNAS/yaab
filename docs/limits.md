# Usage limits & run control

Bound and interrupt runs with two composable controls. Both are optional — with
neither set, runs behave exactly as before.

## UsageLimits

Hard caps enforced by the Runner between steps and before each tool call.
Breaching one raises `UsageLimitExceeded` (with `.limit` naming which cap).

```python
from yaab import Agent, UsageLimits

limits = UsageLimits(
    max_requests=8,            # model calls
    max_input_tokens=20_000,
    max_output_tokens=4_000,
    max_total_tokens=24_000,
    max_tool_calls=12,          # across all tools
    per_tool_calls={"charge": 1, "send_email": 1},  # per-tool caps
    max_wall_seconds=30,
)

result = await agent.run("do the thing", usage_limits=limits)
```

Per-tool caps are the key safety valve for side-effecting tools — e.g. allow at
most one `charge` call per run regardless of what the model attempts.

## Cancellation & timeout

`CancellationToken` is a cooperative stop signal the Runner checks between
supersteps and before each tool call. Cancel it from a signal handler, another
task, or an API endpoint; the run stops with `RunCancelled`.

```python
from yaab import Agent, CancellationToken

token = CancellationToken()

# ... elsewhere (e.g. a /cancel endpoint, a UI button, a watchdog):
token.cancel("user_stop")

result = await agent.run("long task", cancellation=token)   # raises RunCancelled
```

A `timeout` (seconds) wires an automatic deadline onto the same mechanism:

```python
await agent.run("long task", timeout=30)   # RunCancelled(reason="timeout") at the deadline
```

Catching them:

```python
from yaab import RunCancelled, UsageLimitExceeded

try:
    await agent.run(prompt, usage_limits=limits, timeout=30)
except UsageLimitExceeded as e:
    print("hit cap:", e.limit)
except RunCancelled as e:
    print("stopped:", e.reason)   # "timeout" | "cancelled" | custom
```

Both controls also work on `Runner.run` / `Runner.run_stream` and on
`agent.run_sync(...)`.
