# Refund pipeline

A durable business workflow with a human gate. A `Flow` branches on the refund
amount: small refunds auto-approve; large ones **pause for a human** and resume —
on a fresh runner call (even a different process) — once a reviewer decides.

```bash
python -m cookbook.apps.refund_pipeline
```

Output (offline):

```python
{'small': 'auto-refunded $40',
 'large_after_approval': 'refunded $500',
 'paused_kind': 'flow_pause'}
```

## What it shows

| Capability | Where |
|---|---|
| **Flow** | a `Flow` that parses the amount and branches (`.then`) |
| **HITL pause → decide → resume** | a $500 refund calls `ctx.pause_for(...)`; the run suspends, a reviewer decides with `approvals.respond(...)`, and `runner.run(flow, resume=decision)` finishes it |
| **Durable** | the pause is a checkpoint (`MemorySaver`) + an `ApprovalStore` row, so the run survives a restart and resumes on any worker |
| **One idiom** | the *same* `approvals.respond` verb used for tool approvals also resumes a Flow pause |

Swap `MemorySaver()` for `SQLiteSaver`/`PostgresSaver` and the in-memory approval
store for a durable one, and the same flow resumes across pods.
