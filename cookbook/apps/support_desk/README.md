# Support desk

A grounded, governed, safe-to-act customer-support agent — the shape most
"talk to your docs + act" assistants take, with the guardrails a real deployment
needs.

```bash
python -m cookbook.apps.support_desk                      # offline, deterministic
YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash python -m cookbook.apps.support_desk  # live
```

Output (offline):

```python
{'grounded': 'refunds.md',
 'refund_executed': False,
 'tenant_blocked_over_budget': True,
 'answer': 'Your refund request has been recorded.'}
```

## What it shows

| Capability | Where |
|---|---|
| **RAG** | answers grounded in a help-center `KnowledgeBase` (`refunds.md`, `shipping.md`) |
| **Tools** | an `order_status` lookup and a side-effecting `issue_refund` |
| **HITL approval** | `ToolApprovalPlugin` gates `issue_refund`; a **$500 refund is denied and never executes** (`refund_executed: False`) |
| **Spend governance** | `SpendGovernancePlugin` with a per-tenant daily cap; once the `acme` tenant is over budget, the **next request is blocked** (`tenant_blocked_over_budget: True`) |

## Serve it behind the OpenAI API

The same agent is servable for any OpenAI-SDK client:

```python
from yaab import openai_compat_app
from cookbook.apps.support_desk import build

agent, _spend = build()
app = openai_compat_app({"support": agent})   # POST /v1/chat/completions
```
