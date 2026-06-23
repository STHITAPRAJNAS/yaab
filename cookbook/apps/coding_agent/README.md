# Coding agent

A planner → coder pipeline that writes and runs Python in a **sandbox**, gated by
**tool approval** so risky code can be reviewed before it executes.

```bash
python -m cookbook.apps.coding_agent
```

Output (offline):

```python
{'answer': 'The sum of 0..10 is 55.', 'ran_code': 'print(sum(range(11)))'}
```

## What it shows

| Capability | Where |
|---|---|
| **Multi-agent** | a `SequentialAgent` runs a `planner` then a `coder` |
| **Sandboxed execution** | the coder runs Python in the built-in `python_exec` sandbox |
| **Tool approval** | `ToolApprovalPlugin` gates `python_exec`; the approver inspects the code and denies obvious file/network access (defense in depth) |

For a docs-lookup step on sites without an API, add the
[browser tools](../../recipes/browser.py) to the coder — they're gated by the
same approval mechanism.
