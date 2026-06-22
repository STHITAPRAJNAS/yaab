# YAAB cookbook

Mature, runnable examples for every major capability. Each **recipe** is one
focused file; each **app** is a realistic multi-feature application. Everything
runs offline with a deterministic `TestModel` (so it and its CI test need no
key), and runs live by setting one env var:

```bash
export YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash   # any LiteLLM model id
python -m cookbook.recipes.agents
```

## Recipes

| # | Recipe | Capability |
|---|--------|------------|
| 01 | [`agents`](recipes/agents.py) | The typed `Agent` unit of work |

_More recipes land in Wave 2; apps in Wave 3._

## Apps

| App | Scenario |
|-----|----------|
| _(coming in Wave 3)_ | |
