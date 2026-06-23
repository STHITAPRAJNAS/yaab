# Cookbook

Mature, runnable examples for every major capability, in the
[`cookbook/`](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook) tree.
Each **recipe** is one focused file; each **app** is a realistic multi-feature
application. Everything runs offline with a deterministic `TestModel` and runs
live by setting `YAAB_SAMPLE_MODEL`.

```bash
export YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash
python -m cookbook.recipes.agents
```

See the [catalog README](https://github.com/STHITAPRAJNAS/yaab/tree/main/cookbook#readme)
for the full list of recipes and apps.
