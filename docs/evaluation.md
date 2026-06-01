# Evaluation

YAAB treats evaluation as a first-class, **extensible-by-design** concern. A
metric is just an object with a ``name`` and either ``evaluate(case, output) ->
float`` (sync) or ``ascore(case, output) -> float`` (async). Built-in metrics,
RAG groundedness metrics, external suites (RAGAS, DeepEval), and your own all
satisfy that one contract and are discoverable through the metric registry.

## The metric registry

```python
from yaab import available_metrics, get_metric, register_metric

available_metrics()          # ['exact_match', 'faithfulness', 'ragas:faithfulness', ...]
m = get_metric("exact_match")
m = get_metric("numeric_tolerance", tol=0.01)
```

Built-in metrics (`yaab.governance.eval`):

| Metric | Kind |
|---|---|
| `exact_match`, `contains`, `regex`, `json_match` | deterministic |
| `numeric_tolerance`, `levenshtein` | deterministic |
| `llm_judge` | LLM judge (async) |
| `faithfulness`, `context_relevance`, `faithfulness_llm` | RAG groundedness |

## External suites via adapters

RAGAS and DeepEval plug in behind the same contract; their libraries are
imported **only when a metric is instantiated and scored**, so they stay
optional.

```python
# pip install 'yaab-sdk[ragas]'  /  'yaab-sdk[deepeval]'
faith = get_metric("ragas:faithfulness")
rel   = get_metric("deepeval:answer_relevancy", threshold=0.7)
```

RAGAS metrics: `ragas:faithfulness`, `ragas:answer_relevancy`,
`ragas:context_precision`, `ragas:context_recall`.
DeepEval metrics: `deepeval:answer_relevancy`, `deepeval:faithfulness`,
`deepeval:hallucination`, `deepeval:bias`, `deepeval:toxicity`.

Both read the retrieved context from `case.metadata["chunks"]` (a list of
`RetrievedChunk`) and the question from `case.inputs`, so they work directly with
[RAG](rag.md) retrievals.

## Scoring uniformly

`yaab.eval.score` runs any metric — sync or async — the same way:

```python
from yaab.eval import score
from yaab.governance.eval import Case

s = await score(get_metric("faithfulness"), Case(inputs="q", metadata={"chunks": chunks}), answer)
```

## In an experiment / CI

`Experiment` runs a task over a `Dataset` and applies a mix of sync and async
metrics:

```python
from yaab.governance import Dataset, Experiment

ds = Dataset(name="qa", cases=[Case(name="c1", inputs="2+2?", expected="4")])
exp = Experiment(ds, [get_metric("exact_match"), get_metric("llm_judge")])
report = await exp.run(lambda x: my_agent.run_sync(x).output)
print(report.aggregate)        # mean score per metric
```

Results feed the [drift monitor and trust scorer](governance.md#drift-detection--trust-scoring).

## Add your own (extensibility)

Register a metric in-process, or ship it as a package via the `yaab.metrics`
entry point:

```python
class ConcisenessMetric:
    name = "conciseness"
    def evaluate(self, case, output):
        return 1.0 if len(str(output)) < 200 else 0.0

register_metric("conciseness", lambda **kw: ConcisenessMetric())
```

```toml
# pyproject.toml of a plugin package
[project.entry-points."yaab.metrics"]
conciseness = "my_pkg.metrics:ConcisenessMetric"
```
