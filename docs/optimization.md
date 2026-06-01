# Optimization (DSPy-style)

The optional optimizable layer lets you *program* an agent declaratively and
*compile* it against a metric — tuning instructions and few-shot demonstrations
at build time, then **freezing** the result into a versioned artifact so
production runs are deterministic (no runtime optimization).

## Signatures

A signature is a typed input→output spec:

```python
from yaab.optimize import Signature, Predict

sig = Signature.parse("question -> answer", instructions="Answer accurately.")
```

## Modules

Modules are composable strategies over a signature:

```python
from yaab.optimize import Predict, ChainOfThought, ReAct

qa  = Predict("question -> answer", model="openai/gpt-4o")
cot = ChainOfThought("question -> answer", model="openai/gpt-4o")   # adds a reasoning field
react = ReAct("question -> answer", model="openai/gpt-4o", tools=[search])

out = await qa.forward(question="What is the capital of France?")
print(out["answer"])
```

## Optimizers

Compile a module against a trainset and metric. The trainset is a list of
`Case`s (shared with the eval framework); the metric scores a prediction.

```python
from yaab.optimize import BootstrapFewShot, MIPROv2, GEPA
from yaab.governance.eval import Case

trainset = [
    Case(name="c1", inputs={"question": "2+2?"}, expected="4"),
    Case(name="c2", inputs={"question": "3+5?"}, expected="8"),
]

def metric(case, prediction):
    return 1.0 if prediction.get("answer") == case.expected else 0.0

artifact = await BootstrapFewShot(max_demos=4).compile(qa, trainset, metric)
```

Available optimizers:

| Optimizer | Strategy |
|---|---|
| `BootstrapFewShot` | bake in demos the module already answers correctly |
| `MIPROv2` | search over instruction candidates × bootstrapped demo sets |
| `GEPA` | reflective instruction evolution using the worst case as feedback |

> The `MIPROv2` and `GEPA` implementations capture the API and contract; they
> are simplified relative to DSPy's full Bayesian / genetic-Pareto search.

## Freeze & deploy

A `CompiledArtifact` is a versioned, registry-trackable object. Load it into the
module for deterministic production behavior:

```python
artifact.instructions, artifact.demos, artifact.train_score, artifact.artifact_id
qa.load(artifact)            # production: no runtime tuning
frozen = qa.freeze()         # snapshot the current module as an artifact
```

Store the `artifact_id` on the agent's registry card to tie the deployed prompt
to its governance record.

## Inspect the compiled prompt

See exactly what a module will send — instructions + few-shot demos + inputs —
without calling the model:

```python
print(qa.inspect_prompt(question="What is the capital of France?"))
```

Useful for debugging an optimized program and for attaching the rendered prompt
to an audit/validation record.
