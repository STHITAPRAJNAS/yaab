"""Evaluation — a code-first eval framework.

Datasets of :class:`Case`s are run against a task function and scored by one or
more :class:`Evaluator`s (deterministic metrics or LLM-as-judge). Results are
versioned and can be attached to a registry entry as validation / outcomes-
analysis evidence, and the same metrics drive the optimizer layer.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field

TaskFn = Callable[[Any], Union[Any, Awaitable[Any]]]


class Case(BaseModel):
    """One evaluation example: input, optional expected output, metadata."""

    name: str = ""
    inputs: Any = None
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str = "dataset"
    cases: list[Case] = Field(default_factory=list)


@runtime_checkable
class Evaluator(Protocol):
    name: str

    def evaluate(self, case: Case, output: Any) -> float:
        """Return a score in [0, 1]."""
        ...


class ExactMatch:
    name = "exact_match"

    def evaluate(self, case: Case, output: Any) -> float:
        return 1.0 if output == case.expected else 0.0


class Contains:
    name = "contains"

    def evaluate(self, case: Case, output: Any) -> float:
        return 1.0 if str(case.expected) in str(output) else 0.0


class FunctionEvaluator:
    """Wrap an arbitrary scoring function as an :class:`Evaluator`."""

    def __init__(self, fn: Callable[[Case, Any], float], name: str = "custom") -> None:
        self.fn = fn
        self.name = name

    def evaluate(self, case: Case, output: Any) -> float:
        return self.fn(case, output)


class CaseResult(BaseModel):
    case: str
    output: Any = None
    scores: dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None


class ExperimentResult(BaseModel):
    name: str
    timestamp: float = Field(default_factory=time.time)
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def aggregate(self) -> dict[str, float]:
        """Mean score per evaluator across all cases."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for r in self.results:
            for k, v in r.scores.items():
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
        return {k: sums[k] / counts[k] for k in sums}

    @property
    def mean_score(self) -> float:
        agg = self.aggregate
        return sum(agg.values()) / len(agg) if agg else 0.0


class Experiment:
    """Runs a task over a dataset and scores it with a set of evaluators."""

    def __init__(
        self,
        dataset: Dataset,
        evaluators: list[Evaluator],
        *,
        name: str = "experiment",
    ) -> None:
        self.dataset = dataset
        self.evaluators = evaluators
        self.name = name

    async def run(self, task: TaskFn) -> ExperimentResult:
        import inspect

        result = ExperimentResult(name=self.name)
        for case in self.dataset.cases:
            cr = CaseResult(case=case.name or str(case.inputs))
            try:
                out = task(case.inputs)
                if inspect.isawaitable(out):
                    out = await out
                cr.output = out
                for ev in self.evaluators:
                    cr.scores[ev.name] = ev.evaluate(case, out)
            except Exception as exc:  # noqa: BLE001 - record, don't abort the suite
                cr.error = str(exc)
            result.results.append(cr)
        return result
