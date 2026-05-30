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


class Regex:
    """Score 1.0 if the output matches a regex (the pattern is ``case.expected``)."""

    name = "regex"

    def evaluate(self, case: Case, output: Any) -> float:
        import re

        return 1.0 if re.search(str(case.expected), str(output)) else 0.0


class JSONMatch:
    """Score 1.0 if output parses to JSON equal to ``case.expected``."""

    name = "json_match"

    def evaluate(self, case: Case, output: Any) -> float:
        import json

        try:
            parsed = json.loads(output) if isinstance(output, str) else output
        except (json.JSONDecodeError, TypeError):
            return 0.0
        expected = case.expected
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                pass
        return 1.0 if parsed == expected else 0.0


class NumericTolerance:
    """Score 1.0 if the numeric output is within ``tol`` of ``case.expected``."""

    name = "numeric_tolerance"

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def evaluate(self, case: Case, output: Any) -> float:
        try:
            return 1.0 if abs(float(output) - float(case.expected)) <= self.tol else 0.0
        except (ValueError, TypeError):
            return 0.0


class Levenshtein:
    """Normalized edit-distance similarity in [0, 1] vs ``case.expected``."""

    name = "levenshtein"

    @staticmethod
    def _distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def evaluate(self, case: Case, output: Any) -> float:
        a, b = str(output), str(case.expected)
        if not a and not b:
            return 1.0
        return 1.0 - self._distance(a, b) / max(len(a), len(b))


class LLMJudge:
    """Score an output's quality 0-1 with a model judge (call :meth:`ascore`)."""

    name = "llm_judge"

    def __init__(self, model: Any, *, criteria: str = "correct and helpful") -> None:
        from ..models import resolve_model

        self.model = resolve_model(model)
        self.criteria = criteria

    async def ascore(self, case: Case, output: Any) -> float:
        import re

        from ..types import Message, Role

        prompt = (
            f"Rate the OUTPUT from 0 to 1 on: {self.criteria}. Reply with only a number.\n\n"
            f"INPUT: {case.inputs}\nEXPECTED: {case.expected}\nOUTPUT: {output}\n\nScore:"
        )
        try:
            resp = await self.model.complete([Message(role=Role.USER, content=prompt)])
            return float(re.search(r"[01](?:\.\d+)?", resp.content).group())  # type: ignore[union-attr]
        except (AttributeError, ValueError, TypeError):
            return 0.0


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
