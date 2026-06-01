"""Evaluation — a code-first eval framework.

Datasets of :class:`Case`s are run against a task function and scored by one or
more :class:`Evaluator`s (deterministic metrics or LLM-as-judge). Results are
versioned and can be attached to a registry entry as validation / outcomes-
analysis evidence, and the same metrics drive the optimizer layer.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

TaskFn = Callable[[Any], Any | Awaitable[Any]]


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


class ToolTrajectoryMatch:
    """Score how well an agent's tool-call sequence matches an expected one.

    This is YAAB's analogue of ADK's ``tool_trajectory_avg_score``. Unlike the
    output-string metrics above, it scores the *process* — which tools were
    called, in what order, with which arguments — which is what you actually
    want to regression-test for tool-using agents.

    The expected trajectory is a list of ``{"name": str, "arguments"?: dict}``
    steps, read from ``case.metadata["expected_tool_trajectory"]`` (or, as a
    convenience, ``case.expected`` when it is a list). The *actual* trajectory
    is pulled from the run's events by :meth:`Experiment.run` and handed to this
    evaluator via a context dict — so this evaluator's ``output`` argument is the
    context dict, not the final string. That is why it is *context-aware*:
    :meth:`Experiment.run` detects evaluators that accept the context and feeds
    it to them, while keeping plain ``(case, output)`` evaluators working.

    Scoring (``strict=False``, the default): the fraction of expected steps that
    appear in the actual trajectory as an *ordered subsequence* (so a missing or
    reordered step costs proportionally, never more). With ``strict=True`` the
    actual sequence must equal the expected sequence exactly (1.0 or 0.0).

    A step's arguments, when given, must be a *subset* of the actual call's
    arguments (extra actual args are fine) — agents often pass defaults the
    eval author did not pin down, and over-specifying would make tests brittle.
    """

    name = "tool_trajectory"

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    @staticmethod
    def _expected(case: Case) -> list[dict[str, Any]]:
        expected = case.metadata.get("expected_tool_trajectory")
        if expected is None and isinstance(case.expected, list):
            expected = case.expected
        return list(expected or [])

    @staticmethod
    def _actual(output: Any) -> list[dict[str, Any]]:
        # Context-aware path: Experiment.run passes {"tool_trajectory": [...]}.
        if isinstance(output, dict) and "tool_trajectory" in output:
            return list(output.get("tool_trajectory") or [])
        # Tolerate being handed the raw trajectory list directly.
        if isinstance(output, list):
            return output
        return []

    @staticmethod
    def _step_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        if expected.get("name") != actual.get("name"):
            return False
        want_args = expected.get("arguments")
        if not want_args:
            return True
        got_args = actual.get("arguments") or {}
        return all(got_args.get(k) == v for k, v in want_args.items())

    def evaluate(self, case: Case, output: Any) -> float:
        expected = self._expected(case)
        actual = self._actual(output)
        if not expected:
            # Nothing was asked for: a run that also called nothing is perfect.
            return 1.0 if not actual else 0.0

        if self.strict:
            if len(expected) != len(actual):
                return 0.0
            pairs = zip(expected, actual, strict=True)
            return 1.0 if all(self._step_matches(e, a) for e, a in pairs) else 0.0

        # Ordered match: the longest common subsequence (by step-match) between
        # the expected and actual trajectories, normalized by the number of
        # expected steps. LCS (not a greedy single pass) is what makes a missing
        # middle step cost exactly that one step, and a reordered step cost only
        # what falls out of order — never more.
        n, m = len(expected), len(actual)
        # dp[j] = LCS length using all of expected[:i] and actual[:j].
        dp = [0] * (m + 1)
        for i in range(1, n + 1):
            prev_diag = 0
            for j in range(1, m + 1):
                cur = dp[j]
                if self._step_matches(expected[i - 1], actual[j - 1]):
                    dp[j] = prev_diag + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev_diag = cur
        return dp[m] / len(expected)


class CaseResult(BaseModel):
    case: str
    output: Any = None
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


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
                # When the task returns a RunResult, unpack it: the final-output
                # string is what plain metrics score, while the tool-call
                # trajectory (extracted from the run's events) is exposed to
                # context-aware metrics like ToolTrajectoryMatch.
                output, context = _unpack_task_output(out)
                cr.output = output
                for ev in self.evaluators:
                    cr.scores[ev.name] = await _score_evaluator(ev, case, output, context)
            except Exception as exc:  # noqa: BLE001 - record, don't abort the suite
                cr.error = str(exc)
            result.results.append(cr)
        return result


def _unpack_task_output(out: Any) -> tuple[Any, dict[str, Any]]:
    """Split a task result into (final_output, evaluator_context).

    A plain string/value is returned as-is with an empty context. A
    :class:`~yaab.types.RunResult` is unpacked into its ``.output`` plus a
    context dict carrying the tool-call trajectory mined from ``.events``
    (``EventType.TOOL_CALL`` payloads), so trajectory-aware evaluators can score
    the agent's process without the eval author wiring it up by hand.
    """
    events = getattr(out, "events", None)
    if events is None or not hasattr(out, "output"):
        return out, {}
    trajectory: list[dict[str, Any]] = []
    for event in events:
        # Match by the event-type *value* to avoid importing EventType eagerly
        # (and to tolerate plain dicts/duck-typed events in tests).
        etype = getattr(event, "type", None)
        type_value = getattr(etype, "value", etype)
        if type_value == "tool_call":
            payload = getattr(event, "payload", {}) or {}
            trajectory.append(
                {"name": payload.get("name"), "arguments": payload.get("arguments", {})}
            )
    context = {"tool_trajectory": trajectory, "output": out.output, "run_result": out}
    return out.output, context


async def _score_evaluator(
    ev: Evaluator, case: Case, output: Any, context: dict[str, Any]
) -> float:
    """Score one evaluator, feeding the run context to those that accept it.

    Context-aware evaluators (e.g. :class:`ToolTrajectoryMatch`) need the tool
    trajectory, not the final string. We pass ``context`` as the second argument
    to any evaluator that opts in — detected via a ``wants_context`` flag, or by
    falling back to the context on a ``TypeError`` — and pass the plain output to
    everyone else, preserving backward compatibility with ``(case, output)``
    metrics.
    """
    wants_context = getattr(ev, "wants_context", None)
    is_trajectory = isinstance(ev, ToolTrajectoryMatch)
    payload: Any = context if (wants_context or is_trajectory) else output
    if hasattr(ev, "ascore"):
        try:
            return await ev.ascore(case, payload)
        except TypeError:
            return await ev.ascore(case, output)
    try:
        return ev.evaluate(case, payload)
    except TypeError:
        return ev.evaluate(case, output)


# Register the trajectory metric under the "metric" component kind, the same way
# the built-ins are registered in yaab.eval. This is a side effect of importing
# this module (which yaab.eval already does), so `get_metric("tool_trajectory")`
# works without touching files outside this module's ownership.
def _register_trajectory_metric() -> None:
    from ..extensions import register as _register

    _register("metric", "tool_trajectory", lambda **kw: ToolTrajectoryMatch(**kw))


_register_trajectory_metric()
