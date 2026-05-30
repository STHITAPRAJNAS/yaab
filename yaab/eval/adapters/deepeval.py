"""DeepEval adapter — expose DeepEval metrics as YAAB evaluators.

DeepEval offers LLM-eval metrics (answer relevancy, faithfulness, hallucination,
bias, toxicity, G-Eval). This wraps a DeepEval metric so it satisfies YAAB's
``ascore(case, output)`` contract via a ``LLMTestCase`` built from the YAAB
``Case`` + output. ``deepeval`` is imported lazily.
"""

from __future__ import annotations

from typing import Any

from ...governance.eval import Case

# DeepEval metric name -> (module attribute, needs_retrieval_context)
_DEEPEVAL_METRICS = {
    "answer_relevancy": ("AnswerRelevancyMetric", False),
    "faithfulness": ("FaithfulnessMetric", True),
    "hallucination": ("HallucinationMetric", True),
    "bias": ("BiasMetric", False),
    "toxicity": ("ToxicityMetric", False),
}


class DeepEvalMetric:
    """Adapt a single DeepEval metric to the YAAB evaluator contract."""

    def __init__(self, metric: str = "answer_relevancy", *, threshold: float = 0.5, **kwargs: Any):
        if metric not in _DEEPEVAL_METRICS:
            raise ValueError(
                f"unknown DeepEval metric {metric!r}; choose from {sorted(_DEEPEVAL_METRICS)}"
            )
        self.name = f"deepeval:{metric}"
        self.metric = metric
        self.threshold = threshold
        self.kwargs = kwargs

    async def ascore(self, case: Case, output: Any) -> float:
        try:
            from deepeval import metrics as de_metrics  # type: ignore
            from deepeval.test_case import LLMTestCase  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "deepeval is required for the DeepEval adapter. `pip install deepeval`."
            ) from exc

        attr, needs_context = _DEEPEVAL_METRICS[self.metric]
        chunks = case.metadata.get("chunks", [])
        context = [c.text for c in chunks] if chunks else None
        test_case = LLMTestCase(
            input=str(case.inputs),
            actual_output=str(output),
            expected_output=str(case.expected) if case.expected is not None else None,
            retrieval_context=context if needs_context else None,
            context=context if needs_context else None,
        )
        metric_obj = getattr(de_metrics, attr)(threshold=self.threshold, **self.kwargs)
        metric_obj.measure(test_case)
        return float(metric_obj.score)


def register() -> None:
    """Register a ``deepeval:<metric>`` factory for each supported metric."""
    from ...extensions import register as _register

    for metric in _DEEPEVAL_METRICS:
        _register(
            "metric",
            f"deepeval:{metric}",
            lambda metric=metric, **kw: DeepEvalMetric(metric, **kw),
        )


__all__ = ["DeepEvalMetric", "register"]
