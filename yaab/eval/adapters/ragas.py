"""RAGAS adapter — expose RAGAS metrics as YAAB evaluators.

RAGAS scores RAG quality (faithfulness, answer relevancy, context precision/
recall). This wraps a RAGAS metric so it satisfies YAAB's ``ascore(case, output)``
contract: the retrieved context comes from ``case.metadata['chunks']`` (a list of
``RetrievedChunk``) and the question from ``case.inputs``.

``ragas`` is imported lazily — only when a ``ragas:*`` metric is instantiated —
so the dependency is fully optional.
"""

from __future__ import annotations

from typing import Any

from ...governance.eval import Case

# RAGAS metric name -> the attribute to import from `ragas.metrics`.
_RAGAS_METRICS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
}


class RagasMetric:
    """Adapt a single RAGAS metric to the YAAB evaluator contract."""

    def __init__(self, metric: str = "faithfulness", *, llm: Any = None, embeddings: Any = None):
        if metric not in _RAGAS_METRICS:
            raise ValueError(
                f"unknown RAGAS metric {metric!r}; choose from {sorted(_RAGAS_METRICS)}"
            )
        self.name = f"ragas:{metric}"
        self.metric = metric
        self.llm = llm
        self.embeddings = embeddings

    async def ascore(self, case: Case, output: Any) -> float:
        try:
            from ragas import evaluate  # type: ignore
            from ragas import metrics as ragas_metrics  # type: ignore
            from datasets import Dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "ragas (and datasets) are required for the RAGAS adapter. "
                "`pip install ragas datasets`."
            ) from exc

        chunks = case.metadata.get("chunks", [])
        contexts = [c.text for c in chunks] if chunks else [str(case.inputs)]
        row = {
            "question": [str(case.inputs)],
            "answer": [str(output)],
            "contexts": [contexts],
            "ground_truth": [str(case.expected) if case.expected is not None else ""],
        }
        metric_obj = getattr(ragas_metrics, _RAGAS_METRICS[self.metric])
        result = evaluate(Dataset.from_dict(row), metrics=[metric_obj])
        # RAGAS returns a result mapping metric name -> score.
        scores = result if isinstance(result, dict) else result.scores[0]
        return float(next(iter(scores.values())) if isinstance(scores, dict) else scores)


def register() -> None:
    """Register a ``ragas:<metric>`` factory for each supported RAGAS metric."""
    from ...extensions import register as _register

    for metric in _RAGAS_METRICS:
        _register("metric", f"ragas:{metric}", lambda metric=metric, **kw: RagasMetric(metric, **kw))


__all__ = ["RagasMetric", "register"]
