"""Eval adapters — bridge YAAB's eval framework to external eval libraries.

YAAB already ships deterministic + LLM-judge metrics (:mod:`yaab.governance.eval`)
and RAG groundedness metrics (:mod:`yaab.rag.eval`). This module adds an
*adapter layer* so popular external eval suites — RAGAS, DeepEval — plug in
behind the same :class:`~yaab.governance.eval.Evaluator` contract, and so any
metric can be registered/discovered through the component registry.

Design principle: **extensible by default.** A metric is just an object with a
``name`` and an ``evaluate(case, output) -> float`` (or async ``ascore``). Built-in
metrics, RAG metrics, RAGAS, DeepEval, and your own all satisfy it and can be
registered under the ``metric`` component kind:

    from yaab.eval import register_metric, get_metric, available_metrics

    register_metric("my_metric", lambda **kw: MyMetric(**kw))
    metric = get_metric("faithfulness")        # built-in
    metric = get_metric("ragas:faithfulness")  # via the RAGAS adapter (lazy)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..extensions import available as _available
from ..extensions import get as _get
from ..extensions import register as _register
from ..governance.eval import (
    Case,
    Contains,
    ExactMatch,
    JSONMatch,
    Levenshtein,
    LLMJudge,
    NumericTolerance,
    Regex,
)


# --- registry sugar (the "metric" component kind) ---------------------
def register_metric(name: str, factory: Callable[..., Any] | None = None) -> Any:
    """Register a metric factory under the ``metric`` component kind."""
    return _register("metric", name, factory)


def get_metric(name: str, /, **kwargs: Any) -> Any:
    """Instantiate a registered metric by name (e.g. ``"faithfulness"``)."""
    return _get("metric", name, **kwargs)


def available_metrics() -> list[str]:
    """List registered metric names (built-ins + adapters + third-party)."""
    return _available("metric")


# --- normalize any metric to a uniform async scorer -------------------
async def score(metric: Any, case: Case, output: Any) -> float:
    """Score ``output`` for ``case`` with ``metric``, sync or async.

    Accepts anything implementing ``ascore(case, output)`` (async) or
    ``evaluate(case, output)`` (sync) — so built-in, RAG, RAGAS, DeepEval, and
    custom metrics are all callable the same way.
    """
    if hasattr(metric, "ascore"):
        return await metric.ascore(case, output)
    if hasattr(metric, "evaluate"):
        return metric.evaluate(case, output)
    if callable(metric):
        result = metric(case, output)
        return await result if hasattr(result, "__await__") else result
    raise TypeError(f"{metric!r} is not a metric (needs ascore/evaluate/callable)")


# --- register the built-in deterministic metrics ----------------------
_register("metric", "exact_match", lambda **kw: ExactMatch())
_register("metric", "contains", lambda **kw: Contains())
_register("metric", "regex", lambda **kw: Regex())
_register("metric", "json_match", lambda **kw: JSONMatch())
_register("metric", "numeric_tolerance", lambda **kw: NumericTolerance(**kw))
_register("metric", "levenshtein", lambda **kw: Levenshtein())
_register("metric", "llm_judge", lambda **kw: LLMJudge(**kw))


def _register_rag_metrics() -> None:
    # Lazy: RAG metrics live in yaab.rag.eval; register thin wrappers.
    from ..rag.eval import FaithfulnessEvaluator

    class _Faithfulness:
        name = "faithfulness"

        async def ascore(self, case: Case, output: Any) -> float:
            from ..rag.eval import faithfulness

            chunks = case.metadata.get("chunks", [])
            return faithfulness(str(output), chunks)

    class _ContextRelevance:
        name = "context_relevance"

        async def ascore(self, case: Case, output: Any) -> float:
            from ..rag.eval import context_relevance

            chunks = case.metadata.get("chunks", [])
            return context_relevance(str(case.inputs), chunks)

    _register("metric", "faithfulness", lambda **kw: _Faithfulness())
    _register("metric", "context_relevance", lambda **kw: _ContextRelevance())
    _register("metric", "faithfulness_llm", lambda **kw: FaithfulnessEvaluator(**kw))


_register_rag_metrics()

# Register the external-suite adapters lazily (factories import on demand).
from .adapters import deepeval as _deepeval_adapter  # noqa: E402
from .adapters import ragas as _ragas_adapter  # noqa: E402

_ragas_adapter.register()
_deepeval_adapter.register()


__all__ = [
    "register_metric",
    "get_metric",
    "available_metrics",
    "score",
    "Case",
]
