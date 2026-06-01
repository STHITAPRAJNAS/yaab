"""Optimizers — compile a module against a metric.

``BootstrapFewShot`` selects demonstrations from a trainset that the module
answers correctly (per a metric) and bakes them in. ``MIPROv2`` and a
``GEPA``-style reflective optimizer are provided as instruction-search
strategies. Optimization happens at *build time*; the result is a frozen,
registry-trackable :class:`CompiledArtifact` so production runs are deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..governance.eval import Case
from .module import CompiledArtifact, Module

Metric = Callable[[Case, dict], float]


@runtime_checkable
class Optimizer(Protocol):
    name: str

    async def compile(
        self, module: Module, trainset: list[Case], metric: Metric
    ) -> CompiledArtifact: ...


class BootstrapFewShot:
    """Bootstrap few-shot demos from examples the module already gets right."""

    name = "bootstrap_few_shot"

    def __init__(self, max_demos: int = 4, threshold: float = 0.5) -> None:
        self.max_demos = max_demos
        self.threshold = threshold

    async def compile(
        self, module: Module, trainset: list[Case], metric: Metric
    ) -> CompiledArtifact:
        demos: list[dict[str, Any]] = []
        scores: list[float] = []
        for case in trainset:
            inputs = case.inputs if isinstance(case.inputs, dict) else {"input": case.inputs}
            prediction = await module.forward(**inputs)
            score = metric(case, prediction)
            scores.append(score)
            if score >= self.threshold and len(demos) < self.max_demos:
                demo = dict(inputs)
                demo.update(prediction)
                demos.append(demo)
        module.demos = demos
        mean = sum(scores) / len(scores) if scores else 0.0
        return CompiledArtifact(
            instructions=module.signature.instructions,
            demos=demos,
            optimizer=self.name,
            train_score=mean,
        )


class BootstrapFewShotWithRandomSearch:
    """Workhorse optimizer: bootstrap a demo pool, then random-search demo
    subsets and keep the best on a validation set.

    1. Run the module over ``trainset`` and collect every example it answers
       correctly (per ``metric``) into a demo pool.
    2. Build candidate demo sets — zero-shot, the full (capped) pool, and
       ``num_candidates`` random subsets — and score each on ``valset`` (defaults
       to ``trainset``).
    3. Freeze the best-scoring set. Falls back to zero-shot if nothing bootstraps.

    Random choices are seeded (``seed``) so a compile is reproducible.
    """

    name = "bootstrap_rs"

    def __init__(
        self,
        *,
        max_demos: int = 4,
        num_candidates: int = 8,
        threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.max_demos = max_demos
        self.num_candidates = num_candidates
        self.threshold = threshold
        self.seed = seed

    async def compile(
        self,
        module: Module,
        trainset: list[Case],
        metric: Metric,
        *,
        valset: list[Case] | None = None,
    ) -> CompiledArtifact:
        import random

        val = valset or trainset
        # 1. Bootstrap a pool of demos the module already gets right.
        pool: list[dict[str, Any]] = []
        for case in trainset:
            inputs = case.inputs if isinstance(case.inputs, dict) else {"input": case.inputs}
            prediction = await module.forward(**inputs)
            if metric(case, prediction) >= self.threshold:
                demo = dict(inputs)
                demo.update(prediction)
                pool.append(demo)

        # 2. Candidate demo sets: zero-shot, full (capped) pool, random subsets.
        rng = random.Random(self.seed)
        candidates: list[list[dict[str, Any]]] = [[]]
        if pool:
            candidates.append(pool[: self.max_demos])
            for _ in range(self.num_candidates):
                k = rng.randint(1, min(self.max_demos, len(pool)))
                candidates.append(rng.sample(pool, k))

        # 3. Score each candidate on the validation set; keep the best.
        best_demos: list[dict[str, Any]] = []
        best_score = -1.0
        for demos in candidates:
            module.demos = demos
            score = await _mean_score(module, val, metric)
            if score > best_score:
                best_demos, best_score = demos, score

        module.demos = best_demos
        return CompiledArtifact(
            instructions=module.signature.instructions,
            demos=best_demos,
            optimizer=self.name,
            train_score=max(best_score, 0.0),
        )


class MIPROv2:
    """Instruction × demo search with minibatched candidate evaluation.

    Proposes candidate instruction phrasings and demo sets (zero-shot +
    bootstrapped), scores each candidate on a random ``minibatch_size`` slice of
    the trainset (minibatching keeps search cheap), then re-scores the top
    candidates on the full set to pick a winner. A fuller implementation would run
    a Bayesian search; this captures the minibatch-then-confirm contract.
    """

    name = "miprov2"

    def __init__(
        self,
        candidates: list[str] | None = None,
        *,
        bootstrap_demos: bool = True,
        minibatch_size: int = 0,
        seed: int = 0,
    ) -> None:
        self.candidates = candidates or [
            "Answer accurately and concisely.",
            "Think carefully, then give the precise answer.",
            "Be correct. Prefer the exact expected format.",
        ]
        self.bootstrap_demos = bootstrap_demos
        self.minibatch_size = minibatch_size
        self.seed = seed

    async def compile(
        self, module: Module, trainset: list[Case], metric: Metric
    ) -> CompiledArtifact:
        import random

        original = module.signature.instructions
        demo_sets: list[list[dict]] = [[]]
        if self.bootstrap_demos:
            booted = await BootstrapFewShot().compile(module, trainset, metric)
            module.demos = []  # reset; we search demos explicitly
            demo_sets.append(booted.demos)

        rng = random.Random(self.seed)

        def batch() -> list[Case]:
            if self.minibatch_size and 0 < self.minibatch_size < len(trainset):
                return rng.sample(trainset, self.minibatch_size)
            return trainset

        # Stage 1: cheap minibatch scoring of every (instruction, demos) candidate.
        scored: list[tuple[str, list[dict], float]] = []
        for instr in [original, *self.candidates]:
            for demos in demo_sets:
                module.signature.instructions = instr
                module.demos = demos
                score = await _mean_score(module, batch(), metric)
                scored.append((instr, demos, score))

        # Stage 2: confirm the top few candidates on the full trainset.
        scored.sort(key=lambda t: t[2], reverse=True)
        best = (original, list(module.demos), -1.0)
        for instr, demos, _ in scored[:3]:
            module.signature.instructions = instr
            module.demos = demos
            score = await _mean_score(module, trainset, metric)
            if score > best[2]:
                best = (instr, demos, score)

        module.signature.instructions, module.demos = best[0], best[1]
        return CompiledArtifact(
            instructions=best[0], demos=best[1], optimizer=self.name, train_score=best[2]
        )


class GEPA:
    """Genetic-Pareto reflective optimizer (simplified contract).

    Reflectively mutates the instruction using the worst-scoring case as
    feedback, keeping the best variant. The real GEPA evolves a Pareto front of
    candidates with an LLM reflection step; this preserves the API shape.
    """

    name = "gepa"

    def __init__(
        self, generations: int = 3, reflect: Callable[[str, Case, dict], str] | None = None
    ):
        self.generations = generations
        self.reflect = reflect or self._default_reflect

    @staticmethod
    def _default_reflect(instr: str, case: Case, prediction: dict) -> str:
        return f"{instr} Pay special attention to cases like: {case.inputs!r}."

    async def compile(
        self, module: Module, trainset: list[Case], metric: Metric
    ) -> CompiledArtifact:
        best_instr = module.signature.instructions
        best_score = await self._score(module, trainset, metric)
        for _ in range(self.generations):
            worst = await self._worst_case(module, trainset, metric)
            if worst is None:
                break
            case, prediction = worst
            candidate = self.reflect(best_instr, case, prediction)
            module.signature.instructions = candidate
            score = await self._score(module, trainset, metric)
            if score > best_score:
                best_score, best_instr = score, candidate
            else:
                module.signature.instructions = best_instr
        module.signature.instructions = best_instr
        return CompiledArtifact(
            instructions=best_instr,
            demos=list(module.demos),
            optimizer=self.name,
            train_score=best_score,
        )

    async def _score(self, module: Module, trainset: list[Case], metric: Metric) -> float:
        if not trainset:
            return 0.0
        total = 0.0
        for case in trainset:
            inputs = case.inputs if isinstance(case.inputs, dict) else {"input": case.inputs}
            total += metric(case, await module.forward(**inputs))
        return total / len(trainset)

    async def _worst_case(self, module: Module, trainset: list[Case], metric: Metric):
        worst = None
        worst_score = 2.0
        for case in trainset:
            inputs = case.inputs if isinstance(case.inputs, dict) else {"input": case.inputs}
            prediction = await module.forward(**inputs)
            score = metric(case, prediction)
            if score < worst_score:
                worst_score, worst = score, (case, prediction)
        return worst


async def _mean_score(module: Module, trainset: list[Case], metric: Metric) -> float:
    if not trainset:
        return 0.0
    total = 0.0
    for case in trainset:
        inputs = case.inputs if isinstance(case.inputs, dict) else {"input": case.inputs}
        total += metric(case, await module.forward(**inputs))
    return total / len(trainset)


__all__ = [
    "Optimizer",
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "MIPROv2",
    "GEPA",
    "Metric",
]
