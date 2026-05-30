"""Optional optimizable-program layer (DSPy-style)."""

from __future__ import annotations

from .module import ChainOfThought, CompiledArtifact, Module, Predict, ReAct
from .optimizer import GEPA, BootstrapFewShot, Metric, MIPROv2, Optimizer
from .signature import FieldSpec, Signature

__all__ = [
    "Signature",
    "FieldSpec",
    "Module",
    "Predict",
    "ChainOfThought",
    "ReAct",
    "CompiledArtifact",
    "Optimizer",
    "BootstrapFewShot",
    "MIPROv2",
    "GEPA",
    "Metric",
]
