"""Modules — composable, parameterized strategies over a signature (DSPy-style).

``Predict`` does a single signature-driven completion; ``ChainOfThought`` adds a
reasoning field; ``ReAct`` interleaves reasoning with tool calls. Modules carry
*parameters* (instructions + few-shot demos) that an :class:`~yaab.optimize.optimizer.Optimizer`
can tune and then freeze into a deployable, versioned artifact.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..models import ModelProvider, resolve_model
from ..types import Message, Role
from .signature import Signature


class CompiledArtifact(BaseModel):
    """A frozen, versioned module — deterministic in production (no runtime tuning)."""

    artifact_id: str = Field(default_factory=lambda: f"opt_{uuid.uuid4().hex[:12]}")
    version: str = "1"
    instructions: str = ""
    demos: list[dict[str, Any]] = Field(default_factory=list)
    optimizer: str = ""
    train_score: float = 0.0


class Module:
    """Base optimizable module bound to a :class:`Signature`."""

    def __init__(
        self,
        signature: str | Signature,
        *,
        model: str | ModelProvider = "openai/gpt-4o",
        instructions: str = "",
    ) -> None:
        self.signature = (
            Signature.parse(signature, instructions=instructions)
            if isinstance(signature, str)
            else signature
        )
        self._model_spec = model
        self._model: ModelProvider | None = None
        self.demos: list[dict[str, Any]] = []

    @property
    def model(self) -> ModelProvider:
        if self._model is None:
            self._model = resolve_model(self._model_spec)
        return self._model

    def inspect_prompt(self, **inputs: Any) -> str:
        """Return the exact prompt this module would send for ``inputs``.

        Renders the current instructions + few-shot demos + inputs *without*
        calling the model — the visibility DSPy users ask for (#7830) when
        debugging or auditing an optimized program.
        """
        return self.signature.render_prompt(
            {k: str(v) for k, v in inputs.items()}, demos=self.demos
        )

    async def forward(self, **inputs: Any) -> dict[str, str]:
        prompt = self.inspect_prompt(**inputs)
        resp = await self.model.complete([Message(role=Role.USER, content=prompt)])
        return self.signature.parse_output(resp.content)

    def load(self, artifact: CompiledArtifact) -> Module:
        """Apply a compiled artifact (instructions + demos) to this module."""
        if artifact.instructions:
            self.signature.instructions = artifact.instructions
        self.demos = list(artifact.demos)
        return self

    def freeze(self, optimizer: str = "manual", score: float = 0.0) -> CompiledArtifact:
        return CompiledArtifact(
            instructions=self.signature.instructions,
            demos=list(self.demos),
            optimizer=optimizer,
            train_score=score,
        )


class Predict(Module):
    """A single signature-driven prediction."""


class ChainOfThought(Module):
    """Adds an explicit ``reasoning`` output field before the answer."""

    def __init__(self, signature: str | Signature, **kwargs: Any) -> None:
        super().__init__(signature, **kwargs)
        from .signature import FieldSpec

        if not any(f.name == "reasoning" for f in self.signature.outputs):
            self.signature.outputs.insert(
                0, FieldSpec(name="reasoning", description="Think step by step.")
            )


class ReAct(Module):
    """Reason+act module: interleaves reasoning with tool calls.

    Thin wrapper that delegates the tool loop to an :class:`~yaab.agent.Agent`,
    so optimization and execution share one runtime.
    """

    def __init__(
        self, signature: str | Signature, *, tools: list[Any] | None = None, **kwargs: Any
    ):
        super().__init__(signature, **kwargs)
        self.tools = tools or []

    async def forward(self, **inputs: Any) -> dict[str, str]:
        from ..agent import Agent

        agent: Agent = Agent(
            "react",
            model=self._model_spec,
            instructions=self.signature.instructions or "Reason step by step, using tools.",
            tools=self.tools,
        )
        prompt = self.signature.render_prompt(
            {k: str(v) for k, v in inputs.items()}, demos=self.demos
        )
        result = await agent.run(prompt)
        return self.signature.parse_output(result.output)
