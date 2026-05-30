"""Model layer: provider protocol, LiteLLM, instrumentation, and test doubles."""

from __future__ import annotations

from .base import ModelProvider, ModelResponse, StreamChunk
from .instrumented import InstrumentedModel
from .litellm_provider import LiteLLMModel
from .test_model import FunctionModel, TestModel

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "StreamChunk",
    "LiteLLMModel",
    "InstrumentedModel",
    "TestModel",
    "FunctionModel",
    "resolve_model",
]


def resolve_model(model: str | ModelProvider) -> ModelProvider:
    """Coerce a model spec into a :class:`ModelProvider`.

    Strings are treated as LiteLLM model identifiers (e.g.
    ``"openai/gpt-4o"``, ``"anthropic/claude-..."``); provider instances pass
    through unchanged.
    """
    if isinstance(model, str):
        return LiteLLMModel(model)
    return model
