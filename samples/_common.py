"""Shared helpers for samples — model resolution for offline vs. real runs."""

from __future__ import annotations

import os
from typing import Any


def resolve_model(model: Any, *, offline_default: Any) -> Any:
    """Pick the model for a sample.

    Priority: explicit ``model`` arg > ``YAAB_SAMPLE_MODEL`` env var > the
    sample's offline ``TestModel`` default. This lets the same sample run in CI
    (offline, deterministic) and against a real/free model with one env var:

        export YAAB_SAMPLE_MODEL=ollama/llama3
        export YAAB_SAMPLE_MODEL=gemini/gemini-2.0-flash   # free tier
    """
    if model is not None:
        return model
    env = os.environ.get("YAAB_SAMPLE_MODEL")
    if env:
        return env
    return offline_default
