"""Shared cookbook helpers: offline/live model resolution, assertions, cassettes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_model(model: Any = None, *, offline_default: Any) -> Any:
    """Pick the model: explicit arg > ``YAAB_SAMPLE_MODEL`` env > offline default.

    Lets one recipe run deterministically in CI and live with one env var::

        export YAAB_SAMPLE_MODEL=gemini/gemini-2.5-flash
    """
    if model is not None:
        return model
    env = os.environ.get("YAAB_SAMPLE_MODEL")
    if env:
        return env
    return offline_default


def expect(cond: bool, msg: str) -> None:
    """Assert a recipe/app produced the documented result (so it self-verifies)."""
    if not cond:
        raise AssertionError(f"cookbook check failed: {msg}")


def load_cassette(name: str, *, inner: Any = None) -> Any:
    """A ``CassetteModel`` for ``cookbook/cassettes/<name>.json``.

    Replays by default; with ``YAAB_RECORD=1`` and a real ``inner`` model it
    records (author once, replay in CI). Dogfoods the record/replay feature.
    """
    from yaab.testing import CassetteModel

    path = Path(__file__).resolve().parent / "cassettes" / f"{name}.json"
    record = os.environ.get("YAAB_RECORD") == "1" and inner is not None
    return CassetteModel(path, inner=inner, mode="record" if record else "replay")
