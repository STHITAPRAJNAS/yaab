from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest


def test_resolve_model_priority(monkeypatch):
    from cookbook._harness import resolve_model

    sentinel_offline = object()
    sentinel_explicit = object()

    # No env, no explicit -> the offline default.
    monkeypatch.delenv("YAAB_SAMPLE_MODEL", raising=False)
    assert resolve_model(offline_default=sentinel_offline) is sentinel_offline

    # Env set -> the env model string.
    monkeypatch.setenv("YAAB_SAMPLE_MODEL", "gemini/gemini-2.5-flash")
    assert resolve_model(offline_default=sentinel_offline) == "gemini/gemini-2.5-flash"

    # Explicit arg wins over both.
    assert resolve_model(sentinel_explicit, offline_default=sentinel_offline) is sentinel_explicit


def test_expect_raises_with_message():
    from cookbook._harness import expect

    expect(True, "this should not raise")
    with pytest.raises(AssertionError, match="cookbook check failed: bad"):
        expect(False, "bad")


def test_load_cassette_returns_replay_model(monkeypatch):
    from cookbook._harness import load_cassette
    from yaab.testing import CassetteModel

    monkeypatch.delenv("YAAB_RECORD", raising=False)
    model = load_cassette("does_not_exist_yet")
    assert isinstance(model, CassetteModel)
    assert model.mode == "replay"


def _recipe_names() -> list[str]:
    # Defensive: before the recipes package exists (TDD red step), return [] so
    # collection never hard-errors — test_at_least_one_recipe_exists fails cleanly.
    try:
        import cookbook.recipes as pkg
    except ModuleNotFoundError:
        return []
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__))


@pytest.mark.parametrize("name", _recipe_names())
async def test_recipe_runs_offline(name):
    mod = importlib.import_module(f"cookbook.recipes.{name}")
    result = await mod.run()
    assert result, f"recipe {name}.run() returned an empty result"


def test_at_least_one_recipe_exists():
    assert _recipe_names(), "no recipes discovered under cookbook/recipes/"


_COOKBOOK = Path(__file__).resolve().parent.parent / "cookbook"


def _app_names() -> list[str]:
    apps = _COOKBOOK / "apps"
    return sorted(p.parent.name for p in apps.glob("*/__main__.py")) if apps.exists() else []


def test_apps_package_importable():
    import cookbook.apps  # noqa: F401

    # Wave 1 ships no apps yet; later waves add them. The discovery helper must work.
    assert isinstance(_app_names(), list)
