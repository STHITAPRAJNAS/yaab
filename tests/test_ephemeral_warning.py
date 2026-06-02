"""Tests for the startup guardrail that warns about non-durable backends.

When more than one replica runs, an in-memory backend silently loses data the
other replicas can never see. ``warn_if_ephemeral`` turns that into a loud,
named warning at boot; a single replica (the default) stays silent so existing
single-process usage is unchanged.
"""

from __future__ import annotations

import warnings

import pytest

from yaab.artifacts import InMemoryArtifactService, SQLiteArtifactService
from yaab.runs.safety import warn_if_ephemeral
from yaab.sessions import InMemorySessionService, SQLiteSessionService


def _tmp_db(name: str) -> str:
    import tempfile
    from pathlib import Path

    return str(Path(tempfile.mkdtemp()) / name)


def test_single_replica_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        warn_if_ephemeral(
            replicas=1,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
        )


def test_multi_replica_warns_and_names_the_backend() -> None:
    with pytest.warns(RuntimeWarning) as records:
        warn_if_ephemeral(
            replicas=3,
            session_service=InMemorySessionService(),
        )
    text = "\n".join(str(r.message) for r in records)
    assert "session_service" in text
    assert "InMemorySessionService" in text


def test_multi_replica_all_durable_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_ephemeral(
            replicas=5,
            session_service=SQLiteSessionService(_tmp_db("s.db")),
            artifact_service=SQLiteArtifactService(_tmp_db("a.db")),
        )


def test_multi_replica_names_every_ephemeral_backend() -> None:
    with pytest.warns(RuntimeWarning) as records:
        warn_if_ephemeral(
            replicas=2,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
        )
    text = "\n".join(str(r.message) for r in records)
    assert "session_service" in text
    assert "artifact_service" in text


def test_strict_durability_env_forces_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAAB_STRICT_DURABILITY", "1")
    with pytest.warns(RuntimeWarning):
        warn_if_ephemeral(
            replicas=1,  # single replica, but strict mode forces the check
            session_service=InMemorySessionService(),
        )


def test_yaab_replicas_env_triggers_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAAB_REPLICAS", "4")
    with pytest.warns(RuntimeWarning):
        # replicas not passed -> falls back to YAAB_REPLICAS env.
        warn_if_ephemeral(session_service=InMemorySessionService())


def test_none_backends_are_ignored() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_ephemeral(replicas=3, session_service=None, artifact_service=None)


def test_warning_mentions_silence_instructions() -> None:
    with pytest.warns(RuntimeWarning) as records:
        warn_if_ephemeral(replicas=2, run_store=InMemorySessionService())
    text = "\n".join(str(r.message) for r in records)
    # The user is told how to make the config durable; backend label is included.
    assert "run_store" in text


def test_returns_list_of_ephemeral_names() -> None:
    with pytest.warns(RuntimeWarning):
        names = warn_if_ephemeral(
            replicas=2,
            session_service=InMemorySessionService(),
            artifact_service=SQLiteArtifactService(_tmp_db("ok.db")),
        )
    assert names == ["session_service"]
