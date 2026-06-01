"""Tests for the ``yaab eval`` CLI command.

These call ``yaab.cli.main([...])`` directly (no subprocess) for the core
assertions — that keeps the run deterministic and lets us capture stdout and
exit codes cheaply. A single subprocess ``yaab info`` smoke test guards the
console-script wiring. The agent under test is written to a temp module that
uses :class:`~yaab.models.test_model.TestModel`, so scores are deterministic
and no API keys / network are needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from yaab.cli import main

# Monotonic so every temp agent module gets a unique import name within the test
# session — no cross-test sys.modules collisions, no dependence on hash salting.
_MODULE_COUNTER = 0


def _write_agent_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> str:
    """Write a temp agent module on sys.path and return its import name."""
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    module_name = f"_eval_agent_{_MODULE_COUNTER}"
    (tmp_path / f"{module_name}.py").write_text(body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    # Belt-and-suspenders: drop any stale import so we re-import from this tmp dir.
    sys.modules.pop(module_name, None)
    return module_name


def _write_evalset(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "suite.evalset.json"
    body = {
        "schema_version": 1,
        "name": "smoke",
        "version": "1",
        "cases": cases,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# An agent that always echoes a fixed string, so exact_match is deterministic.
_ECHO_AGENT = """
from yaab import Agent
from yaab.models.test_model import TestModel

agent = Agent("echo", model=TestModel("PARIS"), instructions="x")
"""


def test_eval_passes_and_reports_exact_match(tmp_path, monkeypatch, capsys):
    module = _write_agent_module(tmp_path, monkeypatch, _ECHO_AGENT)
    evalset = _write_evalset(
        tmp_path,
        [
            {"id": "c1", "conversation": ["capital of France?"], "expected_output": "PARIS"},
            {"id": "c2", "conversation": ["again?"], "expected_output": "PARIS"},
        ],
    )
    report = tmp_path / "report.json"

    code = main(["eval", f"{module}:agent", str(evalset), "--output", str(report)])

    assert code == 0
    out = capsys.readouterr().out
    # Per-case rows mention each case id and its score.
    assert "c1" in out
    assert "c2" in out
    # Summary line carries the per-metric mean.
    assert "exact_match" in out
    # A perfect run reports a 1.00 mean.
    assert "1.00" in out

    # The JSON report round-trips and records per-case + aggregate scores.
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["evalset"] == "smoke"
    assert data["aggregate"]["exact_match"] == 1.0
    case_ids = {c["case"] for c in data["cases"]}
    assert case_ids == {"c1", "c2"}
    for c in data["cases"]:
        assert c["scores"]["exact_match"] == 1.0


def test_eval_fail_under_gates_ci(tmp_path, monkeypatch, capsys):
    # The agent echoes "PARIS" but the expectation is "LONDON" -> score 0.
    module = _write_agent_module(tmp_path, monkeypatch, _ECHO_AGENT)
    evalset = _write_evalset(
        tmp_path,
        [{"id": "wrong", "conversation": ["q?"], "expected_output": "LONDON"}],
    )

    # Without a gate, a failing score still exits 0 (it merely reports).
    assert main(["eval", f"{module}:agent", str(evalset)]) == 0

    # With --fail-under above the achieved mean (0.0), the gate fails -> exit 1.
    code = main(["eval", f"{module}:agent", str(evalset), "--fail-under", "0.8"])
    assert code == 1
    out = capsys.readouterr().out
    assert "0.00" in out


def test_eval_passes_fail_under_when_above_threshold(tmp_path, monkeypatch):
    module = _write_agent_module(tmp_path, monkeypatch, _ECHO_AGENT)
    evalset = _write_evalset(
        tmp_path,
        [{"id": "ok", "conversation": ["q?"], "expected_output": "PARIS"}],
    )
    # mean 1.0 >= 0.8 -> gate passes.
    assert main(["eval", f"{module}:agent", str(evalset), "--fail-under", "0.8"]) == 0


# An agent that calls a tool once (so a tool trajectory is produced) and then
# answers. TestModel(call_tools=[...]) drives the tool loop deterministically.
_TOOL_AGENT = '''
from yaab import Agent, tool
from yaab.models.test_model import TestModel


@tool
def lookup(q: str = "") -> str:
    """Look something up."""
    return "looked up"


agent = Agent(
    "tooler",
    model=TestModel("done", call_tools=["lookup"]),
    tools=[lookup],
    instructions="x",
)
'''


def test_eval_uses_tool_trajectory_metric(tmp_path, monkeypatch, capsys):
    module = _write_agent_module(tmp_path, monkeypatch, _TOOL_AGENT)
    evalset = _write_evalset(
        tmp_path,
        [
            {
                "id": "traj",
                "conversation": ["do it"],
                "expected_tool_trajectory": [{"name": "lookup"}],
            }
        ],
    )
    report = tmp_path / "report.json"

    code = main(["eval", f"{module}:agent", str(evalset), "--output", str(report)])

    assert code == 0
    out = capsys.readouterr().out
    # The trajectory metric is auto-selected for cases with an expected trajectory.
    assert "tool_trajectory" in out

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["aggregate"]["tool_trajectory"] == 1.0
    case = data["cases"][0]
    assert case["scores"]["tool_trajectory"] == 1.0


def test_eval_explicit_metric_overrides_autodetect(tmp_path, monkeypatch, capsys):
    module = _write_agent_module(tmp_path, monkeypatch, _ECHO_AGENT)
    evalset = _write_evalset(
        tmp_path,
        [{"id": "c1", "conversation": ["q?"], "expected_output": "PARIS"}],
    )
    report = tmp_path / "report.json"

    # Force the "contains" metric explicitly; output "PARIS" contains expected "PARIS".
    code = main(
        [
            "eval",
            f"{module}:agent",
            str(evalset),
            "--metric",
            "contains",
            "--output",
            str(report),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "contains" in out
    # exact_match was NOT auto-selected because an explicit metric was given.
    assert "exact_match" not in out

    data = json.loads(report.read_text(encoding="utf-8"))
    assert "contains" in data["aggregate"]
    assert "exact_match" not in data["aggregate"]


def test_eval_multiple_explicit_metrics(tmp_path, monkeypatch, capsys):
    module = _write_agent_module(tmp_path, monkeypatch, _ECHO_AGENT)
    evalset = _write_evalset(
        tmp_path,
        [{"id": "c1", "conversation": ["q?"], "expected_output": "PARIS"}],
    )
    code = main(
        [
            "eval",
            f"{module}:agent",
            str(evalset),
            "--metric",
            "exact_match",
            "--metric",
            "contains",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "exact_match" in out
    assert "contains" in out


def test_info_smoke_via_subprocess():
    """One real-process smoke test that the console wiring runs end to end."""
    proc = subprocess.run(
        [sys.executable, "-m", "yaab.cli", "info"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "performance backend" in proc.stdout
