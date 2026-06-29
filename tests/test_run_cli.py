from __future__ import annotations

import json

import pytest

from yaab.capabilities import Capability, current_tool_capabilities
from yaab.exceptions import (
    ApprovalRequired,
    GovernanceError,
    MaxStepsExceeded,
    ModelError,
    OutputValidationError,
    PolicyViolation,
    RunCancelled,
    ToolError,
    UsageLimitExceeded,
)
from yaab.run_cli import (
    EXIT,
    build_envelope,
    build_usage_limits,
    caps_approver,
    exit_code_for,
    read_only_approver,
)
from yaab.types import RunContext, RunResult, Usage


# --- exit-code mapping --------------------------------------------------------
@pytest.mark.parametrize(
    "exc,code",
    [
        (ApprovalRequired("shell_exec", {}), EXIT["approval"]),
        (UsageLimitExceeded("over budget", limit="tokens"), EXIT["budget"]),
        (RunCancelled("timed out", reason="timeout"), EXIT["timeout"]),
        (RunCancelled("stopped", reason="cancelled"), EXIT["cancelled"]),
        (MaxStepsExceeded("too many"), EXIT["max_steps"]),
        (OutputValidationError("bad"), EXIT["output"]),
        (ModelError("provider down"), EXIT["model"]),
        (PolicyViolation("nope", scanner="pii", stage="input"), EXIT["policy"]),
        (GovernanceError("blocked"), EXIT["policy"]),
        (ToolError("tool blew up"), EXIT["error"]),
        (RuntimeError("unexpected"), EXIT["error"]),
    ],
)
def test_exit_code_for(exc, code):
    assert exit_code_for(exc) == code


def test_exit_codes_are_distinct_and_in_range():
    vals = list(EXIT.values())
    assert len(vals) == len(set(vals))
    assert all(0 <= v <= 10 for v in vals)


# --- usage limits -------------------------------------------------------------
def test_build_usage_limits_applies_flags():
    lim = build_usage_limits(budget_tokens=1000, max_tool_calls=5, max_wall_seconds=30)
    assert lim.max_total_tokens == 1000
    assert lim.max_tool_calls == 5
    assert lim.max_wall_seconds == 30


def test_build_usage_limits_defaults_to_base():
    base = build_usage_limits(budget_tokens=42)
    # Unset flags fall through to a sane default rather than unbounded.
    assert base.max_total_tokens == 42
    assert base.max_requests is not None


# --- approver policies --------------------------------------------------------
@pytest.mark.asyncio
async def test_read_only_approver_denies_everything():
    approve = read_only_approver()
    assert await approve("file_write", {}, RunContext()) is False


@pytest.mark.asyncio
async def test_caps_approver_allows_only_listed_capabilities():
    approve = caps_approver({Capability.FS_WRITE_IN_ROOT})
    ctx = RunContext()
    tok = current_tool_capabilities.set(frozenset({Capability.FS_WRITE_IN_ROOT}))
    try:
        assert await approve("file_write", {}, ctx) is True
    finally:
        current_tool_capabilities.reset(tok)
    tok = current_tool_capabilities.set(frozenset({Capability.PROCESS_SPAWN}))
    try:
        assert await approve("shell_exec", {}, ctx) is False
    finally:
        current_tool_capabilities.reset(tok)


# --- json envelope ------------------------------------------------------------
def test_envelope_for_completed_run():
    result = RunResult(output="done", status="ok", usage=Usage(total_tokens=12))
    env = build_envelope(result, resume_id="r1")
    assert env["status"] == "ok"
    assert env["output"] == "done"
    assert env["paused"] is False
    assert env["resume_id"] == "r1"
    assert env["usage"]["total_tokens"] == 12
    json.dumps(env)  # must be serializable


def test_envelope_for_paused_run():
    result = RunResult(output=None, status="ok", paused=True, pause_value={"tool": "shell_exec"})
    env = build_envelope(result, resume_id="r2")
    assert env["paused"] is True
    assert env["status"] == "paused"
    assert env["pause"] == {"tool": "shell_exec"}
    assert env["output"] is None


# --- end-to-end through main(["run", ...]) (offline via TestModel) ------------
def test_run_usage_error_when_no_spec_and_no_coding(capsys):
    from yaab.cli import main

    code = main(["run"])
    assert code == EXIT["usage"]
    assert "spec" in capsys.readouterr().err.lower()


def test_run_coding_agent_end_to_end(tmp_path, monkeypatch, capsys):
    # Point the bundled coder at a TestModel so the run is fully offline, and run
    # it through the real CLI dispatch (main -> run_command -> agent.run).
    import yaab.harness as harness
    from yaab.cli import main
    from yaab.testing import TestModel

    real_coding_agent = harness.coding_agent

    def _offline(**kwargs):
        kwargs["model"] = TestModel("all done")
        return real_coding_agent(**kwargs)

    # _resolve_agent does `from .harness import coding_agent` per call, so patching
    # the module attribute is what the lazy import re-reads.
    monkeypatch.setattr(harness, "coding_agent", _offline)

    code = main(["run", "--coding", "--root", str(tmp_path), "--json", "say hi"])
    out = capsys.readouterr().out
    assert code == EXIT["ok"]
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["output"] == "all done"
    assert payload["paused"] is False
