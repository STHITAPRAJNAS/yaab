from __future__ import annotations

import sys

import pytest

from yaab.capabilities import Capability
from yaab.tools.builtin.shell import ShellRule, make_shell_exec
from yaab.tools.exec import (
    SandboxNotIsolatedError,
    SubprocessCommandSandbox,
    require_isolated,
)

# A non-isolating sandbox is fine for these tests — we exercise the allowlist
# and argv plumbing, not the isolation boundary itself.
_SBX = SubprocessCommandSandbox()


def _shell(rules, **kw):
    return make_shell_exec(sandbox=_SBX, rules=rules, require_isolation=False, **kw)


def test_require_isolated_rejects_non_isolated():
    assert _SBX.is_isolated is False
    with pytest.raises(SandboxNotIsolatedError):
        require_isolated(_SBX)


def test_make_shell_exec_fails_closed_without_isolation():
    # Default require_isolation=True + a non-isolating sandbox -> refuse to build.
    with pytest.raises(SandboxNotIsolatedError):
        make_shell_exec(sandbox=_SBX, rules={"git": ShellRule()})


def test_shell_exec_has_process_spawn_capability():
    t = _shell({"git": ShellRule()})
    assert Capability.PROCESS_SPAWN in t.capabilities
    assert t.name == "shell_exec"


@pytest.mark.asyncio
async def test_binary_not_in_allowlist_is_rejected():
    t = _shell({"git": ShellRule()})
    out = await t.fn(command=["rm", "-rf", "/"])
    assert "error" in out.lower() and "allow" in out.lower()


@pytest.mark.asyncio
async def test_command_must_be_argv_list_not_string():
    t = _shell({"git": ShellRule()})
    out = await t.fn(command="git status")  # type: ignore[arg-type]
    assert "error" in out.lower()


@pytest.mark.asyncio
async def test_disallowed_subcommand_is_rejected():
    t = _shell({"git": ShellRule(subcommands=frozenset({"status", "diff"}))})
    out = await t.fn(command=["git", "push"])
    assert "error" in out.lower() and "subcommand" in out.lower()


@pytest.mark.asyncio
async def test_deny_substring_blocks_argument():
    t = _shell({"git": ShellRule(deny_substrings=("..",))})
    out = await t.fn(command=["git", "show", "../escape"])
    assert "error" in out.lower()


@pytest.mark.asyncio
async def test_allowed_command_runs_and_returns_output():
    # Allowlist the exact interpreter path so this is portable across OSes.
    t = _shell({sys.executable: ShellRule()})
    out = await t.fn(command=[sys.executable, "-c", "print('hello-shell')"])
    assert "hello-shell" in out


@pytest.mark.asyncio
async def test_nonzero_exit_is_reported_as_error():
    t = _shell({sys.executable: ShellRule()})
    out = await t.fn(command=[sys.executable, "-c", "import sys; sys.exit(3)"])
    assert "error" in out.lower() and "3" in out


@pytest.mark.asyncio
async def test_timeout_is_enforced():
    t = _shell({sys.executable: ShellRule()}, default_timeout=0.5)
    out = await t.fn(command=[sys.executable, "-c", "import time; time.sleep(5)"])
    assert "error" in out.lower() and "timeout" in out.lower()


@pytest.mark.asyncio
async def test_max_args_enforced():
    t = _shell({"git": ShellRule(max_args=1)})
    out = await t.fn(command=["git", "a", "b", "c"])
    assert "error" in out.lower()
