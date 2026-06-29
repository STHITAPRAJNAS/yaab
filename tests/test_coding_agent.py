from __future__ import annotations

import pytest

from yaab.capabilities import Capability
from yaab.governance.approval import ToolApprovalPlugin
from yaab.harness import coding_agent
from yaab.tools.base import tool
from yaab.tools.builtin.shell import ShellRule
from yaab.tools.exec import SandboxNotIsolatedError, SubprocessCommandSandbox
from yaab.types import RunContext


def _tool_names(agent):
    return {getattr(t, "name", None) for t in agent.tools}


def _gate(agent) -> ToolApprovalPlugin:
    plugins = agent._get_runner().plugins
    gates = [p for p in plugins if isinstance(p, ToolApprovalPlugin)]
    assert len(gates) == 1, f"expected exactly one approval gate, got {len(gates)}"
    return gates[0]


def test_default_agent_has_file_and_web_tools(tmp_path):
    agent = coding_agent(root=str(tmp_path))
    names = _tool_names(agent)
    assert {"file_read", "file_write", "file_list", "file_edit"} <= names
    assert {"web_search", "fetch_url"} <= names
    # Shell is off by default — process spawning is opt-in.
    assert "shell_exec" not in names


def test_default_gate_covers_writes_exec_and_net(tmp_path):
    gate = _gate(coding_agent(root=str(tmp_path)))
    for cap in (
        Capability.FS_WRITE_IN_ROOT,
        Capability.NET_EGRESS,
        Capability.PROCESS_SPAWN,
        Capability.FS_WRITE_OUT,
        Capability.SCHEDULE,
    ):
        assert gate.guards_capability(cap), cap
    # Reads are auto-allowed (not gated).
    assert not gate.guards_capability(Capability.FS_READ)


def test_no_inline_approver_means_gated_tools_pause_not_autorun(tmp_path):
    # Fail-safe default: no approver wired, so the gate raises (pauses) rather
    # than silently letting a destructive tool through.
    gate = _gate(coding_agent(root=str(tmp_path)))
    assert gate.approver is None


def test_enable_shell_without_isolation_fails_closed(tmp_path):
    with pytest.raises(SandboxNotIsolatedError):
        coding_agent(
            root=str(tmp_path),
            enable_shell=True,
            sandbox=SubprocessCommandSandbox(),
            shell_rules={"git": ShellRule()},
        )


def test_enable_shell_with_explicit_downgrade_adds_tool(tmp_path):
    agent = coding_agent(
        root=str(tmp_path),
        enable_shell=True,
        sandbox=SubprocessCommandSandbox(),
        shell_rules={"git": ShellRule(subcommands=frozenset({"status"}))},
        allow_unsandboxed_shell=True,
    )
    assert "shell_exec" in _tool_names(agent)


def test_fail_closed_when_gate_misses_a_destructive_cap(tmp_path):
    # Web tools carry NET_EGRESS; a gate that omits it must be rejected.
    with pytest.raises(ValueError, match="ungated|gate"):
        coding_agent(root=str(tmp_path), gate={Capability.FS_WRITE_IN_ROOT})


def test_fail_closed_when_gate_misses_in_root_write(tmp_path):
    # Regression: FS_WRITE_IN_ROOT is gate-worthy (in DEFAULT_GATE) but not in
    # DESTRUCTIVE; a custom gate that drops it must still fail closed, or
    # file_write/file_edit would run ungated.
    with pytest.raises(ValueError, match="fs_write_in_root|ungated"):
        coding_agent(root=str(tmp_path), allow_net=False, gate={Capability.PROCESS_SPAWN})


def test_agents_md_is_auto_loaded(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Project rule: prefer tabs.\n", encoding="utf-8")
    agent = coding_agent(root=str(tmp_path))
    assert "prefer tabs" in agent.instructions


def test_max_steps_is_bounded(tmp_path):
    agent = coding_agent(root=str(tmp_path), max_steps=12)
    assert agent.max_steps == 12


def test_mcp_tools_are_gated_by_name(tmp_path):
    @tool(name="notion_write")
    async def notion_write(text: str) -> str:
        return text

    agent = coding_agent(root=str(tmp_path), mcp_tools=[notion_write])
    assert "notion_write" in _tool_names(agent)
    gate = _gate(agent)
    # Unknown-capability MCP tool is gated by name (defense for opaque effects).
    assert gate._guarded("notion_write", {}, RunContext())
