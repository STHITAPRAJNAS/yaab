"""The yaab coding harness — a packaged, automatable agent like a CLI coding tool.

:func:`coding_agent` assembles a *plain* :class:`~yaab.agent.Agent` wired for
autonomous, headless software work: sandboxed file read/write/edit, optional
allowlisted shell, optional web access, an ``AGENTS.md`` project brief, a bounded
loop, and — crucially — a capability **approval gate** that is on by default.

It is deliberately not a god-object: the return value is an ordinary ``Agent`` you
can run, embed, sub-agent, or extend. All the safety lives in how it is wired:

* **Gated by capability, not name.** Writes, process spawns, network egress,
  scheduling and env reads are routed through a
  :class:`~yaab.governance.approval.ToolApprovalPlugin`. Reads are auto-allowed.
* **Fail-closed.** If any tool carries a destructive capability the gate does not
  cover, construction raises rather than shipping an ungated destructive tool.
* **No silent auto-approve.** With no ``approver`` wired, a gated tool *pauses*
  the run (surfaces for out-of-band approval); it never runs unreviewed.
* **Shell is opt-in and isolation-checked.** ``enable_shell=True`` requires an
  isolating sandbox unless you *explicitly* downgrade.

Token/wall-clock budgets are applied at the run boundary (the ``yaab run`` CLI
always passes :func:`default_usage_limits`); the loop itself is bounded by
``max_steps`` here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent import Agent
from ..capabilities import DESTRUCTIVE, Capability
from ..governance.approval import ToolApprovalPlugin
from ..limits import UsageLimits
from ..tools.builtin.files import file_toolset
from ..tools.builtin.search import web_search
from ..tools.builtin.shell import ShellRule, make_shell_exec
from ..tools.builtin.url_context import fetch_url
from ..tools.exec import CommandSandbox

#: Default gated set: everything an agent does that is destructive, exfiltrating,
#: or mutating — plus in-root writes and env reads. Only plain reads (``FS_READ``)
#: are auto-allowed. Override with ``gate=`` (still fail-closed-checked).
DEFAULT_GATE: frozenset[Capability] = frozenset(
    DESTRUCTIVE | {Capability.FS_WRITE_IN_ROOT, Capability.ENV_READ}
)

_DEFAULT_INSTRUCTIONS = (
    "You are a coding agent working inside a sandboxed project root. Read before "
    "you edit, make the smallest change that solves the task, and prefer file_edit "
    "(surgical) over file_write (whole-file) for existing files. Explain what you "
    "changed and why. You cannot escape the root, and write/exec/network actions "
    "may require human approval — if a tool reports it was not approved, stop and "
    "ask rather than working around it."
)


def default_usage_limits() -> UsageLimits:
    """A conservative default budget for a headless coding run.

    Applied by the ``yaab run`` CLI on every run; library callers can pass it (or
    their own) to :meth:`Agent.run`. Bounds requests, tokens, and wall-clock so a
    runaway loop cannot burn unbounded cost.
    """
    return UsageLimits(
        max_requests=60,
        max_total_tokens=500_000,
        max_tool_calls=120,
        max_wall_seconds=900,
    )


def coding_agent(
    *,
    root: str,
    model: Any = "openai/gpt-4o",
    name: str = "coder",
    instructions: str | None = None,
    allow_net: bool = True,
    enable_shell: bool = False,
    sandbox: CommandSandbox | None = None,
    shell_rules: dict[str, ShellRule] | None = None,
    allow_unsandboxed_shell: bool = False,
    gate: set[Capability] | frozenset[Capability] | None = None,
    approver: Any | None = None,
    audit: Any | None = None,
    extra_tools: list[Any] | None = None,
    mcp_tools: list[Any] | None = None,
    max_steps: int = 30,
    checkpointer: Any | None = None,
) -> Agent:
    """Build a sandboxed, approval-gated coding :class:`~yaab.agent.Agent`.

    Args:
        root: Project directory the agent is confined to (file tools, AGENTS.md,
            and the shell working directory all resolve against it).
        model: LiteLLM model spec (or a ``ModelProvider``) — provider-neutral.
        allow_net: Include ``web_search`` / ``fetch_url`` (gated by NET_EGRESS).
        enable_shell: Add ``shell_exec``. Requires ``sandbox`` and (unless
            ``allow_unsandboxed_shell``) an *isolating* one.
        sandbox: Command sandbox backing ``shell_exec``.
        shell_rules: ``{binary: ShellRule}`` allowlist for ``shell_exec`` (empty =
            nothing runs).
        allow_unsandboxed_shell: Explicitly accept a non-isolating sandbox for
            shell (an informed downgrade — never the default).
        gate: Capabilities requiring approval. Defaults to :data:`DEFAULT_GATE`.
        approver: Optional inline approver. ``None`` ⇒ gated tools pause the run.
        audit: Optional audit log passed to the approval gate.
        extra_tools: Extra developer tools (carry their own capabilities).
        mcp_tools: External/MCP tools — gated by name regardless of declared caps.
        max_steps: Hard cap on agent loop iterations.
        checkpointer: Optional run checkpointer. Defaults to an in-process
            ``MemorySaver`` (resume works within one process); pass a
            ``SQLiteSaver`` for resume that survives across ``yaab run`` invocations.

    Raises:
        ValueError: if a destructive-capability tool is left ungated (fail-closed),
            or ``enable_shell`` is set without a ``sandbox``.
        SandboxNotIsolatedError: ``enable_shell`` on a non-isolating sandbox
            without ``allow_unsandboxed_shell``.
    """
    tools: list[Any] = list(file_toolset(root=root))
    if allow_net:
        tools += [web_search, fetch_url]
    if enable_shell:
        if sandbox is None:
            raise ValueError("enable_shell=True requires a sandbox=")
        tools.append(
            make_shell_exec(
                sandbox=sandbox,
                rules=shell_rules or {},
                require_isolation=not allow_unsandboxed_shell,
                cwd=root,
            )
        )
    tools += list(extra_tools or [])
    mcp = list(mcp_tools or [])
    tools += mcp
    mcp_names = [n for n in (getattr(t, "name", None) for t in mcp) if n]

    gate_caps = frozenset(gate) if gate is not None else DEFAULT_GATE
    _assert_no_ungated_gateworthy(tools, gate_caps, name_gated=set(mcp_names))

    plugin = ToolApprovalPlugin(
        gate_capabilities=gate_caps,
        tools=mcp_names or None,
        approver=approver,
        audit=audit,
    )

    base = _DEFAULT_INSTRUCTIONS if instructions is None else instructions
    base = _with_agents_md(base, root)

    if checkpointer is None:
        # hitl= sugar wires a gated, durable-in-memory Runner (MemorySaver).
        return Agent(
            name, model=model, instructions=base, tools=tools, max_steps=max_steps, hitl=plugin
        )
    # Power form: an explicit Runner so resume can be durable across processes.
    from ..runner import Runner

    runner = Runner(plugins=[plugin], run_checkpointer=checkpointer)
    return Agent(
        name, model=model, instructions=base, tools=tools, max_steps=max_steps, runner=runner
    )


#: Capabilities the harness considers must-be-gated. This is exactly the default
#: gated set: every destructive/exfiltrating cap PLUS in-root writes and env reads.
#: The fail-closed check masks against this (not just DESTRUCTIVE) so a custom
#: ``gate=`` that drops ``FS_WRITE_IN_ROOT``/``ENV_READ`` is still rejected — only
#: plain reads (``FS_READ``) are ever allowed to run ungated.
GATEWORTHY: frozenset[Capability] = DEFAULT_GATE


def _assert_no_ungated_gateworthy(
    tools: list[Any], gate_caps: frozenset[Capability], *, name_gated: set[str]
) -> None:
    """Fail-closed: refuse to build if a gate-worthy tool is neither capability-
    gated nor name-gated."""
    ungated: set[Capability] = set()
    for t in tools:
        if getattr(t, "name", None) in name_gated:
            continue  # gated by name
        caps = set(getattr(t, "capabilities", frozenset())) & GATEWORTHY
        ungated |= caps - set(gate_caps)
    if ungated:
        missing = ", ".join(sorted(c.value for c in ungated))
        raise ValueError(
            f"fail-closed: gate-worthy capabilities are ungated: {missing}. "
            "Widen `gate=` to cover them or remove the offending tool."
        )


def _with_agents_md(instructions: str, root: str) -> str:
    """Append ``{root}/AGENTS.md`` (the project brief) to the instructions if present."""
    md = Path(root) / "AGENTS.md"
    if not md.is_file():
        return instructions
    try:
        brief = md.read_bytes()[:20_000].decode("utf-8", "replace")
    except OSError:
        return instructions
    if not brief.strip():
        return instructions
    return f"{instructions}\n\n# Project guidance (AGENTS.md)\n\n{brief}"


__all__ = ["coding_agent", "default_usage_limits", "DEFAULT_GATE"]
