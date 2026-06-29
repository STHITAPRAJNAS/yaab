"""Approved-resume path must still run non-approval pre-execution hooks.

Regression for the adversarial-review finding: ``_execute_approved_tool`` used to
execute a human-approved tool *directly*, skipping every plugin's ``before_tool``
(authorization, idempotency, rate-limit, audit) and never setting
``current_tool_capabilities``. Only the approval gate itself should be skipped on
resume — everything else must still run against the approved (possibly edited) args.
"""

from __future__ import annotations

from typing import Any

import pytest

from yaab import Agent, Runner
from yaab.capabilities import Capability, current_tool_capabilities
from yaab.governance.approval import ToolApprovalPlugin
from yaab.plugins import Plugin
from yaab.tools.base import tool
from yaab.types import RunContext


class _Recorder(Plugin):
    """Records before_tool invocations and the capabilities visible at that point;
    optionally short-circuits with a deny."""

    def __init__(self, *, deny: bool = False) -> None:
        self.before_calls: list[str] = []
        self.caps_seen: frozenset[Capability] = frozenset()
        self.deny = deny

    async def before_tool(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> Any:
        self.before_calls.append(tool)
        self.caps_seen = current_tool_capabilities.get()
        if self.deny:
            return f"error: tool '{tool}' denied by policy"
        return None


@tool(name="danger", capabilities={Capability.PROCESS_SPAWN})
async def danger(x: int = 1) -> str:
    return f"ran:{x}"


def _agent() -> Agent:
    return Agent("a", model="openai/gpt-4o", tools=[danger])


@pytest.mark.asyncio
async def test_approved_resume_runs_non_approval_before_tool_and_sets_caps():
    rec = _Recorder()
    runner = Runner(plugins=[rec, ToolApprovalPlugin(gate_capabilities={Capability.PROCESS_SPAWN})])
    result = await runner._execute_approved_tool(_agent(), RunContext(), "danger", {"x": 7})
    # The non-approval plugin's before_tool ran (it was previously skipped entirely)
    assert rec.before_calls == ["danger"]
    # ...and it saw the tool's capabilities (previously an empty set on resume).
    assert Capability.PROCESS_SPAWN in rec.caps_seen
    # The approval gate did NOT re-park the run; the tool executed.
    assert result == "ran:7"


@pytest.mark.asyncio
async def test_approved_resume_honors_a_pre_hook_deny():
    rec = _Recorder(deny=True)
    runner = Runner(plugins=[rec, ToolApprovalPlugin(gate_capabilities={Capability.PROCESS_SPAWN})])
    result = await runner._execute_approved_tool(_agent(), RunContext(), "danger", {"x": 7})
    # An authorization-style deny on the approved path short-circuits execution.
    assert result == "error: tool 'danger' denied by policy"
