"""Coding helper: an agent that runs Python in a sandbox, gated by approval.

Pattern: a tool-using agent where the tool has side effects / risk, so it runs
in an isolated sandbox AND requires approval before execution. Shows the
defense-in-depth combo (sandbox + ToolApprovalPlugin).
"""

from __future__ import annotations

from typing import Any

from yaab import Agent, Runner
from yaab.governance import ToolApprovalPlugin
from yaab.models.test_model import TestModel
from yaab.tools.builtin import python_exec

from .._common import resolve_model


def build(model: Any = None, *, auto_approve: bool = True) -> tuple[Agent, Runner]:
    """Return an agent with python_exec, gated by a human-approval plugin.

    Offline, the model is scripted to call python_exec then report the result.
    ``auto_approve`` simulates the human approving; set False to see a rejection.
    """
    offline = TestModel(custom_output="The sum is 45.", call_tools=["python_exec"])
    agent = Agent(
        "coding-helper",
        model=resolve_model(model, offline_default=offline),
        instructions="Use python_exec to compute answers. Show the result.",
        tools=[python_exec],
    )

    async def approver(tool: str, args: dict, ctx: Any) -> bool:
        return auto_approve

    runner = Runner(plugins=[ToolApprovalPlugin(tools=["python_exec"], approver=approver)])
    return agent, runner


async def run(task: str = "Compute the sum of 0..9 in Python.", model: Any = None) -> str:
    agent, runner = build(model)
    result = await runner.run(agent, task)
    return result.output


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(run()))
