"""Recipe: human-in-the-loop — gate a sensitive tool on approval.

``ToolApprovalPlugin`` runs an approver before a guarded tool. A denial
short-circuits the tool with a message the model adapts to, instead of acting.

    python -m cookbook.recipes.hitl
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, Runner, tool
from yaab.governance import ToolApprovalPlugin
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.types import ToolCall

_ran = {"wired": False}


@tool
def wire_transfer(amount: int, to: str) -> str:
    """Wire money to a recipient."""
    _ran["wired"] = True
    return f"sent ${amount} to {to}"


def _model() -> FunctionModel:
    # Turn 1: ask to wire $5000. Turn 2 (after the denial): back off politely.
    calls = {"n": 0}

    def fn(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(name="wire_transfer", arguments={"amount": 5000, "to": "ACME"})
                ]
            )
        return ModelResponse(content="I can't complete that transfer without approval.")

    return FunctionModel(fn)


async def run() -> dict:
    _ran["wired"] = False

    async def approver(tool_name, args, ctx) -> bool:
        return args.get("amount", 0) < 1000  # deny large transfers

    plugin = ToolApprovalPlugin(tools=["wire_transfer"], approver=approver)
    agent: Agent = Agent(
        "banker",
        model=_model(),
        instructions="Help with transfers.",
        tools=[wire_transfer],
        runner=Runner(plugins=[plugin]),
    )

    result = await agent.run("Wire $5000 to ACME.")
    expect(not _ran["wired"], "the denied transfer must NOT have executed")
    expect("approval" in str(result.output).lower(), "the model should report it needs approval")
    return {"executed": _ran["wired"], "answer": str(result.output)}


if __name__ == "__main__":
    print(asyncio.run(run()))
