"""Coding agent — plan, then run code in a sandbox, gated for safety.

A ``SequentialAgent`` runs a **planner** then a **coder**; the coder runs Python
in the sandboxed ``python_exec`` tool, which is gated by ``ToolApprovalPlugin`` so
risky code can be reviewed. Offline it computes a small result deterministically.

Capabilities: **multi-agent** (planner → coder), **sandboxed execution**, and
**tool approval** (defense in depth).

    python -m cookbook.apps.coding_agent
"""

from __future__ import annotations

import asyncio
from typing import Any

from cookbook._harness import expect, resolve_model
from yaab import Agent, Runner, SequentialAgent
from yaab.governance import ToolApprovalPlugin
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel, TestModel
from yaab.tools.builtin.code import python_exec
from yaab.types import ToolCall

# Records what the sandbox was asked to run, for the assertions.
_audit: dict[str, str | None] = {"approved_code": None}


def _coder_model() -> FunctionModel:
    """Offline coder: run a small computation in the sandbox, then report it."""
    calls = {"n": 0}

    def fn(messages: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(name="python_exec", arguments={"code": "print(sum(range(11)))"})
                ]
            )
        return ModelResponse(content="The sum of 0..10 is 55.")

    return FunctionModel(fn)


def build(model: Any = None) -> SequentialAgent:
    """A planner → coder pipeline whose code execution is approval-gated."""

    async def approve_safe_code(tool_name: str, args: dict, ctx: Any) -> bool:
        # Approve code that only prints/computes; deny obvious file/network access.
        code = str(args.get("code", ""))
        _audit["approved_code"] = code
        return "import os" not in code and "open(" not in code

    approval = ToolApprovalPlugin(tools=["python_exec"], approver=approve_safe_code)

    planner: Agent = Agent(
        "planner",
        model=resolve_model(
            model, offline_default=TestModel(custom_output="Plan: compute the sum 0..10.")
        ),
        instructions="Write a short plan for the task.",
    )
    # The approval plugin lives on the coder's own runner, so its python_exec calls
    # are gated wherever the coder runs (including inside this pipeline).
    coder: Agent = Agent(
        "coder",
        model=resolve_model(model, offline_default=_coder_model()),
        instructions="Implement the plan by running python_exec, then report the result.",
        tools=[python_exec],
        runner=Runner(plugins=[approval]),
    )
    return SequentialAgent("dev", [planner, coder])


async def run() -> dict:
    _audit["approved_code"] = None
    pipeline = build()
    result = await pipeline.run("Compute the sum of 0 through 10.")
    answer = str(result.output)

    expect(_audit["approved_code"] is not None, "the coder should have proposed code")
    expect("55" in answer, f"expected the computed result 55, got {answer!r}")
    return {"answer": answer, "ran_code": _audit["approved_code"]}


if __name__ == "__main__":
    print(asyncio.run(run()))
