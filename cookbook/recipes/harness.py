"""Recipe: the coding harness — a sandboxed, approval-gated coding agent.

``coding_agent()`` returns a plain ``Agent`` wired with sandboxed file tools
(including surgical ``file_edit``), a bounded loop, and a capability approval
gate. Here it reads a file and edits one line — the write is gated, so we wire a
capability-scoped approver that allows in-root writes (mirroring
``yaab run --approve-caps fs_write_in_root``).

Offline, a scripted ``FunctionModel`` drives the exact ``file_read`` → ``file_edit``
sequence. Set ``YAAB_SAMPLE_MODEL`` to run it against a real model, which will
choose those tool calls itself from the prompt::

    python -m cookbook.recipes.harness
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from cookbook._harness import expect, resolve_model
from yaab import coding_agent
from yaab.capabilities import Capability
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.run_cli import caps_approver
from yaab.types import ToolCall


def _scripted_coder() -> FunctionModel:
    """Offline coder: read greeting.py, then edit "hello" -> "hi", then report."""
    calls = {"n": 0}

    def fn(messages: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                tool_calls=[ToolCall(name="file_read", arguments={"path": "greeting.py"})]
            )
        if calls["n"] == 2:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="file_edit",
                        arguments={
                            "path": "greeting.py",
                            "old_string": '"hello"',
                            "new_string": '"hi"',
                        },
                    )
                ]
            )
        return ModelResponse(content="Changed the greeting from hello to hi.")

    return FunctionModel(fn)


async def run() -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="yaab-harness-recipe-"))
    target = workspace / "greeting.py"
    target.write_text('greeting = "hello"\n', encoding="utf-8")

    agent = coding_agent(
        root=str(workspace),
        model=resolve_model(offline_default=_scripted_coder()),
        allow_net=False,  # this task is local-only
        # Writes are gated by default; approve in-root writes (and nothing else).
        approver=caps_approver({Capability.FS_WRITE_IN_ROOT}),
    )

    result = await agent.run(
        'In greeting.py, change the greeting string from "hello" to "hi". '
        "Read the file first, then make a surgical edit."
    )

    edited = target.read_text(encoding="utf-8")
    expect('"hi"' in edited, f"expected the edit to land, file is {edited!r}")
    expect('"hello"' not in edited, f"old string should be gone, file is {edited!r}")
    return {"answer": str(result.output), "file": edited.strip()}


if __name__ == "__main__":
    print(asyncio.run(run()))
