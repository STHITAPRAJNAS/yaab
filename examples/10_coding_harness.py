"""The packaged coding harness: a sandboxed, approval-gated coding agent.

``coding_agent()`` returns a plain Agent wired with sandboxed file tools
(including surgical ``file_edit``), a bounded loop, and a capability approval
gate. This example shows the three behaviors that make it safe by construction:

1. With in-root writes approved, it reads a file and makes a surgical edit.
2. With no approver, the same gated write PAUSES the run instead of running.
3. Fail-closed: a gate that leaves a destructive capability uncovered refuses to
   build at all.

Runs fully offline with a scripted model. Set ``YAAB_SAMPLE_MODEL`` to let a real
model choose the tool calls itself.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from yaab import coding_agent
from yaab.capabilities import Capability
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.run_cli import caps_approver
from yaab.types import ToolCall


def _scripted_editor() -> FunctionModel:
    """Offline coder: read m.py, then edit "1" -> "2", then report."""
    calls = {"n": 0}

    def fn(messages: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                tool_calls=[ToolCall(name="file_read", arguments={"path": "m.py"})]
            )
        if calls["n"] == 2:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="file_edit",
                        arguments={"path": "m.py", "old_string": "x = 1", "new_string": "x = 2"},
                    )
                ]
            )
        return ModelResponse(content="Changed x from 1 to 2.")

    return FunctionModel(fn)


async def main() -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="yaab-harness-example-"))
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")

    # 1) Approve in-root writes (and nothing else) -> the surgical edit lands.
    approved = coding_agent(
        root=str(workspace),
        model=_scripted_editor(),
        allow_net=False,
        approver=caps_approver({Capability.FS_WRITE_IN_ROOT}),
    )
    answer = (await approved.run("change x to 2 in m.py")).output
    edited = (workspace / "m.py").read_text(encoding="utf-8")
    print("1) approved edit ->", edited.strip(), "|", answer)

    # 2) No approver: the gated write PAUSES (awaits human sign-off) instead of
    #    running. A resume id is what `yaab run` would print for you to resume.
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")  # reset
    gated = coding_agent(root=str(workspace), model=_scripted_editor(), allow_net=False)
    paused_result = await gated.run("change x to 2 in m.py", resume_id="demo-1")
    after_pause = (workspace / "m.py").read_text(encoding="utf-8")
    print(
        "2) no approver -> paused:", paused_result.paused, "| file unchanged:", after_pause.strip()
    )

    # 3) Fail-closed: a gate that omits a destructive capability the tools carry
    #    refuses to build, rather than shipping an ungated destructive tool.
    fail_closed = False
    try:
        coding_agent(root=str(workspace), allow_net=False, gate={Capability.PROCESS_SPAWN})
    except ValueError as exc:
        fail_closed = True
        print("3) fail-closed ->", str(exc).split(".")[0])

    return {
        "answer": str(answer),
        "edited": edited.strip(),
        "paused": bool(paused_result.paused),
        "file_after_pause": after_pause.strip(),
        "fail_closed": fail_closed,
    }


if __name__ == "__main__":
    print(asyncio.run(main()))
