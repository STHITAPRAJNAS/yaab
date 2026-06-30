"""Integration tests: the harness driven through the real model→runner→tool loop.

The unit tests cover each piece in isolation; these exercise the wired path — a
scripted model issues tool calls, the runner dispatches them, the capability
approval gate decides, and tools actually touch the sandbox. This is where the
factory's gating + file_edit's read-before-edit guard meet the runner for real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from yaab import coding_agent
from yaab.capabilities import Capability
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.run_cli import caps_approver
from yaab.types import ToolCall


def _read_then_edit_model() -> FunctionModel:
    """file_read('m.py') -> file_edit 1->2 -> final answer."""
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
        return ModelResponse(content="done")

    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_coding_agent_edits_a_file_through_the_loop(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    agent = coding_agent(
        root=str(tmp_path),
        model=_read_then_edit_model(),
        allow_net=False,
        approver=caps_approver({Capability.FS_WRITE_IN_ROOT}),  # approve in-root writes
    )
    result = await agent.run("change x to 2 in m.py")
    assert not result.paused
    assert result.output == "done"
    # The gated edit actually ran and the file changed on disk.
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "x = 2\n"


@pytest.mark.asyncio
async def test_gated_write_pauses_durably_when_no_approver(tmp_path):
    # No approver wired + a resume_id (what `yaab run` always passes): the gated
    # file_edit must DURABLY PAUSE (await human sign-off) rather than run, and the
    # file stays untouched.
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    agent = coding_agent(root=str(tmp_path), model=_read_then_edit_model(), allow_net=False)
    result = await agent.run("change x to 2 in m.py", resume_id="r1")
    assert result.paused is True
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "x = 1\n"  # unchanged


@pytest.mark.asyncio
async def test_gated_write_raises_without_resume_id(tmp_path):
    # Without a resume_id (no durable checkpoint to park into), the gate surfaces
    # as a raised ApprovalRequired — still never a silent run.
    from yaab.exceptions import ApprovalRequired

    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    agent = coding_agent(root=str(tmp_path), model=_read_then_edit_model(), allow_net=False)
    with pytest.raises(ApprovalRequired):
        await agent.run("change x to 2 in m.py")
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "x = 1\n"  # unchanged


@pytest.mark.asyncio
async def test_read_only_caps_deny_blocks_the_edit(tmp_path):
    # An approver scoped to a *different* capability denies the write; the edit
    # comes back as an error the model can see, and the file is unchanged.
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")

    async def deny_writes(tool: str, args: dict, ctx: Any) -> bool:
        return False

    agent = coding_agent(
        root=str(tmp_path), model=_read_then_edit_model(), allow_net=False, approver=deny_writes
    )
    result = await agent.run("change x to 2 in m.py")
    assert not result.paused
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "x = 1\n"  # denied, unchanged


def test_harness_recipe_is_registered():
    # The runnable example is discoverable by the cookbook test harness.
    import cookbook.recipes.harness as recipe

    assert callable(recipe.run)
    assert Path(recipe.__file__).name == "harness.py"
