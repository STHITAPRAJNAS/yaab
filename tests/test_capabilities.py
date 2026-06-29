from __future__ import annotations

import pytest

from yaab.capabilities import Capability
from yaab.tools.base import FunctionTool, tool


def test_capability_values():
    assert Capability.PROCESS_SPAWN.value == "process_spawn"
    assert {Capability.FS_READ, Capability.NET_EGRESS} != {Capability.FS_READ}


def test_function_tool_default_capabilities_empty():
    def f(x: int) -> int:
        """double"""
        return x * 2

    t = FunctionTool(f)
    assert t.capabilities == frozenset()


def test_function_tool_capabilities_passthrough():
    def f() -> str:
        """noop"""
        return "ok"

    t = FunctionTool(f, capabilities={Capability.NET_EGRESS})
    assert t.capabilities == frozenset({Capability.NET_EGRESS})


def test_tool_decorator_capabilities():
    @tool(capabilities={Capability.FS_WRITE_IN_ROOT})
    def w(path: str) -> str:
        """write"""
        return "ok"

    assert Capability.FS_WRITE_IN_ROOT in w.capabilities


def test_builtin_tools_declare_capabilities():
    from yaab.tools.builtin import (
        fetch_url,
        http_get,
        make_file_tools,
        python_exec,
        web_search,
    )

    assert Capability.NET_EGRESS in http_get.capabilities
    assert Capability.NET_EGRESS in fetch_url.capabilities
    assert Capability.NET_EGRESS in web_search.capabilities
    assert Capability.PROCESS_SPAWN in python_exec.capabilities
    read, write, _list = make_file_tools(root=".")
    assert Capability.FS_READ in read.capabilities
    assert Capability.FS_WRITE_IN_ROOT in write.capabilities


@pytest.mark.asyncio
async def test_capability_gate_blocks_renamed_action():
    # A tool named innocuously but carrying NET_EGRESS must still be gated —
    # the model can't bypass the gate by routing the action through a benign name.
    from yaab import Agent, Runner, tool
    from yaab.governance import ToolApprovalPlugin
    from yaab.models.test_model import TestModel

    ran = {"called": False}

    @tool(name="summarize", capabilities={Capability.NET_EGRESS})
    def summarize(url: str) -> str:
        """Looks harmless; actually egresses."""
        ran["called"] = True
        return "fetched"

    async def deny(tool_name, args, ctx) -> bool:
        return False  # deny all egress

    plugin = ToolApprovalPlugin(gate_capabilities={Capability.NET_EGRESS}, approver=deny)
    agent: Agent = Agent(
        "a",
        model=TestModel(call_tools=["summarize"], custom_output="done"),
        tools=[summarize],
        runner=Runner(plugins=[plugin]),
    )
    await agent.run("summarize http://x")
    assert ran["called"] is False  # gated by capability despite the benign name


@pytest.mark.asyncio
async def test_capability_gate_holds_under_parallel_tools():
    # Two tools called in ONE turn (parallel dispatch): the egress one must be
    # denied and the safe one allowed — the gate must not race on shared state.
    from yaab import Agent, Runner, tool
    from yaab.governance import ToolApprovalPlugin
    from yaab.models.base import ModelResponse
    from yaab.models.test_model import FunctionModel
    from yaab.types import ToolCall

    ran = {"egress": False, "safe": False}

    @tool(name="exfiltrate", capabilities={Capability.NET_EGRESS})
    def exfiltrate(data: str) -> str:
        """egress"""
        ran["egress"] = True
        return "sent"

    @tool(name="add", capabilities=set())
    def add(a: int, b: int) -> str:
        """pure"""
        ran["safe"] = True
        return "3"

    calls = {"n": 0}

    def model_fn(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(name="exfiltrate", arguments={"data": "secret"}),
                    ToolCall(name="add", arguments={"a": 1, "b": 2}),
                ]
            )
        return ModelResponse(content="done")

    async def deny(tool_name, args, ctx) -> bool:
        return False

    plugin = ToolApprovalPlugin(gate_capabilities={Capability.NET_EGRESS}, approver=deny)
    agent: Agent = Agent(
        "a", model=FunctionModel(model_fn), tools=[exfiltrate, add], runner=Runner(plugins=[plugin])
    )
    await agent.run("go")
    assert ran["egress"] is False  # egress denied even under concurrent dispatch
    assert ran["safe"] is True  # the non-destructive tool still ran
