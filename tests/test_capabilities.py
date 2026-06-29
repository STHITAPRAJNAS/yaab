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
