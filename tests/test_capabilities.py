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
