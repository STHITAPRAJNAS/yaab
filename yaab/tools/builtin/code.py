"""Sandboxed Python execution tool.

Delegates to a pluggable :class:`~yaab.tools.sandbox.Sandbox` backend. The
default :class:`SubprocessSandbox` isolates crashes/hangs (separate process +
timeout) but is NOT a security boundary — for untrusted code use
``set_default_sandbox(DockerSandbox(...))`` (real container isolation) and gate
the tool behind tool authorization / approval.
"""

from __future__ import annotations

from ..base import tool
from ..sandbox import Sandbox, get_default_sandbox


def make_python_exec(sandbox: Sandbox):
    """Build a ``python_exec`` tool bound to a specific sandbox backend."""

    @tool(name="python_exec")
    async def python_exec(code: str, timeout_seconds: float = 5.0) -> str:
        """Execute a short Python snippet and return its stdout.

        Use ``print(...)`` to produce output. Runs in the configured sandbox.
        """
        return await sandbox.run(code, timeout=timeout_seconds)

    return python_exec


@tool
async def python_exec(code: str, timeout_seconds: float = 5.0) -> str:
    """Execute a short Python snippet and return its stdout.

    Runs in the default sandbox (subprocess; swap with ``set_default_sandbox`` for
    container isolation). Use ``print(...)`` to produce output. Intended for
    calculation/data-wrangling — gate behind tool approval for untrusted callers.
    """
    return await get_default_sandbox().run(code, timeout=timeout_seconds)
