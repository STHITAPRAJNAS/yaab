"""Sandboxed Python execution tool.

Runs a snippet in a **separate subprocess** with a wall-clock timeout and no
inherited globals, capturing stdout. This isolates crashes/hangs from the agent
process. It is NOT a full security sandbox (a determined attacker can still abuse
a Python subprocess) — gate it behind tool authorization / approval for
untrusted input, and run the agent itself in a container for real isolation.
"""

from __future__ import annotations

import subprocess
import sys

from ..base import tool

_PREAMBLE = (
    "import builtins, math, json, statistics, re\n"
    "import sys as _sys\n"
)


@tool
async def python_exec(code: str, timeout_seconds: float = 5.0) -> str:
    """Execute a short Python snippet and return its stdout.

    Runs in an isolated subprocess with a timeout. Use ``print(...)`` to produce
    output. Intended for calculation/data-wrangling, not system access — gate it
    behind tool approval for untrusted callers.
    """
    program = _PREAMBLE + code
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"error: execution exceeded {timeout_seconds}s timeout"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        return f"error: {err[-1] if err else 'non-zero exit'}"
    out = proc.stdout.strip()
    return out if out else "(no output)"
