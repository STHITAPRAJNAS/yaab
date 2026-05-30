"""Code-execution sandboxes for the ``python_exec`` tool.

A :class:`Sandbox` runs a Python snippet and returns stdout. Two backends ship:

* :class:`SubprocessSandbox` — isolated subprocess + timeout (default). Fast and
  dependency-free, but **not** a security boundary: it limits crashes/hangs, not
  a determined attacker.
* :class:`DockerSandbox` — runs the snippet in a throwaway container with no
  network, a read-only root, and CPU/memory/time caps — a real isolation
  boundary for untrusted code. Requires Docker on the host.

Select the backend when constructing the tool with
:func:`~yaab.tools.builtin.code.make_python_exec`, or set a global default with
:func:`set_default_sandbox`.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional, Protocol, runtime_checkable

_PREAMBLE = "import builtins, math, json, statistics, re\nimport sys as _sys\n"


@runtime_checkable
class Sandbox(Protocol):
    async def run(self, code: str, *, timeout: float) -> str:
        ...


class SubprocessSandbox:
    """Run code in an isolated subprocess (default; not a security boundary)."""

    async def run(self, code: str, *, timeout: float) -> str:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _PREAMBLE + code],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"error: execution exceeded {timeout}s timeout"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            return f"error: {err[-1] if err else 'non-zero exit'}"
        return proc.stdout.strip() or "(no output)"


class DockerSandbox:
    """Run code in a locked-down throwaway container (real isolation).

    Defaults: no network, read-only root fs, dropped capabilities, and
    memory/CPU/time limits. Requires a Docker daemon and the chosen image.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        memory: str = "256m",
        cpus: str = "1.0",
        network: bool = False,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network

    async def run(self, code: str, *, timeout: float) -> str:
        cmd = [
            "docker", "run", "--rm", "-i",
            "--memory", self.memory, "--cpus", self.cpus,
            "--read-only", "--cap-drop", "ALL", "--pids-limit", "64",
        ]
        if not self.network:
            cmd += ["--network", "none"]
        cmd += [self.image, "python", "-I", "-c", _PREAMBLE + code]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            return f"error: execution exceeded {timeout}s timeout"
        except FileNotFoundError:
            return "error: docker is not available on this host"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            return f"error: {err[-1] if err else 'non-zero exit'}"
        return proc.stdout.strip() or "(no output)"


_default_sandbox: Optional[Sandbox] = None


def set_default_sandbox(sandbox: Optional[Sandbox]) -> None:
    """Set the sandbox used by the built-in ``python_exec`` tool."""
    global _default_sandbox
    _default_sandbox = sandbox


def get_default_sandbox() -> Sandbox:
    return _default_sandbox or SubprocessSandbox()


__all__ = [
    "Sandbox",
    "SubprocessSandbox",
    "DockerSandbox",
    "set_default_sandbox",
    "get_default_sandbox",
]
