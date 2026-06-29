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

import os
import signal
import subprocess
import sys
from typing import Protocol, runtime_checkable

_PREAMBLE = "import builtins, math, json, statistics, re\nimport sys as _sys\n"

#: Environment variables the sandboxed child is allowed to see. Everything else
#: (every ``*_API_KEY``, cloud credential, DB URL in ``os.environ``) is dropped
#: so executed code cannot read secrets out of the parent process.
_SAFE_ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
)


def _minimal_env() -> dict[str, str]:
    """A scrubbed environment — no inherited secrets."""
    return {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}


def _set_rlimits() -> None:  # pragma: no cover - runs in the child, POSIX only
    if sys.platform == "win32":
        return
    import resource

    for res, soft in (
        (resource.RLIMIT_CPU, 60),
        (resource.RLIMIT_NPROC, 64),
        (resource.RLIMIT_FSIZE, 64 * 1024 * 1024),
    ):
        try:
            resource.setrlimit(res, (soft, soft))
        except (ValueError, OSError):
            pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Terminate the process and its whole group/tree, cross-platform."""
    if sys.platform == "win32":
        try:
            subprocess.run(  # noqa: S603,S607
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        except Exception:  # noqa: BLE001 - best-effort cleanup
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


@runtime_checkable
class Sandbox(Protocol):
    async def run(self, code: str, *, timeout: float) -> str: ...


class SubprocessSandbox:
    """Run code in an isolated subprocess (default; not a security boundary).

    Hardened: the child gets a scrubbed environment (no inherited secrets), runs
    in its own process group, and the whole group is killed on timeout so a
    spawned grandchild cannot outlive the deadline. POSIX additionally sets
    CPU/process/file-size rlimits. This bounds crashes, hangs, and accidental
    resource exhaustion — it is still **not** a boundary against a determined
    attacker (use :class:`DockerSandbox` for untrusted code).
    """

    async def run(self, code: str, *, timeout: float) -> str:
        kwargs: dict = {}
        if sys.platform == "win32":  # new process group so the tree can be killed
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # own process group for group-kill
            kwargs["preexec_fn"] = _set_rlimits  # noqa: PLW1509

        try:
            proc = subprocess.Popen(  # noqa: S603
                [sys.executable, "-I", "-c", _PREAMBLE + code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_minimal_env(),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            return f"error: execution exceeded {timeout}s timeout"
        except Exception as exc:  # noqa: BLE001
            _kill_tree(proc)
            return f"error: {exc}"
        if proc.returncode != 0:
            tail = (err or "").strip().splitlines()
            return f"error: {tail[-1] if tail else 'non-zero exit'}"
        return (out or "").strip() or "(no output)"


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
            "docker",
            "run",
            "--rm",
            "-i",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "64",
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


_default_sandbox: Sandbox | None = None


def set_default_sandbox(sandbox: Sandbox | None) -> None:
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
