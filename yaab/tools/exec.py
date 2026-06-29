"""Command-execution sandboxes for the ``shell_exec`` harness tool.

The :mod:`~yaab.tools.sandbox` module runs *Python snippets*; this one runs a
*single argv vector* — no shell, no string parsing, no metacharacter expansion.
That argv discipline is the first line of defence: there is no ``sh -c`` for an
injected ``; rm -rf /`` to ride in on.

Two backends ship, mirroring the Python sandboxes:

* :class:`SubprocessCommandSandbox` — runs the argv in a child process with a
  scrubbed environment, its own process group, and (POSIX) rlimits. It bounds
  blast radius but is **not** an isolation boundary: ``is_isolated`` is ``False``.
* :class:`DockerCommandSandbox` — runs the argv in a throwaway container with the
  workspace bind-mounted, no network, and dropped capabilities. ``is_isolated``
  is ``True`` — a real boundary for untrusted commands.

:func:`require_isolated` turns ``is_isolated`` into an enforced gate so a
``process_spawn`` tool can refuse to run on a non-isolating backend.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .sandbox import _kill_tree, _minimal_env, _set_rlimits

#: Absolute ceiling on a single command's wall-clock, regardless of caller input.
_MAX_TIMEOUT = 600.0
#: Cap on captured output bytes so a chatty command can't blow up memory/context.
_MAX_OUTPUT = 100_000


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one argv execution."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@runtime_checkable
class CommandSandbox(Protocol):
    #: Whether this backend is a real isolation boundary (see module docstring).
    is_isolated: bool

    async def run_argv(
        self, argv: list[str], *, timeout: float, cwd: str | None = None
    ) -> CommandResult: ...


class SandboxNotIsolatedError(RuntimeError):
    """Raised when a ``process_spawn`` tool is built on a non-isolating sandbox."""


def require_isolated(sandbox: CommandSandbox) -> CommandSandbox:
    """Return ``sandbox`` if it is an isolation boundary, else raise.

    This is the fail-closed gate for spawning processes: a caller that wants to
    run shell commands must pass a sandbox whose ``is_isolated`` is ``True`` (e.g.
    :class:`DockerCommandSandbox`) or explicitly opt out of isolation at a higher
    layer.
    """
    if not getattr(sandbox, "is_isolated", False):
        raise SandboxNotIsolatedError(
            f"{type(sandbox).__name__} is not an isolation boundary; "
            "process_spawn requires an isolating sandbox (e.g. DockerCommandSandbox)"
        )
    return sandbox


def _clamp(out: str) -> str:
    return out if len(out) <= _MAX_OUTPUT else out[:_MAX_OUTPUT] + "\n... (output truncated)"


class SubprocessCommandSandbox:
    """Run an argv in a hardened child process (not an isolation boundary).

    The child gets a scrubbed environment (no inherited secrets), runs in its own
    process group so the whole tree can be killed on timeout, and on POSIX is
    capped by CPU/process/file-size rlimits. Good enough to bound a *trusted*
    command's blast radius; use :class:`DockerCommandSandbox` for untrusted input.
    """

    is_isolated = False

    async def run_argv(
        self, argv: list[str], *, timeout: float, cwd: str | None = None
    ) -> CommandResult:
        timeout = min(timeout, _MAX_TIMEOUT)
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _set_rlimits  # noqa: PLW1509

        try:
            proc = subprocess.Popen(  # noqa: S603
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_minimal_env(),
                cwd=cwd,
                **kwargs,
            )
        except FileNotFoundError:
            return CommandResult(127, "", f"command not found: {argv[0]}")
        except OSError as exc:
            return CommandResult(1, "", f"failed to launch {argv[0]}: {exc}")

        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            return CommandResult(-1, "", f"command exceeded {timeout}s timeout", timed_out=True)
        except Exception as exc:  # noqa: BLE001
            _kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            return CommandResult(1, "", f"command failed: {exc}")
        return CommandResult(proc.returncode, _clamp(out or ""), _clamp(err or ""))


class DockerCommandSandbox:
    """Run an argv in a locked-down throwaway container (real isolation).

    The workspace (``workdir``) is bind-mounted at ``/work`` and made the working
    directory. Defaults: no network, dropped capabilities, pids cap, and
    memory/CPU limits. Requires a Docker daemon and the chosen image.
    """

    is_isolated = True

    def __init__(
        self,
        *,
        workdir: str,
        image: str = "python:3.11-slim",
        memory: str = "512m",
        cpus: str = "1.0",
        network: bool = False,
        read_only: bool = False,
    ) -> None:
        self.workdir = workdir
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.read_only = read_only

    async def run_argv(
        self, argv: list[str], *, timeout: float, cwd: str | None = None
    ) -> CommandResult:
        timeout = min(timeout, _MAX_TIMEOUT)
        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "256",
            "-v",
            f"{self.workdir}:/work" + (":ro" if self.read_only else ""),
            "-w",
            "/work",
        ]
        if not self.network:
            cmd += ["--network", "none"]
        cmd += [self.image, *argv]
        try:
            proc = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=timeout + 5
            )
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"command exceeded {timeout}s timeout", timed_out=True)
        except FileNotFoundError:
            return CommandResult(127, "", "docker is not available on this host")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(1, "", f"command failed: {exc}")
        return CommandResult(proc.returncode, _clamp(proc.stdout or ""), _clamp(proc.stderr or ""))


__all__ = [
    "CommandResult",
    "CommandSandbox",
    "SubprocessCommandSandbox",
    "DockerCommandSandbox",
    "SandboxNotIsolatedError",
    "require_isolated",
]
