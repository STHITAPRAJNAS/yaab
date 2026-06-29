"""``shell_exec`` — an allowlisted, argv-only command tool for the harness.

Running shell commands is the single most dangerous capability an agent can
have, so this tool stacks several independent controls:

1. **Argv only.** The model supplies a list ``["git", "status"]``, never a
   string. There is no shell, so ``;``, ``|``, ``$(...)`` and friends are inert
   literal arguments — command injection has nothing to inject into.
2. **Exact-binary allowlist.** ``argv[0]`` must be a key in ``rules``. Nothing
   else runs. An empty ruleset therefore allows *nothing* (fail-closed).
3. **Per-binary arg validation.** Each :class:`ShellRule` constrains its binary's
   arguments — allowed subcommands, an arg-count cap, denied substrings, or a
   custom validator.
4. **Isolation + capability gate.** Execution happens on a
   :class:`~yaab.tools.exec.CommandSandbox` (an isolating one unless the caller
   explicitly downgrades), and the tool carries ``Capability.PROCESS_SPAWN`` so an
   approval plugin can gate every call.

Errors come back as ``error: ...`` strings (fed to the model), never raised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ...capabilities import Capability
from ..base import FunctionTool, tool
from ..exec import CommandSandbox, require_isolated

#: Default per-command wall-clock when the caller does not pass one.
_DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class ShellRule:
    """Argument policy for one allowlisted binary.

    All checks are conjunctive — every configured constraint must pass.

    * ``subcommands`` — if non-empty, ``argv[1]`` must be one of these.
    * ``max_args`` — cap on the number of arguments after the binary.
    * ``deny_substrings`` — reject if any argument contains one of these (handy
      for blocking ``..`` traversal or ``--upload-pack`` style flags).
    * ``validator`` — custom callable over the argument list (excluding the
      binary); returns an error string to reject, or ``None`` to allow.
    """

    subcommands: frozenset[str] = field(default_factory=frozenset)
    max_args: int | None = None
    deny_substrings: tuple[str, ...] = ()
    validator: Callable[[list[str]], str | None] | None = None

    def check(self, args: list[str]) -> str | None:
        if self.subcommands and (not args or args[0] not in self.subcommands):
            allowed = ", ".join(sorted(self.subcommands))
            return f"subcommand must be one of: {allowed}"
        if self.max_args is not None and len(args) > self.max_args:
            return f"too many arguments (max {self.max_args}, got {len(args)})"
        for a in args:
            for bad in self.deny_substrings:
                if bad in a:
                    return f"argument {a!r} contains disallowed text {bad!r}"
        if self.validator is not None:
            return self.validator(args)
        return None


def make_shell_exec(
    *,
    sandbox: CommandSandbox,
    rules: dict[str, ShellRule],
    require_isolation: bool = True,
    default_timeout: float = _DEFAULT_TIMEOUT,
    cwd: str | None = None,
) -> FunctionTool:
    """Build a ``shell_exec`` tool bound to ``sandbox`` and an allowlist.

    ``rules`` maps an exact ``argv[0]`` to its :class:`ShellRule`. By default the
    sandbox must be an isolation boundary (:func:`require_isolated`); pass
    ``require_isolation=False`` to *explicitly* accept a weaker subprocess
    sandbox (an informed downgrade, never a silent default).
    """
    if require_isolation:
        require_isolated(sandbox)
    # Snapshot so later mutation of the caller's dict can't widen the allowlist.
    allow: dict[str, ShellRule] = dict(rules)

    @tool(name="shell_exec", capabilities={Capability.PROCESS_SPAWN})
    async def shell_exec(command: list[str], timeout: float = default_timeout) -> str:
        """Run an allowlisted command given as an argv list (no shell).

        ``command`` is a list like ``["git", "status"]`` — the first element is
        the binary (which must be allowlisted) and the rest are literal
        arguments. Returns the command's combined output, or ``error: ...`` if it
        is rejected, fails, or times out.
        """
        # Argv discipline: a bare string would otherwise be iterated char-by-char.
        if isinstance(command, str):
            return "error: command must be a list of arguments, e.g. ['git', 'status']"
        if not command or not all(isinstance(a, str) for a in command):
            return "error: command must be a non-empty list of string arguments"
        if any("\x00" in a for a in command):
            return "error: command arguments must not contain NUL bytes"

        binary = command[0]
        rule = allow.get(binary)
        if rule is None:
            allowed = ", ".join(sorted(allow)) or "(none)"
            return f"error: '{binary}' is not in the allowlist; allowed binaries: {allowed}"
        why = rule.check(command[1:])
        if why is not None:
            return f"error: {binary}: {why}"

        if timeout <= 0:
            return "error: timeout must be positive"
        result = await sandbox.run_argv(command, timeout=timeout, cwd=cwd)
        if result.timed_out:
            return f"error: {result.stderr or 'command timed out'}"
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()
            return f"error: exited {result.returncode}: {tail or '(no output)'}"
        body = result.stdout.strip()
        if result.stderr.strip():
            body = (body + "\n" + result.stderr.strip()).strip()
        return body or "(no output)"

    return shell_exec


__all__ = ["ShellRule", "make_shell_exec"]
