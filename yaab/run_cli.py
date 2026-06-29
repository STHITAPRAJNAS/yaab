"""``yaab run`` — headless, automatable execution of an agent (the harness CLI).

This is the command-line front door to the packaged coding harness and to any
user agent (``module:agent``). It is built for automation and CI:

* **Deterministic exit codes** (:data:`EXIT`) — a caller can branch on *why* a
  run stopped: approval required, budget exhausted, timeout, cancelled, etc.
* **Clean streams** — the final answer goes to **stdout**; all progress, prompts,
  and diagnostics go to **stderr**. ``--json`` makes stdout a single envelope.
* **Bounded by default** — every run gets a :class:`~yaab.limits.UsageLimits`
  (tokens / requests / tool-calls / wall-clock) so nothing runs away.
* **Safe approvals** — gated (destructive) tools are never silently auto-run.
  Non-interactive with no policy ⇒ the run *pauses* (exit ``approval``) and prints
  a resume id. ``--auto`` / ``--read-only`` / ``--approve-caps`` are explicit
  opt-ins; there is deliberately no blanket ``--yes``.
* **Resumable & interruptible** — a durable checkpointer makes pauses survive the
  process; ``SIGTERM``/``SIGINT`` cancels cooperatively, checkpoints, and prints a
  resume id.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid
from typing import Any

from .exceptions import (
    ApprovalPending,
    ApprovalRequired,
    GovernanceError,
    MaxStepsExceeded,
    ModelError,
    OutputValidationError,
    PolicyViolation,
    RunCancelled,
    UsageLimitExceeded,
)
from .limits import CancellationToken, UsageLimits

#: Named exit codes — the contract a CI/automation caller branches on.
EXIT: dict[str, int] = {
    "ok": 0,
    "error": 1,  # unexpected / tool / generic failure
    "usage": 2,  # CLI misuse (argparse-style)
    "approval": 3,  # paused awaiting human approval
    "budget": 4,  # usage limit exceeded
    "timeout": 5,  # exceeded --timeout
    "cancelled": 6,  # SIGTERM/SIGINT
    "max_steps": 7,  # agent loop hit max steps
    "output": 8,  # output validation failed
    "model": 9,  # model/provider error
    "policy": 10,  # governance / policy violation
}


def exit_code_for(exc: BaseException) -> int:
    """Map an exception raised by a run to a stable :data:`EXIT` code.

    Ordered most-specific first so a subclass never matches a broader parent.
    """
    if isinstance(exc, (ApprovalRequired, ApprovalPending)):
        return EXIT["approval"]
    if isinstance(exc, UsageLimitExceeded):
        return EXIT["budget"]
    if isinstance(exc, RunCancelled):
        return EXIT["timeout"] if getattr(exc, "reason", "") == "timeout" else EXIT["cancelled"]
    if isinstance(exc, MaxStepsExceeded):
        return EXIT["max_steps"]
    if isinstance(exc, OutputValidationError):
        return EXIT["output"]
    if isinstance(exc, ModelError):
        return EXIT["model"]
    if isinstance(exc, (PolicyViolation, GovernanceError)):
        return EXIT["policy"]
    return EXIT["error"]


def build_usage_limits(
    *,
    budget_tokens: int | None = None,
    max_requests: int | None = None,
    max_tool_calls: int | None = None,
    max_wall_seconds: float | None = None,
) -> UsageLimits:
    """Build a :class:`UsageLimits` from CLI flags over a conservative default base.

    Unset flags fall through to the harness default budget, so a run is *always*
    bounded — never unlimited — unless the caller raises the ceilings explicitly.
    """
    from .harness import default_usage_limits

    base = default_usage_limits()
    return UsageLimits(
        max_requests=max_requests if max_requests is not None else base.max_requests,
        max_total_tokens=budget_tokens if budget_tokens is not None else base.max_total_tokens,
        max_tool_calls=max_tool_calls if max_tool_calls is not None else base.max_tool_calls,
        max_wall_seconds=(
            max_wall_seconds if max_wall_seconds is not None else base.max_wall_seconds
        ),
    )


# --- approval policies --------------------------------------------------------
# An approver is ``async (tool, args, ctx) -> bool`` (True = allow, False = deny).
# Returning False feeds the model an "not approved" tool result so it can adapt;
# *pausing* (no approver) is a different, stronger behavior chosen by the caller.


def auto_approver() -> Any:
    """Approve every gated tool. Explicit opt-in (``--auto``) for trusted automation."""

    async def approve(tool: str, args: dict, ctx: Any) -> bool:
        return True

    return approve


def read_only_approver() -> Any:
    """Deny every gated tool — with the default gate this makes the agent read-only."""

    async def approve(tool: str, args: dict, ctx: Any) -> bool:
        return False

    return approve


def caps_approver(approved: set[Any] | frozenset[Any]) -> Any:
    """Approve a gated tool iff its capabilities are all within ``approved``.

    Reads the dispatching tool's capabilities from the ``current_tool_capabilities``
    ContextVar the runner sets before approval — so this gates by *effect*, not by
    the tool's name.
    """
    from .capabilities import current_tool_capabilities

    allowed = frozenset(approved)

    async def approve(tool: str, args: dict, ctx: Any) -> bool:
        caps = current_tool_capabilities.get()
        return bool(caps) and caps <= allowed

    return approve


def interactive_approver(stream: Any = sys.stderr) -> Any:
    """Prompt a human on ``stream`` for each gated tool (y/N). Default-deny."""

    async def approve(tool: str, args: dict, ctx: Any) -> bool:
        print(
            f"\nApprove tool '{tool}' with args {args!r}? [y/N] ", end="", file=stream, flush=True
        )
        try:
            answer = sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    return approve


# --- output envelope ----------------------------------------------------------
def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "requests": getattr(usage, "requests", 0),
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "cost_usd": getattr(usage, "cost_usd", 0.0),
    }


def build_envelope(result: Any, *, resume_id: str | None) -> dict[str, Any]:
    """A JSON-serializable summary of a run for ``--json`` consumers."""
    paused = bool(getattr(result, "paused", False))
    return {
        "status": "paused" if paused else getattr(result, "status", "ok"),
        "paused": paused,
        "output": None if paused else getattr(result, "output", None),
        "pause": getattr(result, "pause_value", None) if paused else None,
        "resume_id": resume_id,
        "run_id": getattr(result, "run_id", None),
        "usage": _usage_dict(getattr(result, "usage", None)),
    }


# --- approver selection from parsed args -------------------------------------
def select_approver(
    *, auto: bool, read_only: bool, approve_caps: list[str] | None, interactive: bool
) -> Any | None:
    """Choose the approver from the approval flags (``None`` ⇒ pause-on-gate).

    Precedence: ``--read-only`` (deny) > ``--auto`` (allow) > ``--approve-caps``
    (subset allow) > interactive TTY prompt > ``None`` (pause for out-of-band
    sign-off). There is intentionally no blanket auto-approve default.
    """
    if read_only:
        return read_only_approver()
    if auto:
        return auto_approver()
    if approve_caps:
        from .capabilities import Capability

        caps = {Capability(c.strip()) for c in approve_caps if c.strip()}
        return caps_approver(caps)
    if interactive:
        return interactive_approver()
    return None


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def add_run_subparser(sub: Any) -> None:
    """Register the ``run`` command on the top-level argparse subparsers."""
    p = sub.add_parser("run", help="run an agent headlessly (coding harness or module:agent)")
    p.add_argument(
        "spec",
        nargs="?",
        default=None,
        help="agent as module:attribute; omit with --coding to use the built-in coder",
    )
    p.add_argument("prompt", nargs="?", default="", help="the task/prompt for the agent")
    p.add_argument("--coding", action="store_true", help="use the bundled coding_agent")
    p.add_argument("--root", default=".", help="project root for --coding (sandbox confine)")
    p.add_argument("--model", default=None, help="model spec override (LiteLLM)")
    p.add_argument("--json", action="store_true", dest="as_json", help="emit a JSON envelope")
    p.add_argument("--budget-tokens", type=int, default=None, help="max total tokens")
    p.add_argument("--max-tool-calls", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="cap agent loop iterations")
    p.add_argument("--timeout", type=float, default=None, help="wall-clock seconds (soft+hard)")
    p.add_argument("--auto", action="store_true", help="auto-approve gated tools (explicit)")
    p.add_argument("--read-only", action="store_true", help="deny all gated (write/exec) tools")
    p.add_argument(
        "--approve-caps",
        default=None,
        help="comma-separated capabilities to auto-approve (e.g. fs_write_in_root,net_egress)",
    )
    p.add_argument("--enable-shell", action="store_true", help="add shell_exec (--coding)")
    p.add_argument("--state-db", default=None, help="SQLite path for durable resume")
    p.add_argument("--resume-id", default=None, help="resume a previously paused run")


def run_command(args: Any) -> int:
    """Entry point for ``yaab run`` — returns a :data:`EXIT` code."""
    if not args.coding and not args.spec:
        _eprint("error: provide a module:agent spec or --coding")
        return EXIT["usage"]

    interactive = sys.stdin.isatty() and sys.stderr.isatty()
    approver = select_approver(
        auto=args.auto,
        read_only=args.read_only,
        approve_caps=(args.approve_caps.split(",") if args.approve_caps else None),
        interactive=interactive,
    )
    limits = build_usage_limits(
        budget_tokens=args.budget_tokens,
        max_tool_calls=args.max_tool_calls,
        max_wall_seconds=args.timeout,
    )
    resume_id = args.resume_id or uuid.uuid4().hex

    try:
        agent = _resolve_agent(args, approver=approver, resume_id=resume_id)
    except Exception as exc:  # noqa: BLE001 - surface resolution errors as usage
        _eprint(f"error: could not load agent: {exc}")
        return EXIT["usage"]

    cancel = CancellationToken()
    _install_signal_handlers(cancel)
    try:
        result = asyncio.run(
            agent.run(
                args.prompt,
                usage_limits=limits,
                cancellation=cancel,
                timeout=args.timeout,
                resume_id=resume_id,
            )
        )
    except KeyboardInterrupt:
        _eprint(f"\ncancelled — resume with: yaab run --resume-id {resume_id}")
        return EXIT["cancelled"]
    except BaseException as exc:  # noqa: BLE001 - map every failure to an exit code
        code = exit_code_for(exc)
        if code == EXIT["approval"]:
            _eprint(f"paused for approval — resume with: yaab run --resume-id {resume_id}")
        else:
            _eprint(f"error: {type(exc).__name__}: {exc}")
        return code

    return _emit(result, resume_id=resume_id, as_json=args.as_json)


def _emit(result: Any, *, resume_id: str | None, as_json: bool) -> int:
    paused = bool(getattr(result, "paused", False))
    if as_json:
        import json

        print(json.dumps(build_envelope(result, resume_id=resume_id)))
    elif paused:
        _eprint(f"paused — resume with: yaab run --resume-id {resume_id}")
    else:
        print(getattr(result, "output", "") or "")
    if paused:
        return EXIT["approval"]
    return EXIT["ok"] if getattr(result, "status", "ok") == "ok" else EXIT["error"]


def _resolve_agent(args: Any, *, approver: Any | None, resume_id: str) -> Any:
    if args.coding:
        from .harness import coding_agent

        checkpointer = None
        if args.state_db:
            from .graph.checkpoint import SQLiteSaver

            checkpointer = SQLiteSaver(args.state_db)
        kwargs: dict[str, Any] = {
            "root": args.root,
            "approver": approver,
            "enable_shell": args.enable_shell,
            "checkpointer": checkpointer,
        }
        if args.model:
            kwargs["model"] = args.model
        if args.max_steps is not None:
            kwargs["max_steps"] = args.max_steps
        return coding_agent(**kwargs)

    # module:attribute agent (resolved like `yaab serve`).
    from .cli import _load_attr

    return _load_attr(args.spec)


def _install_signal_handlers(cancel: CancellationToken) -> None:
    """Cancel cooperatively on SIGINT/SIGTERM so the run can checkpoint and exit."""

    def _handler(signum: int, frame: Any) -> None:
        cancel.cancel()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not on the main thread (e.g. under pytest) — skip; cancellation still
            # works via the token, just not signal-driven.
            pass


__all__ = [
    "EXIT",
    "exit_code_for",
    "build_usage_limits",
    "build_envelope",
    "select_approver",
    "auto_approver",
    "read_only_approver",
    "caps_approver",
    "interactive_approver",
    "add_run_subparser",
    "run_command",
]
