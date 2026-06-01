"""Usage limits and run cancellation — bounded, interruptible runs.

Two small, composable controls the ecosystem repeatedly asks for:

* :class:`UsageLimits` — hard caps on requests, input/output/total tokens, tool
  calls (overall and **per-tool**), and wall-clock seconds. Checked by the
  Runner between steps and before each tool call; breaching one raises
  :class:`~yaab.exceptions.UsageLimitExceeded`.
* :class:`CancellationToken` — a cooperative stop signal. Call
  :meth:`CancellationToken.cancel` (from a signal handler, a timeout, another
  task, an API endpoint) and the Runner stops at the next checkpoint with
  :class:`~yaab.exceptions.RunCancelled`. A ``timeout`` on the run wires an
  automatic deadline to the same mechanism.

Both are optional; with neither set, runs behave exactly as before.
"""

from __future__ import annotations

import time

from .exceptions import RunCancelled, UsageLimitExceeded
from .types import Usage


class UsageLimits:
    """Declarative caps enforced across an agent run.

    Any limit left ``None`` is unbounded. ``per_tool_calls`` maps a tool name to
    its own maximum call count (e.g. ``{"charge": 1}``).
    """

    def __init__(
        self,
        *,
        max_requests: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_total_tokens: int | None = None,
        max_tool_calls: int | None = None,
        per_tool_calls: dict[str, int] | None = None,
        max_wall_seconds: float | None = None,
    ) -> None:
        self.max_requests = max_requests
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.max_tool_calls = max_tool_calls
        self.per_tool_calls = dict(per_tool_calls or {})
        self.max_wall_seconds = max_wall_seconds

    def check_usage(self, usage: Usage) -> None:
        """Raise if the accumulated token/request usage breaches a cap."""
        if self.max_requests is not None and usage.requests > self.max_requests:
            raise UsageLimitExceeded(
                f"request limit exceeded: {usage.requests} > {self.max_requests}",
                limit="requests",
            )
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            raise UsageLimitExceeded(
                f"input-token limit exceeded: {usage.input_tokens} > {self.max_input_tokens}",
                limit="input_tokens",
            )
        if self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            raise UsageLimitExceeded(
                f"output-token limit exceeded: {usage.output_tokens} > {self.max_output_tokens}",
                limit="output_tokens",
            )
        if self.max_total_tokens is not None and usage.total_tokens > self.max_total_tokens:
            raise UsageLimitExceeded(
                f"total-token limit exceeded: {usage.total_tokens} > {self.max_total_tokens}",
                limit="total_tokens",
            )

    def check_wall_clock(self, started_at: float) -> None:
        """Raise if the run has exceeded its wall-clock budget.

        ``started_at`` is a ``time.monotonic()`` timestamp from the run's start.
        """
        if self.max_wall_seconds is None:
            return
        elapsed = time.monotonic() - started_at
        if elapsed > self.max_wall_seconds:
            raise UsageLimitExceeded(
                f"wall-clock limit exceeded: {elapsed:.3f}s > {self.max_wall_seconds}s",
                limit="wall_seconds",
            )

    def check_tool_call(self, tool_name: str, counts: dict[str, int]) -> None:
        """Raise if invoking ``tool_name`` would breach an overall/per-tool cap.

        ``counts`` is the running tally of calls per tool *including* this one.
        """
        total = sum(counts.values())
        if self.max_tool_calls is not None and total > self.max_tool_calls:
            raise UsageLimitExceeded(
                f"tool-call limit exceeded: {total} > {self.max_tool_calls}",
                limit="tool_calls",
            )
        cap = self.per_tool_calls.get(tool_name)
        if cap is not None and counts.get(tool_name, 0) > cap:
            raise UsageLimitExceeded(
                f"per-tool limit exceeded for '{tool_name}': {counts[tool_name]} > {cap}",
                limit=f"tool:{tool_name}",
            )


class CancellationToken:
    """A cooperative cancellation signal shared with a run.

    The Runner checks :meth:`raise_if_cancelled` between supersteps and before
    each tool call. A token can carry a wall-clock deadline so timeouts and
    explicit cancels flow through one path.
    """

    def __init__(self, *, deadline: float | None = None) -> None:
        self._cancelled = False
        self._reason = "cancelled"
        self.deadline = deadline

    @classmethod
    def with_timeout(cls, seconds: float) -> CancellationToken:
        return cls(deadline=time.monotonic() + seconds)

    @property
    def cancelled(self) -> bool:
        if self._cancelled:
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._cancelled = True
            self._reason = "timeout"
            return True
        return False

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancelled = True
        self._reason = reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled(f"run {self._reason}", reason=self._reason)


__all__ = ["UsageLimits", "CancellationToken"]
