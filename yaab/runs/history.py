"""Run history, inspection, and forking — time-travel debugging for durable runs.

A run that uses a checkpointer leaves a trail of per-step snapshots. :class:`RunHistory`
reads that trail so a developer (or a dev console) can:

* **list** every checkpoint of a run, oldest first, to render its timeline;
* **inspect** the captured state at any one checkpoint, including a paused one;
* **fork** from a checkpoint into a new thread — optionally editing the state —
  to explore an alternate timeline without disturbing the original.

It is a thin read/transform layer over the same ``Checkpointer`` the Runner and
the durable graph already use (``put``/``get``/``history``), so it works with the
in-memory, SQLite, Postgres, and Redis checkpoint backends without any new store.
"""

from __future__ import annotations

from typing import Any


class RunHistory:
    """Time-travel access to a run's checkpoints over any ``Checkpointer``.

    Each checkpoint is the snapshot the engine persisted for one step: a mapping
    with a ``"state"`` payload (the run's accumulated values) plus engine
    bookkeeping (``"frontier"``, ``"retries"``). The methods here never mutate the
    original timeline — :meth:`fork` writes to a *different* thread.
    """

    def __init__(self, checkpointer: Any) -> None:
        self._cp = checkpointer

    def list(self, thread_id: str) -> list[tuple[int, dict[str, Any]]]:
        """Every checkpoint of ``thread_id`` as ``(step, snapshot)``, oldest first."""
        return list(self._cp.history(thread_id))

    def inspect(self, thread_id: str, step: int) -> dict[str, Any] | None:
        """The snapshot captured at ``step`` (or ``None`` if there is no such step)."""
        for s, snapshot in self._cp.history(thread_id):
            if s == step:
                return snapshot
        return None

    def latest(self, thread_id: str) -> dict[str, Any] | None:
        """The most recent snapshot of ``thread_id`` (including a paused one)."""
        got = self._cp.get(thread_id)
        if got is None:
            return None
        _, snapshot = got
        return snapshot

    def fork(
        self,
        thread_id: str,
        step: int,
        *,
        to_thread: str,
        edits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Copy the checkpoint at ``step`` into ``to_thread``, optionally edited.

        Returns the snapshot written to ``to_thread``. The source timeline is left
        untouched, so forking is a non-destructive "what if I had changed X here?"
        — the new thread can then be resumed (``runner.run(..., session_id=to_thread,
        resume_from_checkpoint=True)``) to continue from the edited state.
        """
        snapshot = self.inspect(thread_id, step)
        if snapshot is None:
            raise KeyError(f"no checkpoint at step {step} for run {thread_id!r}")
        # Deep-ish copy so editing the fork never reaches back into the source.
        forked: dict[str, Any] = {
            "state": dict(snapshot.get("state", {})),
            "frontier": list(snapshot.get("frontier", [])),
            "retries": dict(snapshot.get("retries", {})),
        }
        if edits:
            forked["state"].update(edits)
        self._cp.put(to_thread, step, forked)
        return forked
