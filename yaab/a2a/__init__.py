"""Agent-to-Agent (A2A) interop.

The server side lives in :mod:`yaab.serve` (agent card + ``/a2a/tasks``). This
package is the *client* side: :class:`RemoteAgent` discovers a remote agent via
its Agent Card and delegates tasks to it. A ``RemoteAgent`` satisfies both the
:class:`~yaab.tools.base.Tool` protocol (so it can be handed to a local agent as
a tool) and the workflow ``run`` surface (so it composes like any agent).
"""

from __future__ import annotations

from .client import RemoteAgent

__all__ = ["RemoteAgent"]
