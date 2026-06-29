"""Tool capabilities — what a tool can *do*, so policy gates by effect not name.

A tool routed through a differently-named tool (e.g. file deletion via a Python
exec tool) must still be gated; name-based gating is bypassable, capability-based
gating is not.
"""

from __future__ import annotations

from contextvars import ContextVar
from enum import Enum


class Capability(str, Enum):
    FS_READ = "fs_read"
    FS_WRITE_IN_ROOT = "fs_write_in_root"
    FS_WRITE_OUT = "fs_write_out"
    NET_EGRESS = "net_egress"
    PROCESS_SPAWN = "process_spawn"
    ENV_READ = "env_read"
    SCHEDULE = "schedule"


#: Capabilities whose misuse is destructive/irreversible or exfiltrating; these
#: require a covering approval gate (the harness fails closed if one is missing).
DESTRUCTIVE: frozenset[Capability] = frozenset(
    {
        Capability.FS_WRITE_OUT,
        Capability.PROCESS_SPAWN,
        Capability.NET_EGRESS,
        Capability.SCHEDULE,
    }
)

#: The capabilities of the tool the runner is about to dispatch, so approval can
#: gate by effect. A ``ContextVar`` (not an attribute on the shared RunContext)
#: so concurrent tool calls in one turn — each running in its own asyncio Task,
#: which copies the context — never read each other's value (no gate race).
current_tool_capabilities: ContextVar[frozenset[Capability]] = ContextVar(
    "current_tool_capabilities", default=frozenset()
)
