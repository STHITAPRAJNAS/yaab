from __future__ import annotations

import pytest

from yaab.capabilities import Capability
from yaab.governance import ToolApprovalPlugin


def test_gate_capabilities_matches_by_effect():
    plugin = ToolApprovalPlugin(gate_capabilities={Capability.NET_EGRESS})
    assert plugin.guards_capability(Capability.NET_EGRESS) is True
    assert plugin.guards_capability(Capability.FS_READ) is False


def test_timeout_approve_rejected_for_destructive():
    with pytest.raises(ValueError, match="on_timeout"):
        ToolApprovalPlugin(gate_capabilities={Capability.PROCESS_SPAWN}, on_timeout="approve")


def test_gate_capabilities_alone_is_valid():
    # Previously you had to pass tools/needs_approval; capabilities now suffice.
    plugin = ToolApprovalPlugin(gate_capabilities={Capability.FS_WRITE_OUT})
    assert plugin is not None
