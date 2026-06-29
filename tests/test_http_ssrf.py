from __future__ import annotations

from yaab.tools.builtin.http import _host_is_blocked


def test_blocks_metadata_and_private_ips():
    assert _host_is_blocked("169.254.169.254") is True  # cloud metadata
    assert _host_is_blocked("127.0.0.1") is True
    assert _host_is_blocked("localhost") is True
    assert _host_is_blocked("10.0.0.5") is True
    assert _host_is_blocked("192.168.1.1") is True


def test_blocks_cgnat_range():
    assert _host_is_blocked("100.64.1.1") is True  # RFC 6598 carrier-grade NAT


def test_allows_public_host():
    # A literal public IP (no DNS needed) is allowed.
    assert _host_is_blocked("93.184.216.34") is False
