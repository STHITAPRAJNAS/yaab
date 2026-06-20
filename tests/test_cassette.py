from __future__ import annotations

import pytest

from yaab.exceptions import CassetteMiss, YaabError


def test_cassette_miss_is_yaab_error():
    err = CassetteMiss("no interaction for request abc123 in 'c.json' (mode=replay)")
    assert isinstance(err, YaabError)
    assert "abc123" in str(err)
