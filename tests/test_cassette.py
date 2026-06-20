from __future__ import annotations

import pytest

from yaab.exceptions import CassetteMiss, YaabError


def test_cassette_miss_is_yaab_error():
    err = CassetteMiss("no interaction for request abc123 in 'c.json' (mode=replay)")
    assert isinstance(err, YaabError)
    assert "abc123" in str(err)


def test_request_key_is_stable_and_redacted():
    from yaab.models.cassette import _canonical_request, _request_key
    from yaab.types import Message, Role

    msgs = [Message(role=Role.USER, content="hi")]
    canon = _canonical_request(
        msgs,
        tools=None,
        output_schema=None,
        tool_choice=None,
        model="gemini/x",
        params={"temperature": 0.0, "api_key": "SECRET"},
    )
    # API keys are stripped from the canonical request.
    assert "SECRET" not in str(canon)
    assert "api_key" not in canon["params"]
    # Same inputs -> same key; different content -> different key.
    k1 = _request_key(canon)
    k2 = _request_key(
        _canonical_request(
            msgs,
            tools=None,
            output_schema=None,
            tool_choice=None,
            model="gemini/x",
            params={"temperature": 0.0},
        )
    )
    assert k1 == k2 and len(k1) == 64
    msgs2 = [Message(role=Role.USER, content="bye")]
    k3 = _request_key(
        _canonical_request(
            msgs2,
            tools=None,
            output_schema=None,
            tool_choice=None,
            model="gemini/x",
            params={},
        )
    )
    assert k3 != k1


def test_cassette_roundtrips_and_sequences(tmp_path):
    from yaab.models.cassette import _Cassette

    path = tmp_path / "c.json"
    cass = _Cassette(path)
    cass.append("k1", {"req": 1}, response={"content": "a"}, stream=None)
    cass.append("k1", {"req": 1}, response={"content": "b"}, stream=None)
    cass.save()

    reloaded = _Cassette(path)
    # Same key replays in recorded order, then exhausts.
    assert reloaded.next("k1")["response"]["content"] == "a"
    assert reloaded.next("k1")["response"]["content"] == "b"
    assert reloaded.next("k1") is None
    assert reloaded.next("unknown") is None
