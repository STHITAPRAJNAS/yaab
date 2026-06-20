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


@pytest.mark.asyncio
async def test_complete_records_then_replays(tmp_path):
    from yaab.models.cassette import CassetteModel
    from yaab.testing import TestModel
    from yaab.types import Message, Role

    path = tmp_path / "c.json"
    inner = TestModel(custom_output="hello world")

    rec = CassetteModel(path, inner=inner, mode="record")
    msgs = [Message(role=Role.USER, content="hi")]
    out1 = await rec.complete(msgs)
    assert out1.content == "hello world"
    assert path.exists()

    # Replay with NO inner and a model that would raise if called.
    rep = CassetteModel(path, mode="replay")
    out2 = await rep.complete(msgs)
    assert out2.content == "hello world"


@pytest.mark.asyncio
async def test_replay_miss_raises(tmp_path):
    from yaab.models.cassette import CassetteModel
    from yaab.testing import TestModel
    from yaab.types import Message, Role

    path = tmp_path / "c.json"
    # Record an unrelated request so the file exists but the key below misses.
    rec = CassetteModel(path, inner=TestModel(custom_output="x"), mode="record")
    await rec.complete([Message(role=Role.USER, content="recorded")])
    rep = CassetteModel(path, mode="replay")
    with pytest.raises(CassetteMiss):
        await rep.complete([Message(role=Role.USER, content="never recorded")])


@pytest.mark.asyncio
async def test_stream_records_then_replays(tmp_path):
    from yaab.models.cassette import CassetteModel
    from yaab.testing import TestModel
    from yaab.types import Message, Role

    path = tmp_path / "s.json"
    inner = TestModel(custom_output="abc")  # TestModel.stream yields deltas + done

    rec = CassetteModel(path, inner=inner, mode="record")
    chunks1 = [c.delta async for c in rec.stream([Message(role=Role.USER, content="hi")])]
    assert "".join(chunks1) == "abc"

    rep = CassetteModel(path, mode="replay")
    chunks2 = [c.delta async for c in rep.stream([Message(role=Role.USER, content="hi")])]
    assert chunks2 == chunks1


@pytest.mark.asyncio
async def test_use_cassette_default_mode(monkeypatch, tmp_path):
    from yaab.testing import use_cassette
    from yaab.testing import TestModel
    from yaab.types import Message, Role

    path = tmp_path / "u.json"
    # No YAAB_RECORD -> replay mode.
    monkeypatch.delenv("YAAB_RECORD", raising=False)
    with use_cassette(path) as model:
        assert model.mode == "replay"
    # YAAB_RECORD=1 with an inner -> record mode.
    monkeypatch.setenv("YAAB_RECORD", "1")
    with use_cassette(path, inner=TestModel(custom_output="z")) as model:
        assert model.mode == "record"
        out = await model.complete([Message(role=Role.USER, content="hi")])
        assert out.content == "z"


def test_testing_reexports():
    from yaab.testing import CassetteModel as ExportedCassetteModel

    assert ExportedCassetteModel is not None
