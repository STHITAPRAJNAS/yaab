"""Turn-based voice pipeline tests.

These exercise :class:`yaab.voice.VoiceAgent` with *fake* transcriber/speaker
implementations so no network, API key, or ``litellm`` install is needed. The
fakes also prove the design's central claim: every provider call is injectable,
so the default LiteLLM-backed implementations are never imported during a test
run.
"""

from __future__ import annotations

import builtins

import pytest

from yaab import Agent
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.voice import (
    LiteLLMSpeaker,
    LiteLLMTranscriber,
    LiveVoiceSession,
    Speaker,
    Transcriber,
    VoiceAgent,
    VoiceTurn,
)


class FakeTranscriber:
    """Returns a scripted transcript; records what audio/format it saw."""

    def __init__(self, transcript: str = "what time is it") -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio_bytes: bytes, *, format: str = "wav") -> str:
        self.calls.append((audio_bytes, format))
        return self.transcript


class FakeSpeaker:
    """Returns deterministic audio derived from the text; records voice used."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def speak(self, text: str, *, voice: str = "alloy") -> bytes:
        self.calls.append((text, voice))
        return b"AUDIO:" + text.encode("utf-8")


def _agent(model: TestModel) -> Agent:
    return Agent("voice-bot", model=model)


@pytest.mark.asyncio
async def test_speak_turn_full_pipeline():
    """transcribe -> agent -> speak produces a populated VoiceTurn."""
    transcriber = FakeTranscriber("what time is it")
    speaker = FakeSpeaker()
    agent = _agent(TestModel("it is noon"))
    va = VoiceAgent(agent, transcriber=transcriber, speaker=speaker)

    turn = await va.speak_turn(b"\x00\x01wav-bytes")

    assert isinstance(turn, VoiceTurn)
    assert turn.transcript == "what time is it"
    assert turn.response_text == "it is noon"
    # Audio came from the speaker, keyed off the agent's text.
    assert turn.response_audio == b"AUDIO:it is noon"
    # The transcriber actually received the audio we passed in.
    assert transcriber.calls[0][0] == b"\x00\x01wav-bytes"
    # Speaker was asked to render exactly the agent's response.
    assert speaker.calls[0][0] == "it is noon"
    # Usage is propagated from the underlying run.
    assert turn.usage.requests >= 1


@pytest.mark.asyncio
async def test_speak_false_skips_tts():
    """speak=False yields text only and never touches the speaker."""
    speaker = FakeSpeaker()
    va = VoiceAgent(
        _agent(TestModel("text only")),
        transcriber=FakeTranscriber("hello"),
        speaker=speaker,
    )

    turn = await va.speak_turn(b"audio", speak=False)

    assert turn.response_text == "text only"
    assert turn.response_audio is None
    assert speaker.calls == []  # speaker untouched


@pytest.mark.asyncio
async def test_session_continuity_across_turns():
    """Passing session_id threads conversation memory across voice turns.

    The model echoes how many prior USER messages it has seen, so a growing
    count proves the session history is being replayed turn-to-turn.
    """

    def count_users(messages):
        n = sum(1 for m in messages if m.role.value == "user")
        return ModelResponse(content=f"seen {n}", model="counter")

    from yaab.models.test_model import FunctionModel

    va = VoiceAgent(
        _agent(FunctionModel(count_users)),
        transcriber=FakeTranscriber("turn"),
        speaker=FakeSpeaker(),
    )

    first = await va.speak_turn(b"a1", session_id="s1")
    second = await va.speak_turn(b"a2", session_id="s1")

    assert first.response_text == "seen 1"
    # Second turn sees the prior user message too (history replayed).
    assert second.response_text == "seen 2"


@pytest.mark.asyncio
async def test_stream_turn_order_transcript_deltas_audio():
    """stream_turn yields transcript dict, then text deltas, then audio dict."""
    va = VoiceAgent(
        _agent(TestModel("hello world foo")),
        transcriber=FakeTranscriber("say something"),
        speaker=FakeSpeaker(),
    )

    items = [item async for item in va.stream_turn(b"audio")]

    # First item carries the transcript.
    assert items[0] == {"transcript": "say something"}
    # Last item carries the rendered audio.
    assert isinstance(items[-1], dict) and "audio" in items[-1]
    assert items[-1]["audio"] == b"AUDIO:hello world foo"
    # The middle items are plain text deltas, in order, reconstructing the answer.
    deltas = [it for it in items[1:-1] if isinstance(it, str)]
    assert "".join(deltas).strip() == "hello world foo"


@pytest.mark.asyncio
async def test_stream_turn_speak_false_omits_audio():
    """stream_turn with speak=False still yields transcript + deltas, no audio."""
    speaker = FakeSpeaker()
    va = VoiceAgent(
        _agent(TestModel("just text")),
        transcriber=FakeTranscriber("hi"),
        speaker=speaker,
    )

    items = [item async for item in va.stream_turn(b"audio", speak=False)]

    assert items[0] == {"transcript": "hi"}
    assert not any(isinstance(it, dict) and "audio" in it for it in items)
    assert speaker.calls == []


@pytest.mark.asyncio
async def test_protocols_are_runtime_checkable():
    """The fakes structurally satisfy the Transcriber/Speaker protocols."""
    assert isinstance(FakeTranscriber(), Transcriber)
    assert isinstance(FakeSpeaker(), Speaker)


def test_imports_without_litellm(monkeypatch):
    """The module imports and default impls construct without litellm present.

    Only *calling* a default impl should require litellm; importing the module
    and instantiating must not, so offline test suites are unaffected.
    """
    real_import = builtins.__import__

    def no_litellm(name, *args, **kwargs):
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("litellm is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_litellm)

    import importlib

    import yaab.voice as voice_mod

    importlib.reload(voice_mod)
    # Construction is lazy: no litellm import happens here.
    t = voice_mod.LiteLLMTranscriber(model="whisper-1")
    s = voice_mod.LiteLLMSpeaker(model="tts-1")
    assert t.model == "whisper-1"
    assert s.model == "tts-1"


@pytest.mark.asyncio
async def test_litellm_transcriber_lazy_import_error(monkeypatch):
    """Calling the default transcriber without litellm raises a helpful error."""
    real_import = builtins.__import__

    def no_litellm(name, *args, **kwargs):
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_litellm)

    from yaab.exceptions import ModelError

    with pytest.raises(ModelError) as exc:
        await LiteLLMTranscriber(model="whisper-1").transcribe(b"audio")
    assert "litellm" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_litellm_speaker_lazy_import_error(monkeypatch):
    """Calling the default speaker without litellm raises a helpful error."""
    real_import = builtins.__import__

    def no_litellm(name, *args, **kwargs):
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_litellm)

    from yaab.exceptions import ModelError

    with pytest.raises(ModelError) as exc:
        await LiteLLMSpeaker(model="tts-1").speak("hello")
    assert "litellm" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_litellm_transcriber_uses_injected_client():
    """A fake litellm-like callable is awaited; its text is returned.

    Proves the provider call is injectable (no real litellm needed) and that we
    read the transcription text off the response object.
    """
    captured: dict = {}

    class _Resp:
        text = "transcribed text"

    async def fake_atranscription(**kwargs):
        captured.update(kwargs)
        return _Resp()

    t = LiteLLMTranscriber(model="whisper-1", _atranscription=fake_atranscription)
    out = await t.transcribe(b"raw-audio", format="mp3")

    assert out == "transcribed text"
    assert captured["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_litellm_speaker_uses_injected_client():
    """A fake litellm-like aspeech is awaited; its audio bytes are returned."""
    captured: dict = {}

    class _Resp:
        # litellm's HttpxBinaryResponseContent exposes .content
        content = b"SPOKEN"

    async def fake_aspeech(**kwargs):
        captured.update(kwargs)
        return _Resp()

    s = LiteLLMSpeaker(model="tts-1", _aspeech=fake_aspeech)
    out = await s.speak("hello", voice="nova")

    assert out == b"SPOKEN"
    assert captured["model"] == "tts-1"
    assert captured["voice"] == "nova"
    assert captured["input"] == "hello"


@pytest.mark.asyncio
async def test_live_voice_session_not_implemented():
    """The bidi LiveVoiceSession interface stub raises NotImplementedError.

    This documents the future-facing contract a real Gemini Live / OpenAI
    Realtime backend would implement; the error message must point there.
    """
    session = LiveVoiceSession()
    with pytest.raises(NotImplementedError) as exc:
        await session.connect()
    msg = str(exc.value).lower()
    assert "bidi" in msg or "realtime" in msg or "live" in msg

    with pytest.raises(NotImplementedError):
        await session.send_audio(b"chunk")

    with pytest.raises(NotImplementedError):
        # receive() is an async iterator; iterating must raise.
        async for _ in session.receive():
            pass


@pytest.mark.asyncio
async def test_voice_agent_open_live_raises():
    """VoiceAgent.open_live advertises the planned bidi path but isn't built."""
    va = VoiceAgent(
        _agent(TestModel("x")),
        transcriber=FakeTranscriber(),
        speaker=FakeSpeaker(),
    )
    with pytest.raises(NotImplementedError):
        await va.open_live()
