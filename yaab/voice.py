"""Turn-based voice agents: audio in -> agent -> audio out.

Fully *bidirectional* audio (the user speaks while the model speaks back, with
barge-in, as in the Gemini Live / OpenAI Realtime APIs) needs a persistent
realtime WebSocket the model server drives — something LiteLLM (a
request/response HTTP client over chat completions) cannot do. Rather than fake
it, this module ships the honest, genuinely useful subset:

    speech-to-text  ->  the normal YAAB agent loop  ->  text-to-speech

i.e. a **turn-based** voice pipeline. The user finishes speaking, we transcribe,
the agent answers (with full tool use, memory, governance — everything a text
run gets), and we render the answer to audio. :meth:`VoiceAgent.stream_turn`
streams the transcript and the text answer as it generates so a UI stays
responsive while TTS renders.

The interface is deliberately shaped so a *future* bidi backend (Gemini Live /
OpenAI Realtime) can implement the same surface: see :class:`LiveVoiceSession`
and :meth:`VoiceAgent.open_live`, which document the planned streaming contract
and raise ``NotImplementedError`` today. When such a backend lands, callers
switch from ``speak_turn``/``stream_turn`` to ``open_live`` without relearning
the model.

Provider calls (Whisper STT, OpenAI/ElevenLabs TTS) go through the
:class:`Transcriber` / :class:`Speaker` protocols. The defaults are LiteLLM
backed and import ``litellm`` *lazily* — so this module, and tests using fake
transcribers/speakers, work with no ``litellm`` install and no network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import ModelError
from .types import EventType, Usage

if TYPE_CHECKING:  # avoid importing the heavy agent module at import time
    from .agent import Agent


@runtime_checkable
class Transcriber(Protocol):
    """Speech-to-text: turn recorded audio into a text transcript.

    Implementations must be async because real STT is a network call; keeping it
    a one-method protocol means a test can pass a trivial fake (and a future
    on-device Whisper can drop in) without subclassing anything.
    """

    async def transcribe(self, audio_bytes: bytes, *, format: str = "wav") -> str:
        """Transcribe ``audio_bytes`` (encoded as ``format``) to text."""
        ...


@runtime_checkable
class Speaker(Protocol):
    """Text-to-speech: render text into spoken audio bytes."""

    async def speak(self, text: str, *, voice: str = "alloy") -> bytes:
        """Synthesize ``text`` into audio bytes using ``voice``."""
        ...


def _require_litellm() -> Any:
    """Import ``litellm`` lazily with a helpful error if it's missing.

    Mirrors the model layer's contract: the SDK (and any fake STT/TTS) works
    offline; only *invoking* a default LiteLLM-backed impl needs the extra.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ModelError(
            "litellm is not installed. Install the voice/model layer with "
            "`pip install 'yaab-sdk[litellm]'`, or inject a custom "
            "Transcriber/Speaker (e.g. for offline tests)."
        ) from exc
    return litellm


class LiteLLMTranscriber:
    """Default :class:`Transcriber` over ``litellm.atranscription`` (Whisper).

    ``litellm`` is imported only when :meth:`transcribe` is first called, so
    constructing this is free and offline-safe. ``_atranscription`` is an
    injection seam for tests: pass an async callable to bypass litellm entirely.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        *,
        api_key: str | None = None,
        _atranscription: Callable[..., Awaitable[Any]] | None = None,
        **params: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.params = params
        self._atranscription = _atranscription

    async def transcribe(self, audio_bytes: bytes, *, format: str = "wav") -> str:
        atranscription = self._atranscription
        if atranscription is None:
            atranscription = _require_litellm().atranscription
        # litellm's atranscription wants a file-like object with a name so the
        # provider can infer the container format from the extension.
        import io

        buf = io.BytesIO(audio_bytes)
        buf.name = f"audio.{format}"
        kwargs: dict[str, Any] = {"model": self.model, "file": buf, **self.params}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        resp = await atranscription(**kwargs)
        # litellm returns a TranscriptionResponse with a ``text`` attribute;
        # tolerate a plain dict/string for forward-compat and simple fakes.
        text = getattr(resp, "text", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("text")
        return text if isinstance(text, str) else str(resp)


class LiteLLMSpeaker:
    """Default :class:`Speaker` over ``litellm.aspeech`` (OpenAI TTS).

    Lazy ``litellm`` import (same contract as :class:`LiteLLMTranscriber`);
    ``_aspeech`` is the test injection seam.
    """

    def __init__(
        self,
        model: str = "tts-1",
        *,
        voice: str = "alloy",
        response_format: str = "mp3",
        api_key: str | None = None,
        _aspeech: Callable[..., Awaitable[Any]] | None = None,
        **params: Any,
    ) -> None:
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.api_key = api_key
        self.params = params
        self._aspeech = _aspeech

    async def speak(self, text: str, *, voice: str | None = None) -> bytes:
        aspeech = self._aspeech
        if aspeech is None:
            aspeech = _require_litellm().aspeech
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": voice or self.voice,
            "response_format": self.response_format,
            **self.params,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        resp = await aspeech(**kwargs)
        # litellm wraps the audio in an HttpxBinaryResponseContent exposing
        # ``.content``; tolerate raw bytes for simple fakes/forward-compat.
        if isinstance(resp, bytes | bytearray):
            return bytes(resp)
        content = getattr(resp, "content", None)
        if isinstance(content, bytes | bytearray):
            return bytes(content)
        # Some versions stream via ``read()``.
        reader = getattr(resp, "read", None)
        if callable(reader):
            data = reader()
            if isinstance(data, bytes | bytearray):
                return bytes(data)
        raise ModelError(f"speaker returned non-audio response: {type(resp)!r}")


class VoiceTurn(BaseModel):
    """One completed voice exchange: what the user said and what came back.

    ``response_audio`` is ``None`` when TTS was skipped (``speak=False``), so a
    caller can request a text-only answer (e.g. to render captions only) without
    paying for synthesis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transcript: str
    response_text: str
    response_audio: bytes | None = None
    usage: Usage = Field(default_factory=Usage)


class LiveVoiceSession:
    """Planned bidirectional (realtime) voice session — NOT yet implemented.

    This is the interface a future streaming backend (Gemini Live API or OpenAI
    Realtime API) will implement so YAAB can offer live conversational audio:
    full-duplex audio where the user can interrupt the model mid-utterance
    (barge-in), with the server driving a persistent WebSocket.

    The contract a real implementation must provide:

    * :meth:`connect` — open the realtime socket and start a session.
    * :meth:`send_audio` — push a chunk of user audio as it is captured (the
      user keeps talking; no "finish the turn" boundary required).
    * :meth:`receive` — async-iterate server events: partial transcripts,
      model text, and synthesized audio chunks, interleaved as they arrive.

    Today every method raises :class:`NotImplementedError` with a pointer to the
    turn-based path (:meth:`VoiceAgent.speak_turn` / :meth:`VoiceAgent.stream_turn`),
    which is fully functional. Shipping the *interface* now means application
    code can be written against it and a bidi backend can land later without a
    breaking change.
    """

    _MESSAGE = (
        "Bidirectional (realtime) voice streaming is not implemented: it requires "
        "a Gemini Live / OpenAI Realtime backend (planned), which needs a "
        "persistent server-driven WebSocket that LiteLLM cannot provide. Use the "
        "turn-based pipeline today: VoiceAgent.speak_turn() or .stream_turn()."
    )

    async def connect(self) -> None:
        """Open the realtime session. Planned; raises today."""
        raise NotImplementedError(self._MESSAGE)

    async def send_audio(self, chunk: bytes) -> None:
        """Send a user-audio chunk to the live session. Planned; raises today."""
        raise NotImplementedError(self._MESSAGE)

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Async-iterate live server events. Planned; raises today."""
        raise NotImplementedError(self._MESSAGE)
        # An ``async def`` with no ``yield`` would be a coroutine, not an async
        # generator, so callers couldn't ``async for`` it. The yield below is
        # unreachable but makes this a proper async iterator that raises on use.
        yield {}  # pragma: no cover


class VoiceAgent:
    """Wrap any YAAB :class:`~yaab.agent.Agent` as a turn-based voice agent.

    Composes a :class:`Transcriber` and :class:`Speaker` around the agent so a
    blob of recorded audio becomes a spoken answer. The agent runs exactly as it
    would for text — tools, sessions/memory, governance, structured output all
    apply — which is why a ``session_id`` threaded across turns gives a voice
    conversation real memory.

        va = VoiceAgent(agent, transcriber=LiteLLMTranscriber(), speaker=LiteLLMSpeaker())
        turn = await va.speak_turn(wav_bytes, session_id="call-42")
        play(turn.response_audio)
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        transcriber: Transcriber,
        speaker: Speaker,
        voice: str = "alloy",
        audio_format: str = "wav",
    ) -> None:
        self.agent = agent
        self.transcriber = transcriber
        self.speaker = speaker
        #: Default TTS voice for answers; overridable per call.
        self.voice = voice
        #: Encoding of the *input* audio handed to the transcriber.
        self.audio_format = audio_format

    async def speak_turn(
        self,
        audio_bytes: bytes,
        *,
        session_id: str | None = None,
        deps: Any = None,
        speak: bool = True,
        voice: str | None = None,
        identity: str | None = None,
    ) -> VoiceTurn:
        """Run one full voice turn: transcribe -> agent -> (optionally) speak.

        ``session_id`` threads conversation history so multi-turn voice chats
        remember context. ``speak=False`` returns a text-only answer (no TTS
        call, ``response_audio is None``).
        """
        transcript = await self.transcriber.transcribe(audio_bytes, format=self.audio_format)
        result = await self.agent.run(
            transcript, deps=deps, session_id=session_id, identity=identity
        )
        response_text = _as_text(result.output)
        audio: bytes | None = None
        if speak:
            audio = await self.speaker.speak(response_text, voice=voice or self.voice)
        return VoiceTurn(
            transcript=transcript,
            response_text=response_text,
            response_audio=audio,
            usage=result.usage,
        )

    async def stream_turn(
        self,
        audio_bytes: bytes,
        *,
        session_id: str | None = None,
        deps: Any = None,
        speak: bool = True,
        voice: str | None = None,
        identity: str | None = None,
    ) -> AsyncIterator[dict[str, Any] | str]:
        """Stream a voice turn for responsive UIs.

        Yields, in order:

        1. ``{"transcript": <text>}`` once STT completes — the UI can show what
           the user said immediately.
        2. plain ``str`` text deltas as the agent generates its answer (so the
           UI streams captions live while it thinks).
        3. ``{"audio": <bytes>}`` last (omitted when ``speak=False``) — the
           rendered TTS for the full answer.

        TTS is turn-based, so the audio is synthesized once the text answer is
        complete; the early transcript + deltas keep the experience live.
        """
        transcript = await self.transcriber.transcribe(audio_bytes, format=self.audio_format)
        yield {"transcript": transcript}

        parts: list[str] = []
        final_text: str | None = None
        async for event in self.agent.stream_events(
            transcript, deps=deps, session_id=session_id, identity=identity
        ):
            if event.type is EventType.TEXT_DELTA:
                delta = event.payload.get("delta", "")
                if delta:
                    parts.append(delta)
                    yield delta
            elif event.type is EventType.RUN_END:
                # Authoritative final answer (already governance-scanned/coerced);
                # prefer it over the concatenated deltas for the audio render.
                final_text = _as_text(event.payload["result"].output)

        if speak:
            text = final_text if final_text is not None else "".join(parts)
            audio = await self.speaker.speak(text, voice=voice or self.voice)
            yield {"audio": audio}

    async def open_live(self) -> LiveVoiceSession:
        """Open a bidirectional realtime voice session — planned, not built.

        This is the entry point a future Gemini Live / OpenAI Realtime backend
        will implement for true full-duplex (barge-in) voice. It raises
        :class:`NotImplementedError` today; use :meth:`speak_turn` /
        :meth:`stream_turn` for the working turn-based pipeline.
        """
        raise NotImplementedError(LiveVoiceSession._MESSAGE)


def _as_text(output: Any) -> str:
    """Coerce an agent's output (str or a structured model) to spoken text."""
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        # A structured answer: speak its JSON rather than a repr. Callers wanting
        # nicer narration should give the agent a str output_type.
        return output.model_dump_json()
    return str(output)


__all__ = [
    "Transcriber",
    "Speaker",
    "LiteLLMTranscriber",
    "LiteLLMSpeaker",
    "VoiceTurn",
    "VoiceAgent",
    "LiveVoiceSession",
]
