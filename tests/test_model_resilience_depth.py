"""Retry-After honoring (G1) + cached-token accounting (G2).

Both are exercised against fakes (no network): a fake litellm whose acompletion
raises a rate-limit error carrying retry_after, and a fake response whose usage
includes cached-token details.
"""

from __future__ import annotations

import pytest

from yaab.models.litellm_provider import LiteLLMModel, _retry_after_seconds
from yaab.types import Message, Role, Usage


# --- G1: Retry-After parsing -------------------------------------------
def test_retry_after_from_attribute():
    class Err(Exception):
        retry_after = 7

    assert _retry_after_seconds(Err()) == 7.0


def test_retry_after_from_message_text():
    err = Exception("RateLimitError: try again in 12 seconds")
    assert _retry_after_seconds(err) == 12.0


def test_retry_after_from_headers():
    class Err(Exception):
        response_headers = {"retry-after": "5"}

    assert _retry_after_seconds(Err()) == 5.0


def test_retry_after_none_when_absent():
    assert _retry_after_seconds(Exception("some other error")) is None


@pytest.mark.asyncio
async def test_retry_honors_retry_after(monkeypatch):
    # A fake litellm that fails once with a retry_after, then succeeds. We assert
    # the backoff used the retry_after value rather than the default schedule.
    slept: list[float] = []

    class _RL(Exception):
        retry_after = 3

    calls = {"n": 0}

    class FakeMsg:
        content = "ok"
        tool_calls = None

    class FakeChoice:
        finish_reason = "stop"
        message = FakeMsg()

    class FakeResp:
        choices = [FakeChoice()]
        usage = None

    class FakeLiteLLM:
        async def acompletion(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RL("rate limited")
            return FakeResp()

        def completion_cost(self, **kw):
            return 0.0

    model = LiteLLMModel("x/y", max_retries=2, retry_base_delay=0.5)
    monkeypatch.setattr(
        "yaab.models.litellm_provider._require_litellm", lambda: FakeLiteLLM()
    )

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("yaab.models.litellm_provider.asyncio.sleep", fake_sleep)

    resp = await model.complete([Message(role=Role.USER, content="hi")])
    assert resp.content == "ok"
    # Backoff used the provider's retry_after (3s), not the default 0.5s base.
    assert slept and slept[0] == 3.0


# --- G2: cached-token accounting ---------------------------------------
def test_usage_tracks_cached_tokens():
    u = Usage(input_tokens=100, cached_input_tokens=80)
    u2 = Usage(input_tokens=50, cached_input_tokens=40)
    u.add(u2)
    assert u.cached_input_tokens == 120


def test_normalize_captures_cached_tokens():
    model = LiteLLMModel("x/y", track_cost=False)

    class Details:
        cached_tokens = 64

    class RawUsage:
        prompt_tokens = 100
        completion_tokens = 20
        total_tokens = 120
        prompt_tokens_details = Details()

    class Msg:
        content = "hi"
        tool_calls = None
        reasoning_content = None

    class Choice:
        finish_reason = "stop"
        message = Msg()

    class Resp:
        choices = [Choice()]
        usage = RawUsage()

    out = model._normalize(Resp(), "x/y", None)
    assert out.usage.input_tokens == 100
    assert out.usage.cached_input_tokens == 64
