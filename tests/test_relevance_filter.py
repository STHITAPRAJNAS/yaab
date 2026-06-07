"""RelevanceFilter context strategy: keep only history that earns its place.

Unlike truncate (drop oldest) or summarize (compress), RelevanceFilter scores
each prior turn for relevance to the latest user message and drops the
irrelevant ones — so a long, topic-switching conversation doesn't carry noise
into every model call. The relevance scorer is injectable, so it runs offline.
"""

from __future__ import annotations

import pytest

from yaab.context import RelevanceFilter
from yaab.types import Message, Role


def _msgs() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are helpful."),
        Message(role=Role.USER, content="Tell me about the Eiffel Tower."),
        Message(role=Role.ASSISTANT, content="The Eiffel Tower is in Paris."),
        Message(role=Role.USER, content="What is the capital of Japan?"),
        Message(role=Role.ASSISTANT, content="Tokyo is the capital of Japan."),
        Message(role=Role.USER, content="How tall is the Eiffel Tower?"),
    ]


@pytest.mark.asyncio
async def test_keeps_system_and_latest_always():
    # A scorer that calls everything irrelevant must still keep the system prompt
    # and the latest user message (you cannot drop the question you must answer).
    strat = RelevanceFilter(scorer=lambda query, text: 0.0, min_score=0.5)
    out = await strat.apply(_msgs())
    assert out[0].role is Role.SYSTEM
    assert out[-1].content == "How tall is the Eiffel Tower?"


@pytest.mark.asyncio
async def test_drops_irrelevant_turns_keyword_default():
    # The default scorer is keyword overlap with the latest user message.
    out = await RelevanceFilter(min_score=0.1).apply(_msgs())
    texts = [m.content for m in out]
    # Eiffel-Tower turns are relevant to "How tall is the Eiffel Tower?"; the
    # Japan/Tokyo turns are not and should be dropped.
    assert any("Eiffel Tower is in Paris" in t for t in texts)
    assert not any("Tokyo" in t for t in texts)


@pytest.mark.asyncio
async def test_keeps_everything_when_threshold_is_zero():
    out = await RelevanceFilter(min_score=0.0).apply(_msgs())
    assert len(out) == len(_msgs())


@pytest.mark.asyncio
async def test_no_user_message_is_a_noop():
    only_system = [Message(role=Role.SYSTEM, content="hi")]
    out = await RelevanceFilter().apply(only_system)
    assert out == only_system
