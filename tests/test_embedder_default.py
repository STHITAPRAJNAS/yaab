"""Default embedder auto-upgrade + warning + string shorthand (Phase A4).

No network in CI: auto-upgrade only *constructs* a LiteLLMEmbedder (it embeds
lazily on call), so selection is asserted by type, never by calling it.
"""

from __future__ import annotations

import logging

import pytest

import yaab.memory as memory
from yaab.memory.embedders import LiteLLMEmbedder

_EMBED_KEYS = [
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "VOYAGE_API_KEY",
]


@pytest.fixture
def _no_embed_keys(monkeypatch):
    for k in _EMBED_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(memory, "_warned_hashing", False, raising=False)


def test_default_is_hashing_without_key(_no_embed_keys):
    emb = memory.default_embedder()
    assert not isinstance(emb, LiteLLMEmbedder)
    # Deterministic stub: same text -> same vector.
    assert emb("hello world") == emb("hello world")


def test_default_autoupgrades_with_key(_no_embed_keys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = memory.default_embedder()
    assert isinstance(emb, LiteLLMEmbedder)
    assert emb.model == "openai/text-embedding-3-small"


def test_default_autoupgrades_gemini(_no_embed_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    emb = memory.default_embedder()
    assert isinstance(emb, LiteLLMEmbedder)
    assert "embedding" in emb.model


def test_hashing_warns_once(_no_embed_keys, caplog):
    with caplog.at_level(logging.WARNING, logger="yaab"):
        memory.default_embedder()
        memory.default_embedder()
    warnings = [r for r in caplog.records if "hashing embedder" in r.getMessage()]
    assert len(warnings) == 1  # one-time, not per call


def test_resolve_string_shorthand():
    emb = memory.resolve_embedder("cohere/embed-english-v3.0")
    assert isinstance(emb, LiteLLMEmbedder)
    assert emb.model == "cohere/embed-english-v3.0"


def test_resolve_passthrough_callable():
    sentinel = lambda t: [1.0]  # noqa: E731
    assert memory.resolve_embedder(sentinel) is sentinel


def test_knowledgebase_accepts_string_embedder():
    from yaab import KnowledgeBase

    kb = KnowledgeBase(embedder="openai/text-embedding-3-small")
    assert isinstance(kb.embedder, LiteLLMEmbedder)


def test_memory_manager_accepts_string_embedder():
    from yaab.memory import InMemoryVectorMemory

    store = InMemoryVectorMemory(embedder="mistral/mistral-embed")
    assert isinstance(store.embedder, LiteLLMEmbedder)
