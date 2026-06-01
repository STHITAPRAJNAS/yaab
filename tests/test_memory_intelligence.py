"""Tests for memory intelligence.

Covers:
- LLM-based memory extraction (distilled memories, not raw lines),
- consolidation that dedups near-duplicate memories,
- backward-compatible raw copying when ``extract=False``,
- ``KnowledgeBaseMemory`` add/search round-trip with namespace filtering,
- ``MemoryManager`` working when handed a ``KnowledgeBaseMemory`` service.

All tests are network-free: they drive a ``FunctionModel``/``TestModel`` and the
deterministic hashing embedder.
"""

from __future__ import annotations

import json

import pytest

from yaab.memory import MemoryExtractor, hashing_embedder
from yaab.memory.manager import MemoryManager
from yaab.models.test_model import FunctionModel, TestModel
from yaab.rag import KnowledgeBase, KnowledgeBaseMemory
from yaab.sessions.base import Session
from yaab.types import Message, Role


def _session(*pairs: tuple[Role, str]) -> Session:
    return Session(messages=[Message(role=r, content=c) for r, c in pairs])


# --- Feature A: extraction ---------------------------------------------
@pytest.mark.asyncio
async def test_extractor_returns_distilled_memories():
    """The extractor distills durable facts via one LLM call returning a JSON
    array of short statements — not the raw conversation lines."""
    memories = ["User prefers dark mode.", "User lives in Berlin."]
    model = FunctionModel(lambda msgs: json.dumps(memories))
    extractor = MemoryExtractor(model)
    session = _session(
        (Role.USER, "Hey, please always use dark mode, I find it easier on the eyes."),
        (Role.ASSISTANT, "Sure, I'll keep things dark. Anything else?"),
        (Role.USER, "I'm based in Berlin by the way."),
    )

    out = await extractor.extract(session.messages)

    assert out == memories
    # Distilled, not verbatim copies of the raw lines.
    assert all(m not in {msg.content for msg in session.messages} for m in out)


@pytest.mark.asyncio
async def test_extractor_tolerates_markdown_fences():
    """Providers often wrap JSON in ```json fences despite the instruction; the
    parser must tolerate them (reusing parse_partial_json)."""
    fenced = "```json\n" + json.dumps(["User is vegetarian."]) + "\n```"
    model = FunctionModel(lambda msgs: fenced)
    extractor = MemoryExtractor(model)

    out = await extractor.extract([Message(role=Role.USER, content="I don't eat meat.")])

    assert out == ["User is vegetarian."]


@pytest.mark.asyncio
async def test_extractor_empty_on_garbage():
    """A non-JSON / unparseable response yields no memories rather than raising."""
    model = FunctionModel(lambda msgs: "I could not find anything noteworthy.")
    extractor = MemoryExtractor(model)

    out = await extractor.extract([Message(role=Role.USER, content="hello")])

    assert out == []


@pytest.mark.asyncio
async def test_extractor_ignores_non_string_items():
    """Defensive parsing: only string items survive, others are dropped."""
    model = FunctionModel(lambda msgs: json.dumps(["valid memory", 42, {"k": "v"}, None]))
    extractor = MemoryExtractor(model)

    out = await extractor.extract([Message(role=Role.USER, content="x")])

    assert out == ["valid memory"]


# --- Feature A: consolidation / dedup ----------------------------------
@pytest.mark.asyncio
async def test_consolidation_dedups_near_duplicates():
    """An extracted memory near-identical to an existing one is skipped."""
    manager = MemoryManager(embedder=hashing_embedder())
    await manager.add("User prefers dark mode", app_name="app", user_id="u1")

    # FunctionModel emits a near-duplicate of the stored memory.
    model = FunctionModel(lambda msgs: json.dumps(["User prefers dark mode"]))
    extractor = MemoryExtractor(model)
    session = _session((Role.USER, "I really like dark mode"))

    records = await manager.add_session_to_memory(
        session, app_name="app", user_id="u1", extractor=extractor
    )

    # The near-duplicate was consolidated away (nothing new stored).
    assert records == []
    hits = await manager.search("dark mode", app_name="app", user_id="u1", k=10)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_consolidation_keeps_distinct_memories():
    """A genuinely new memory is stored even when other memories exist."""
    manager = MemoryManager(embedder=hashing_embedder())
    await manager.add("User prefers dark mode", app_name="app", user_id="u1")

    model = FunctionModel(lambda msgs: json.dumps(["User lives in Tokyo"]))
    extractor = MemoryExtractor(model)
    session = _session((Role.USER, "I moved to Tokyo last year"))

    records = await manager.add_session_to_memory(
        session, app_name="app", user_id="u1", extractor=extractor
    )

    assert len(records) == 1
    assert records[0].text == "User lives in Tokyo"


@pytest.mark.asyncio
async def test_extracted_memories_scoped_to_namespace():
    """Consolidation only dedups within the same (app, user) namespace."""
    manager = MemoryManager(embedder=hashing_embedder())
    # Same memory text, but a different user — must NOT block storage.
    await manager.add("User prefers dark mode", app_name="app", user_id="other")

    model = FunctionModel(lambda msgs: json.dumps(["User prefers dark mode"]))
    extractor = MemoryExtractor(model)
    session = _session((Role.USER, "dark mode please"))

    records = await manager.add_session_to_memory(
        session, app_name="app", user_id="u1", extractor=extractor
    )

    assert len(records) == 1
    mine = await manager.search("dark mode", app_name="app", user_id="u1", k=10)
    assert len(mine) == 1


# --- Feature A: backward compatibility ---------------------------------
@pytest.mark.asyncio
async def test_extract_false_keeps_raw_copying():
    """Default behavior (extract=False) copies raw user/assistant lines verbatim."""
    manager = MemoryManager(embedder=hashing_embedder())
    session = _session(
        (Role.USER, "What is the capital of France?"),
        (Role.ASSISTANT, "Paris."),
        (Role.SYSTEM, "ignored"),
    )

    records = await manager.add_session_to_memory(session, app_name="app", user_id="u1")

    texts = [r.text for r in records]
    assert texts == ["What is the capital of France?", "Paris."]


@pytest.mark.asyncio
async def test_extract_true_uses_default_extractor_model():
    """extract=True with a model builds an extractor and stores distilled lines."""
    manager = MemoryManager(embedder=hashing_embedder())
    model = TestModel(custom_output=json.dumps(["User asked about France's capital."]))
    session = _session(
        (Role.USER, "What is the capital of France?"),
        (Role.ASSISTANT, "Paris."),
    )

    records = await manager.add_session_to_memory(
        session, app_name="app", user_id="u1", extract=True, model=model
    )

    assert [r.text for r in records] == ["User asked about France's capital."]


# --- Feature B: KnowledgeBaseMemory ------------------------------------
@pytest.mark.asyncio
async def test_kb_memory_add_search_roundtrip():
    """add() then search() round-trips through the KnowledgeBase backend."""
    kb = KnowledgeBase(embedder=hashing_embedder())
    mem = KnowledgeBaseMemory(kb)

    rec = await mem.add("The sky is blue", metadata={"app_name": "app", "user_id": "u1"})
    assert rec.text == "The sky is blue"

    hits = await mem.search("sky", k=5)
    assert hits
    assert hits[0][0].text == "The sky is blue"
    assert isinstance(hits[0][1], float)


@pytest.mark.asyncio
async def test_kb_memory_namespace_filtering():
    """search() honors app_name/user_id namespace kwargs (Runner threading)."""
    kb = KnowledgeBase(embedder=hashing_embedder())
    mem = KnowledgeBaseMemory(kb)
    await mem.add("alice secret", metadata={"app_name": "app", "user_id": "alice"})
    await mem.add("bob secret", metadata={"app_name": "app", "user_id": "bob"})

    alice_hits = await mem.search("secret", app_name="app", user_id="alice", k=10)
    texts = {rec.text for rec, _ in alice_hits}
    assert "alice secret" in texts
    assert "bob secret" not in texts


@pytest.mark.asyncio
async def test_kb_memory_search_signature_exposes_namespace_kwargs():
    """The Runner inspects search()'s signature for app_name/user_id; they must
    be real named parameters, not just **kwargs."""
    import inspect

    mem = KnowledgeBaseMemory(KnowledgeBase(embedder=hashing_embedder()))
    params = inspect.signature(mem.search).parameters
    assert "app_name" in params
    assert "user_id" in params


@pytest.mark.asyncio
async def test_manager_with_kb_memory_service():
    """MemoryManager works when given a KnowledgeBaseMemory as its service."""
    kb = KnowledgeBase(embedder=hashing_embedder())
    manager = MemoryManager(service=KnowledgeBaseMemory(kb))

    await manager.add("User likes hiking", app_name="app", user_id="u1")
    await manager.add("Other user note", app_name="app", user_id="u2")

    hits = await manager.search("hiking", app_name="app", user_id="u1", k=5)
    assert hits
    assert hits[0][0].text == "User likes hiking"
    # Cross-user note is filtered out by the manager's namespace scoping.
    assert all(rec.metadata.get("user_id") == "u1" for rec, _ in hits)


@pytest.mark.asyncio
async def test_kb_memory_persists_across_instances_sharing_store():
    """Durability: two KnowledgeBaseMemory views over one store see each other's
    writes (the durable-backend property, vs process-local InMemoryVectorMemory)."""
    from yaab.rag.store import InMemoryVectorStore

    store = InMemoryVectorStore()
    writer = KnowledgeBaseMemory(KnowledgeBase(embedder=hashing_embedder(), store=store))
    reader = KnowledgeBaseMemory(KnowledgeBase(embedder=hashing_embedder(), store=store))

    await writer.add("durable fact", metadata={"app_name": "app", "user_id": "u1"})
    hits = await reader.search("durable", k=5)
    assert hits and hits[0][0].text == "durable fact"
