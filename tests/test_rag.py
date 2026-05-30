"""Tests for built-in RAG: chunking, store, knowledge base, retriever tool."""

from __future__ import annotations

import pytest

from yaab import Agent, Document, KnowledgeBase
from yaab.memory.embedders import CachingEmbedder
from yaab.models.test_model import TestModel
from yaab.rag import (
    CharacterChunker,
    InMemoryVectorStore,
    KeywordReranker,
    ParagraphChunker,
    SentenceChunker,
)
from yaab.rag.types import Document as Doc


# --- chunking ----------------------------------------------------------
def test_character_chunker_overlap():
    text = "abcdefghij" * 30  # 300 chars
    chunks = CharacterChunker(chunk_size=100, overlap=20).split(Doc(text=text))
    assert len(chunks) >= 3
    assert all(len(c.text) <= 100 for c in chunks)
    # chunks carry source/document_id into metadata
    assert chunks[0].metadata["document_id"] == chunks[0].document_id


def test_character_chunker_small_doc():
    chunks = CharacterChunker(chunk_size=1000).split(Doc(text="short"))
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_sentence_chunker_packs_sentences():
    text = "One. Two. Three. Four. Five."
    chunks = SentenceChunker(chunk_size=12).split(Doc(text=text))
    assert len(chunks) >= 2


def test_paragraph_chunker_splits_on_blank_lines():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = ParagraphChunker().split(Doc(text=text))
    assert len(chunks) == 3


# --- vector store ------------------------------------------------------
@pytest.mark.asyncio
async def test_knowledge_base_add_and_retrieve():
    kb = KnowledgeBase()
    kb.add(Document(text="Paris is the capital of France.", source="geo.md"))
    kb.add(Document(text="Bananas are a yellow fruit.", source="food.md"))
    results = await kb.retrieve("What is the capital of France?", k=1)
    assert results
    assert "Paris" in results[0].text
    assert results[0].citation().startswith("geo.md")


@pytest.mark.asyncio
async def test_metadata_filter_isolation():
    kb = KnowledgeBase()
    kb.add(Document(text="Alice's private note", source="a", metadata={"user": "alice"}))
    kb.add(Document(text="Bob's private note", source="b", metadata={"user": "bob"}))
    results = await kb.retrieve("note", k=5, where={"user": "alice"})
    assert results
    assert all(r.chunk.metadata["user"] == "alice" for r in results)


def test_dedup_skips_repeated_content():
    kb = KnowledgeBase()
    n1 = kb.add(Document(text="same content here", source="x"))
    n2 = kb.add(Document(text="same content here", source="x"))  # identical
    assert n1 == 1
    assert n2 == 0  # deduped
    assert kb.count() == 1


def test_delete_and_reindex_by_source():
    kb = KnowledgeBase()
    kb.add(Document(text="version one of the doc", source="doc.md"))
    assert kb.count() == 1
    kb.reindex(Document(text="version two of the doc", source="doc.md"), source="doc.md")
    assert kb.count() == 1  # old source chunks replaced


# --- reranking ---------------------------------------------------------
@pytest.mark.asyncio
async def test_keyword_reranker_reorders():
    kb = KnowledgeBase(reranker=KeywordReranker(weight=0.9))
    kb.add(Document(text="The mitochondria is the powerhouse of the cell.", source="bio"))
    kb.add(Document(text="Paris hosts the Eiffel Tower and the Louvre museum.", source="geo"))
    results = await kb.retrieve("Eiffel Tower Paris Louvre", k=1)
    assert "Paris" in results[0].text


# --- embedding cache ---------------------------------------------------
def test_caching_embedder_avoids_recompute():
    calls = {"n": 0}

    def base(text: str) -> list[float]:
        calls["n"] += 1
        return [float(len(text))]

    emb = CachingEmbedder(base)
    emb("hello")
    emb("hello")
    emb("world")
    assert calls["n"] == 2  # "hello" embedded once
    assert emb.hits == 1
    assert emb.misses == 2


# --- retriever tool ----------------------------------------------------
@pytest.mark.asyncio
async def test_knowledge_base_as_tool():
    kb = KnowledgeBase(name="docs")
    kb.add(Document(text="The API key lives in the vault.", source="ops.md"))
    tool = kb.as_tool()
    assert tool.name == "search_docs"

    # The agent calls the retriever tool, then answers.
    model = TestModel(custom_output="It's in the vault.", call_tools=["search_docs"])
    agent = Agent("a", model=model, tools=[tool])
    result = await agent.run("Where is the API key?")
    assert result.output == "It's in the vault."


@pytest.mark.asyncio
async def test_augment_returns_context_with_citations():
    kb = KnowledgeBase()
    kb.add(Document(text="YAAB has built-in RAG.", source="readme.md"))
    block, results = await kb.augment("Does YAAB have RAG?", k=1)
    assert "YAAB has built-in RAG" in block
    assert "readme.md" in block
    assert len(results) == 1
