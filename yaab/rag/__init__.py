"""Built-in, provider-neutral RAG (retrieval-augmented generation).

Unlike SDKs that delegate RAG to a managed cloud service, YAAB ships the whole
pipeline as open, swappable components — and includes governance pieces:
per-user/document access control at retrieval, source citations, embedding
caching, incremental dedup indexing, retrieval guardrails, and RAG faithfulness
evaluation.

    from yaab import Agent
    from yaab.rag import KnowledgeBase, Document

    kb = KnowledgeBase()
    kb.add(Document(text="Paris is the capital of France.", source="geo.md"))
    agent = Agent("assistant", model="openai/gpt-4o", tools=[kb.as_tool()])
"""

from __future__ import annotations

from .chunking import (
    CharacterChunker,
    Chunker,
    ParagraphChunker,
    SentenceChunker,
)
from .eval import FaithfulnessEvaluator, context_relevance, faithfulness
from .knowledge import KnowledgeBase
from .loaders import load, load_bytes, load_directory
from .memory_service import KnowledgeBaseMemory
from .rerank import CrossEncoderReranker, KeywordReranker, LLMReranker, Reranker
from .store import InMemoryVectorStore, PgVectorStore, VectorStore
from .types import Chunk, Document, RetrievedChunk

__all__ = [
    "Document",
    "Chunk",
    "RetrievedChunk",
    "Chunker",
    "CharacterChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "VectorStore",
    "InMemoryVectorStore",
    "PgVectorStore",
    "Reranker",
    "KeywordReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "KnowledgeBase",
    # durable, vector-store-backed long-term memory (MemoryService impl)
    "KnowledgeBaseMemory",
    # document loaders
    "load",
    "load_directory",
    "load_bytes",
    # RAG evaluation (groundedness / faithfulness)
    "faithfulness",
    "context_relevance",
    "FaithfulnessEvaluator",
]


def __getattr__(name: str):
    # Lazy access to external stores (avoid importing their client libs eagerly).
    if name in (
        "ChromaVectorStore",
        "QdrantVectorStore",
        "OpenSearchVectorStore",
        "OracleVectorStore",
        "PineconeVectorStore",
        "WeaviateVectorStore",
    ):
        from . import stores_external

        return getattr(stores_external, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
