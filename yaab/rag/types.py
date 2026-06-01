"""Core RAG types: Document, Chunk, and retrieval results.

Names follow the familiar `Document`/`Chunk` vocabulary so the mental model
transfers, while staying provider-neutral and Pydantic-typed for clean
(de)serialization into sessions, checkpoints, and the audit log.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document before chunking: raw text plus metadata.

    ``source`` is a stable identifier (path, URL, db id) used for lineage and
    document-level access control; ``metadata`` carries anything else
    (``app_name``, ``user_id``, tags, timestamps).
    """

    id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    text: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit produced by splitting a :class:`Document`.

    Carries an embedding (filled at index time) and a back-reference to its
    document so retrieved context can be attributed to a source (citations).
    """

    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    text: str
    document_id: str | None = None
    source: str | None = None
    index: int = 0  # position within the parent document
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, with its relevance score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    def citation(self) -> str:
        """A short, human-readable source attribution for this chunk."""
        src = self.chunk.source or self.chunk.document_id or "unknown"
        return f"{src}#{self.chunk.index}"


__all__ = ["Document", "Chunk", "RetrievedChunk"]
