"""Document chunkers — split a Document into retrievable Chunks.

A :class:`Chunker` is a protocol so callers can drop in their own (token-based,
markdown-aware, semantic). Three dependency-free strategies ship:

* :class:`CharacterChunker` — fixed character windows with overlap (the safe
  default; deterministic and provider-free);
* :class:`SentenceChunker`  — pack whole sentences up to a size budget;
* :class:`ParagraphChunker` — split on blank lines, then fall back to character
  windows for oversized paragraphs.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from .types import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    def split(self, document: Document) -> list[Chunk]:
        ...


def _make_chunks(document: Document, texts: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        # Carry source + document id into metadata so vector stores can filter
        # on them (delete-by-source, document-level access control).
        meta = dict(document.metadata)
        if document.source is not None:
            meta.setdefault("source", document.source)
        meta.setdefault("document_id", document.id)
        chunks.append(
            Chunk(
                text=text,
                document_id=document.id,
                source=document.source,
                index=i,
                metadata=meta,
            )
        )
    return chunks


class CharacterChunker:
    """Fixed-size character windows with overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        if len(text) <= self.chunk_size:
            return _make_chunks(document, [text]) if text.strip() else []
        step = self.chunk_size - self.overlap
        windows = [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
        # Drop a trailing window fully contained in the previous one.
        if len(windows) >= 2 and windows[-1] in windows[-2]:
            windows.pop()
        return _make_chunks(document, windows)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker:
    """Pack whole sentences up to ``chunk_size`` characters."""

    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunk_size = chunk_size

    def split(self, document: Document) -> list[Chunk]:
        sentences = _SENTENCE_RE.split(document.text.strip())
        out: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > self.chunk_size:
                out.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current.strip():
            out.append(current.strip())
        return _make_chunks(document, out)


class ParagraphChunker:
    """Split on blank lines; oversized paragraphs fall back to character windows."""

    def __init__(self, max_size: int = 1500, overlap: int = 100) -> None:
        self.max_size = max_size
        self._char = CharacterChunker(chunk_size=max_size, overlap=overlap)

    def split(self, document: Document) -> list[Chunk]:
        paragraphs = re.split(r"\n\s*\n", document.text.strip())
        texts: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.max_size:
                texts.append(para)
            else:
                sub = Document(text=para, source=document.source, metadata=document.metadata)
                texts.extend(c.text for c in self._char.split(sub))
        return _make_chunks(document, texts)


__all__ = ["Chunker", "CharacterChunker", "SentenceChunker", "ParagraphChunker"]
