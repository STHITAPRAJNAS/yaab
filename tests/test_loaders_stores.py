"""Tests for document loaders, external store registration, cross-encoder."""

from __future__ import annotations

import json

import pytest

from yaab.rag import load, load_directory
from yaab.rag.loaders import html_to_text, load_bytes


def test_load_text_and_markdown(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    docs = load(str(p))
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].metadata["format"] == "text"

    m = tmp_path / "b.md"
    m.write_text("# Title\n\nbody")
    assert load(str(m))[0].metadata["format"] == "markdown"


def test_load_csv_one_doc_per_row(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,role\nAlice,eng\nBob,pm\n")
    docs = load(str(p))
    assert len(docs) == 2
    assert "Alice" in docs[0].text and "eng" in docs[0].text
    assert docs[1].metadata["row"] == 1


def test_load_json_list(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    docs = load(str(p))
    assert len(docs) == 2
    assert docs[0].metadata["format"] == "json"


def test_html_to_text_strips_tags():
    html = "<html><body><script>x=1</script><p>Hello <b>world</b></p></body></html>"
    text = html_to_text(html)
    assert "Hello" in text and "world" in text
    assert "<p>" not in text and "x=1" not in text


def test_load_bytes_html():
    docs = load_bytes(b"<p>hi there</p>", source="upload", fmt="html")
    assert "hi there" in docs[0].text


def test_load_directory(tmp_path):
    (tmp_path / "a.md").write_text("doc a")
    (tmp_path / "b.txt").write_text("doc b")
    (tmp_path / "skip.bin").write_text("binary")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("doc c")
    docs = load_directory(str(tmp_path), glob="**/*")
    texts = {d.text for d in docs}
    assert {"doc a", "doc b", "doc c"} <= texts
    assert "binary" not in texts  # .bin not a known format


def test_external_stores_registered():
    from yaab.extensions import available

    names = available("vectorstore")
    assert "chroma" in names
    assert "qdrant" in names
    assert "memory" in names
    assert "pgvector" in names


def test_cross_encoder_with_injected_model():
    from yaab.rag import CrossEncoderReranker
    from yaab.rag.types import Chunk, RetrievedChunk

    class FakeCE:
        def predict(self, pairs):
            # Score by length of chunk text (deterministic, offline).
            return [len(c) for _, c in pairs]

    rr = CrossEncoderReranker(model=FakeCE())
    results = [
        RetrievedChunk(chunk=Chunk(text="short"), score=0.1),
        RetrievedChunk(chunk=Chunk(text="a much longer passage"), score=0.1),
    ]
    out = rr.rerank("q", results, top_n=1)
    assert out[0].chunk.text == "a much longer passage"


def test_external_store_missing_dep_raises():
    # chromadb isn't installed in CI; constructing should raise a clear error.
    from yaab.rag import ChromaVectorStore

    with pytest.raises(RuntimeError):
        ChromaVectorStore()
