"""Hybrid (sparse + dense) retrieval with reciprocal-rank fusion.

A dense (embedding) recall misses exact-term matches the default hashing
embedder can't represent; a sparse BM25 index catches them. With ``hybrid=True``
a KnowledgeBase runs both and fuses their rankings, so a rare keyword in the
query reliably surfaces the chunk that contains it.
"""

from __future__ import annotations

from yaab.rag import Document, KnowledgeBase
from yaab.rag.hybrid import BM25Index, reciprocal_rank_fusion


def test_bm25_ranks_exact_term_match_first():
    idx = BM25Index()
    idx.add("d1", "the cat sat on the mat")
    idx.add("d2", "a dog ran across the field")
    idx.add("d3", "quantum entanglement in physics")
    ranked = idx.search("quantum physics", k=3)
    assert ranked[0][0] == "d3"


def test_rrf_fuses_two_rankings():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "d"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    # 'a' (ranks 1 and 2) and 'c' (ranks 3 and 1) score highest; both beat b/d.
    top2 = {doc for doc, _ in fused[:2]}
    assert top2 == {"a", "c"}


def test_hybrid_knowledge_base_surfaces_keyword_match():
    kb = KnowledgeBase(hybrid=True)
    kb.add(
        [
            Document(text="The Eiffel Tower is a landmark in Paris.", source="a"),
            Document(text="Photosynthesis converts light into chemical energy.", source="b"),
            Document(text="The mitochondria is the powerhouse of the cell.", source="c"),
        ]
    )
    import asyncio

    results = asyncio.run(kb.retrieve("mitochondria powerhouse", k=1))
    assert results
    assert "mitochondria" in results[0].text.lower()


def test_non_hybrid_default_unchanged():
    kb = KnowledgeBase()  # hybrid defaults off
    kb.add_text("hello world", source="s")
    import asyncio

    results = asyncio.run(kb.retrieve("hello", k=1))
    assert results and "hello" in results[0].text.lower()
