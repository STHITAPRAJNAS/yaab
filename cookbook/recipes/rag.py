"""Recipe: RAG — ground answers in your own documents, with citations.

A ``KnowledgeBase`` chunks, embeds, and retrieves; an agent answers from the
retrieved context. Offline uses the deterministic hashing embedder.

    python -m cookbook.recipes.rag
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import KnowledgeBase

_HELP_DOCS = [
    ("refunds.md", "Refunds are processed within 5 business days to the original payment method."),
    ("shipping.md", "Standard shipping takes 3-5 business days; express ships next day."),
    ("returns.md", "Items can be returned within 30 days with the order number."),
]


async def run() -> dict:
    kb = KnowledgeBase(name="helpcenter")
    for source, text in _HELP_DOCS:
        kb.add_text(text, source=source)

    hits = await kb.retrieve("how long do refunds take?", k=1)
    expect(bool(hits), "expected at least one retrieved chunk")
    top = hits[0]
    expect("refund" in top.chunk.text.lower(), f"expected the refunds doc, got {top.chunk.text!r}")
    return {"citation": top.chunk.source, "text": top.chunk.text}


if __name__ == "__main__":
    print(asyncio.run(run()))
