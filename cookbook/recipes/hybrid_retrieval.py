"""Recipe: hybrid retrieval — fuse sparse (BM25) and dense recall.

``KnowledgeBase(hybrid=True)`` runs a BM25 keyword search alongside the dense
(embedding) search and fuses them by reciprocal rank, so exact rare terms surface
even when an embedding glosses over them.

    python -m cookbook.recipes.hybrid_retrieval
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import KnowledgeBase
from yaab.rag import Document


async def run() -> dict:
    kb = KnowledgeBase(name="kb", hybrid=True)
    kb.add(
        [
            Document(
                text="The Falcon X drone uses part SKU-7537 for its rotor.", source="parts.md"
            ),
            Document(text="Photosynthesis converts light into chemical energy.", source="bio.md"),
        ]
    )
    hits = await kb.retrieve("SKU-7537", k=1)
    expect(bool(hits), "expected a hybrid hit on the exact term")
    expect(
        "sku-7537" in hits[0].chunk.text.lower(),
        f"expected the parts doc, got {hits[0].chunk.text!r}",
    )
    return {"text": hits[0].chunk.text}


if __name__ == "__main__":
    print(asyncio.run(run()))
