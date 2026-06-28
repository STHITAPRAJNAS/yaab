"""Research assistant — a durable, multi-step research flow.

A ``Flow`` retrieves grounding from a **hybrid (BM25 + dense)** knowledge base,
then drafts a brief from it; the draft also **streams** token-by-token. The live
version adds a **browser** tool for sources without an API and records the real
model run into a **cassette** so the docs example replays deterministically (see
README).

    python -m cookbook.apps.research_assistant
"""

from __future__ import annotations

import asyncio
from typing import Any

from cookbook._harness import expect, resolve_model
from yaab import Agent, Flow, KnowledgeBase
from yaab.models.test_model import TestModel
from yaab.rag import Document

_CORPUS = [
    Document(text="The Antikythera mechanism is an ancient Greek analog computer.", source="kb1"),
    Document(text="It was used to predict astronomical positions and eclipses.", source="kb2"),
    Document(text="Photosynthesis converts light into chemical energy.", source="kb3"),
]


def _knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase(name="research", hybrid=True)  # BM25 + dense, fused by rank
    kb.add(_CORPUS)
    return kb


def build(model: Any = None) -> Flow:
    """A research flow: retrieve grounding (hybrid), then draft a brief."""
    kb = _knowledge_base()

    async def retrieve(state: Any, ctx: Any) -> dict:
        hits = await kb.retrieve("Antikythera mechanism", k=2)
        return {"notes": " ".join(h.chunk.text for h in hits)}

    drafter: Agent = Agent(
        "drafter",
        model=resolve_model(
            model,
            offline_default=TestModel(
                custom_output="The Antikythera mechanism is an ancient analog computer."
            ),
        ),
        instructions="Write a one-sentence brief grounded in: {notes}",
    )

    return (
        Flow[None, str]("research")
        .step("retrieve", fn=retrieve)
        .step("draft", agent=drafter, writes="brief")
        .start_at("retrieve")
        .then("retrieve", "draft")
        .then("draft", Flow.DONE)
        .returns("brief")
    )


async def run() -> dict:
    flow = build()
    result = await flow.run("Tell me about the Antikythera mechanism.")
    brief = str(result.output)
    expect("antikythera" in brief.lower(), f"the brief should be grounded, got {brief!r}")

    # The same drafting agent streams its answer token-by-token for a live UI.
    drafter: Agent = Agent(
        "narrator",
        model=TestModel(custom_output="an ancient analog computer"),
        instructions="Brief.",
    )
    chunks = [tok async for tok in drafter.stream("summarize")]
    expect(len(chunks) >= 1, "expected streamed deltas")

    return {"brief": brief, "streamed_chunks": len(chunks)}


if __name__ == "__main__":
    print(asyncio.run(run()))
