"""Document Q&A: load files into a KnowledgeBase and answer with citations.

Pattern: classic RAG "answer from my documents." Shows ingestion from text/dir,
retrieval with source attribution, and the augment() context-stuffing flow.
"""

from __future__ import annotations

from typing import Any

from yaab import Agent, KnowledgeBase
from yaab.models.test_model import TestModel

from .._common import resolve_model

_DOCS = [
    ("policy.md", "Employees may work remotely up to 3 days per week."),
    ("pto.md", "Full-time employees accrue 20 days of paid time off per year."),
    ("security.md", "All laptops must use full-disk encryption and a screen lock."),
]


def build_kb() -> KnowledgeBase:
    kb = KnowledgeBase(name="handbook")
    for source, text in _DOCS:
        kb.add_text(text, source=source)
    return kb


def build(model: Any = None) -> tuple[Agent, KnowledgeBase]:
    kb = build_kb()
    agent = Agent(
        "handbook-qa",
        model=resolve_model(
            model,
            offline_default=TestModel(
                custom_output="You accrue 20 days of PTO per year.",
                call_tools=["search_handbook"],
            ),
        ),
        instructions="Answer strictly from the handbook. Cite the source.",
        tools=[kb.as_tool()],
    )
    return agent, kb


async def run(question: str = "How much PTO do I get?", model: Any = None) -> dict[str, Any]:
    agent, kb = build(model)
    # Show both the agent answer and the raw retrieved context with citations.
    block, chunks = await kb.augment(question, k=2)
    answer = (await agent.run(question)).output
    return {"answer": answer, "citations": [c.citation() for c in chunks], "context": block}


if __name__ == "__main__":
    import asyncio

    out = asyncio.run(run())
    print("answer:", out["answer"])
    print("citations:", out["citations"])
