"""Built-in RAG: ingest documents, retrieve with citations, use as a tool.

Runs fully offline with the default hashing embedder + in-memory store. Swap in
LiteLLMEmbedder + PgVectorStore for production (see docs/rag.md).
"""

import asyncio

from yaab import Agent, Document, KnowledgeBase
from yaab.rag import KeywordReranker
from yaab.testing import TestModel


async def main():
    # 1) Build a knowledge base and ingest documents (chunk -> embed -> store).
    kb = KnowledgeBase(reranker=KeywordReranker(weight=0.4), name="handbook")
    kb.add(
        [
            Document(text="YAAB ships built-in, provider-neutral RAG.", source="readme.md"),
            Document(text="Expense reports are filed in the finance portal.", source="hr.md"),
            Document(text="The on-call rotation is published every Monday.", source="ops.md"),
        ]
    )
    print("chunks indexed:", kb.count())

    # 2) Retrieve with source citations.
    results = await kb.retrieve("How do I file an expense report?", k=1)
    for r in results:
        print(f"  [{r.citation()}] {r.text}  (score={r.score:.2f})")

    # 3) Incremental update: re-index one source without duplicating.
    kb.reindex(Document(text="Expense reports now use the new portal.", source="hr.md"),
               source="hr.md")
    print("after reindex:", kb.count())

    # 4) Give retrieval to an agent as a tool.
    agent = Agent(
        "assistant",
        model=TestModel(custom_output="File it in the finance portal.",
                        call_tools=["search_handbook"]),
        tools=[kb.as_tool()],
    )
    answer = await agent.run("Where do expense reports go?")
    print("agent:", answer.output)


asyncio.run(main())
