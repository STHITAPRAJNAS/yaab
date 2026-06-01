"""Document loaders, structured streaming, and batch inference (offline)."""

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from yaab import Agent, KnowledgeBase, batch_run
from yaab.rag import load_directory
from yaab.testing import TestModel


class Weather(BaseModel):
    city: str
    temp_c: int


async def main() -> dict:
    """Run the loader, structured-streaming, and batch patterns."""
    # 1) Document loaders — point YAAB at files instead of pre-extracting text.
    with tempfile.TemporaryDirectory() as d:
        Path(d, "faq.md").write_text("# FAQ\n\nRefunds are processed in 5 business days.")
        Path(d, "people.csv").write_text("name,role\nAlice,CEO\nBob,CTO")
        docs = load_directory(d, glob="**/*")
        print("loaded docs:", len(docs), "from", {dd.metadata["format"] for dd in docs})

        kb = KnowledgeBase()
        kb.add(docs)
        # (Default embedder is a lexical hash; use a real embedder in production.)
        hits = await kb.retrieve("Refunds processed business days", k=1)
        retrieved = hits[0].text if hits else ""
        print("retrieved:", retrieved or "(none)")

    # 2) Structured-output streaming — partial typed objects as they generate.
    agent = Agent("w", model=TestModel('{"city": "Paris", "temp_c": 21}'), output_type=Weather)
    partials = []
    print("streaming partials:")
    async for partial in agent.stream_structured("weather in Paris?", output_type=Weather):
        partials.append(partial)
        print("  ->", partial)

    # 3) Batch / offline inference — many prompts, bounded concurrency.
    batch_agent = Agent("b", model=TestModel("processed"))
    result = await batch_run(batch_agent, [f"item {i}" for i in range(5)], concurrency=3)
    print(f"batch: {result.succeeded} ok, {result.failed} failed -> {result.outputs}")

    return {
        "loaded": len(docs),
        "retrieved": retrieved,
        "partials": partials,
        "batch_succeeded": result.succeeded,
        "batch_failed": result.failed,
    }


if __name__ == "__main__":
    asyncio.run(main())
