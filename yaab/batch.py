"""Batch / offline inference — high-throughput runs over many inputs.

Agent loops are built for one interactive request; offline jobs (label a
dataset, embed a corpus, backfill answers) want many inputs run concurrently
with a bounded worker pool, partial-failure tolerance, and progress. This module
provides that over the same agents/embedders.

    from yaab.batch import batch_run, batch_embed
    results = await batch_run(agent, prompts, concurrency=16)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field


class BatchItem(BaseModel):
    """One result in a batch: the input, the output (or error), and timing."""

    index: int
    input: Any = None
    output: Any = None
    error: str | None = None
    ok: bool = True


class BatchResult(BaseModel):
    items: list[BatchItem] = Field(default_factory=list)

    @property
    def outputs(self) -> list[Any]:
        return [i.output for i in self.items]

    @property
    def succeeded(self) -> int:
        return sum(1 for i in self.items if i.ok)

    @property
    def failed(self) -> int:
        return sum(1 for i in self.items if not i.ok)


async def batch_map(
    fn: Callable[[Any], Awaitable[Any]],
    inputs: list[Any],
    *,
    concurrency: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Run ``fn`` over ``inputs`` with bounded concurrency, tolerating failures.

    Order is preserved; a failed item records its error and ``ok=False`` rather
    than aborting the batch. ``on_progress(done, total)`` is called as items
    complete.
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(inputs)
    done = 0
    items: list[BatchItem] = [BatchItem(index=i, input=x) for i, x in enumerate(inputs)]
    lock = asyncio.Lock()

    async def worker(item: BatchItem) -> None:
        nonlocal done
        async with sem:
            try:
                item.output = await fn(item.input)
            except Exception as exc:  # noqa: BLE001 - record, don't abort
                item.error = str(exc)
                item.ok = False
        async with lock:
            done += 1
            if on_progress is not None:
                on_progress(done, total)

    await asyncio.gather(*(worker(it) for it in items))
    return BatchResult(items=items)


async def batch_run(
    agent: Any,
    prompts: list[str],
    *,
    concurrency: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
    **run_kwargs: Any,
) -> BatchResult:
    """Run an agent over many prompts concurrently; outputs are the run outputs."""

    async def one(prompt: str) -> Any:
        result = await agent.run(prompt, **run_kwargs)
        return result.output

    return await batch_map(one, prompts, concurrency=concurrency, on_progress=on_progress)


async def batch_embed(
    embedder: Callable[[str], list[float]],
    texts: list[str],
    *,
    concurrency: int = 16,
) -> list[list[float]]:
    """Embed many texts concurrently (sync embedders run in a thread pool)."""

    async def one(text: str) -> list[float]:
        return await asyncio.to_thread(embedder, text)

    result = await batch_map(one, texts, concurrency=concurrency)
    return [item.output for item in result.items]


__all__ = ["batch_map", "batch_run", "batch_embed", "BatchResult", "BatchItem"]
