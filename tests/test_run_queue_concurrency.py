"""Bounded concurrency — a thousand submissions never spawn a thousand tasks.

The worker drains the durable queue with a hard ceiling on how many runs execute
at once, so enqueueing far more work than the ceiling builds queue depth (the
natural backpressure signal) instead of unbounded in-flight tasks. These tests
gate a fake tool on an ``asyncio.Event`` to freeze every claimed run mid-flight,
then assert the number of simultaneously executing runs never exceeds the cap
and that the worker's live task set stays bounded.

All offline: TestModel agents, an in-memory run store, no sleeps for correctness.
"""

from __future__ import annotations

import asyncio
import time

from yaab import Agent, tool
from yaab.models.base import ModelResponse
from yaab.runs.base import RunRecord, RunStatus
from yaab.runs.memory import InMemoryRunStore
from yaab.runs.worker import RunWorker
from yaab.types import Role, ToolCall, Usage


def _record(run_id: str) -> RunRecord:
    now = time.time()
    return RunRecord(
        run_id=run_id,
        agent="svc",
        status=RunStatus.QUEUED,
        prompt="hi",
        background=True,
        resume_id=run_id,
        created_at=now,
        updated_at=now,
    )


class _AlwaysGateModel:
    """A model that calls the gated tool on every run's first turn, then finishes.

    Unlike a scripted model with shared once-only state, this inspects the
    conversation per call: if the gated tool has not yet produced a result it
    requests the tool (so every run parks on the gate); once the tool result is
    present it returns the final answer. This makes one shared agent saturate the
    worker's slots with many concurrently-parked runs.
    """

    name = "always-gate"

    async def complete(self, messages, *, tools=None, output_schema=None, tool_choice=None, **kw):
        usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        if any(m.role is Role.TOOL for m in messages):
            return ModelResponse(content="done", usage=usage, model="always-gate")
        return ModelResponse(
            tool_calls=[ToolCall(name="gated", arguments={})],
            finish_reason="tool_calls",
            usage=usage,
            model="always-gate",
        )

    async def stream(self, messages, *, tools=None, **kw):  # pragma: no cover - unused
        resp = await self.complete(messages, tools=tools, **kw)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                from yaab.models.base import StreamChunk

                yield StreamChunk(tool_call=tc)
        else:
            from yaab.models.base import StreamChunk

            yield StreamChunk(delta=resp.content)
        from yaab.models.base import StreamChunk

        yield StreamChunk(done=True)


def _gated_agent(gate: asyncio.Event, live: dict[str, int]) -> Agent:
    """An agent whose single tool parks on a shared gate while counting itself.

    ``live["now"]`` is the count of runs currently inside the gated tool;
    ``live["peak"]`` is the high-water mark — the maximum ever in flight at once.
    """

    @tool
    async def gated() -> str:
        """Block on the shared gate so concurrency can be observed."""
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        try:
            await gate.wait()
        finally:
            live["now"] -= 1
        return "released"

    return Agent("svc", model=_AlwaysGateModel(), tools=[gated])


def test_concurrency_cap_never_exceeded():
    """Enqueue 50 runs, cap 5: never more than 5 execute simultaneously."""
    gate = asyncio.Event()
    live = {"now": 0, "peak": 0}
    store = InMemoryRunStore()
    worker = RunWorker(_gated_agent(gate, live), store, max_concurrency=5, poll_interval=0.001)

    async def go() -> None:
        for i in range(50):
            await store.create(_record(f"r{i}"))

        loop_task = asyncio.create_task(worker.run_forever())

        # Let the worker saturate its slots: wait until 5 runs are parked on the
        # gate (or the in-flight count stabilizes at the cap).
        for _ in range(2000):
            await asyncio.sleep(0)
            if live["now"] >= 5:
                break
        # Give the loop extra turns to (try to) over-claim past the cap.
        for _ in range(200):
            await asyncio.sleep(0)
        assert live["now"] == 5, f"expected exactly the cap in flight, got {live['now']}"
        assert worker.in_flight <= 5

        # Release everything and let the queue drain fully.
        gate.set()
        for _ in range(5000):
            await asyncio.sleep(0)
            done = await store.list(status=RunStatus.COMPLETED, limit=100)
            if len(done) == 50:
                break
        worker.stop()
        await asyncio.wait_for(loop_task, timeout=5.0)

        done = await store.list(status=RunStatus.COMPLETED, limit=100)
        assert len(done) == 50
        # The high-water mark across the whole run never breached the cap.
        assert live["peak"] <= 5

    asyncio.run(go())


def test_live_task_count_bounded():
    """The worker's tracked task set never exceeds the cap while saturated."""
    gate = asyncio.Event()
    live = {"now": 0, "peak": 0}
    store = InMemoryRunStore()
    worker = RunWorker(_gated_agent(gate, live), store, max_concurrency=3, poll_interval=0.001)

    async def go() -> None:
        for i in range(20):
            await store.create(_record(f"r{i}"))

        loop_task = asyncio.create_task(worker.run_forever())
        peak_tasks = 0
        for _ in range(2000):
            await asyncio.sleep(0)
            peak_tasks = max(peak_tasks, worker.in_flight)
            if live["now"] >= 3:
                # Keep sampling a bit past saturation to catch any over-claim.
                for _ in range(100):
                    await asyncio.sleep(0)
                    peak_tasks = max(peak_tasks, worker.in_flight)
                break
        assert peak_tasks <= 3, f"task set grew past the cap: {peak_tasks}"
        # 17 runs are still queued (only the cap is in flight) — backpressure.
        queued = await store.list(status=RunStatus.QUEUED, limit=100)
        assert len(queued) == 20 - 3

        gate.set()
        for _ in range(5000):
            await asyncio.sleep(0)
            done = await store.list(status=RunStatus.COMPLETED, limit=100)
            if len(done) == 20:
                break
        worker.stop()
        await asyncio.wait_for(loop_task, timeout=5.0)
        assert len(await store.list(status=RunStatus.COMPLETED, limit=100)) == 20

    asyncio.run(go())
