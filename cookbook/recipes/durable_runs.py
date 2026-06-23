"""Recipe: durable runs — a run is a durable record, not a fleeting task.

With a ``RunStore`` a background run becomes a durable row you can poll, update,
and (with SQLite/Postgres) see from any replica. ``durable_backends()`` wires one
from a single URL for multi-pod deployments.

    python -m cookbook.recipes.durable_runs
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import durable_backends
from yaab.runs import RunRecord
from yaab.runs.memory import InMemoryRunStore


async def run() -> dict:
    store = InMemoryRunStore()
    # A background submission is a durable record (here with a fixed timestamp).
    await store.create(
        RunRecord(
            run_id="job-1",
            agent="summarizer",
            prompt="summarize Q3",
            background=True,
            created_at=1.0,
            updated_at=1.0,
        )
    )
    record = await store.get("job-1")
    expect(record is not None, "the run should be retrievable as a durable record")
    assert record is not None
    expect(record.status.value == "queued", f"a new run is queued, got {record.status.value}")

    # durable_backends() bundles every shared store (incl. spend) for multi-pod.
    backends = durable_backends()  # in-memory default; pass dsn=... for Postgres
    expect(backends.run_store is not None, "durable_backends wires a run store")
    expect("spend_store" in backends.serve_kwargs(), "and the spend store for the server")
    return {"run_id": record.run_id, "status": record.status.value}


if __name__ == "__main__":
    print(asyncio.run(run()))
