"""Recipe: long-term memory — recall facts across sessions, scoped per user.

``MemoryManager`` stores durable facts and retrieves the relevant ones by
similarity, scoped to ``(app_name, user_id)`` so one user never sees another's.
Uses the offline hashing embedder by default; swap in a real embedder for prod.

    python -m cookbook.recipes.memory
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import MemoryManager
from yaab.memory import InMemoryVectorMemory


async def run() -> dict:
    memory = MemoryManager(InMemoryVectorMemory())

    await memory.add("Alice prefers email over phone.", app_name="crm", user_id="alice")
    await memory.add("Alice's plan renews in March.", app_name="crm", user_id="alice")
    await memory.add("Bob prefers phone calls.", app_name="crm", user_id="bob")

    hits = await memory.search("how should we contact Alice?", app_name="crm", user_id="alice", k=2)
    recalled = [r.text for r, _score in hits]
    expect(
        any("email" in t.lower() for t in recalled), f"expected email preference, got {recalled}"
    )

    # Scoping: Bob's memories never surface for Alice.
    leaked = any("bob" in t.lower() for t in recalled)
    expect(not leaked, "memory must be isolated per user")
    return {"recalled": recalled, "isolated": not leaked}


if __name__ == "__main__":
    print(asyncio.run(run()))
