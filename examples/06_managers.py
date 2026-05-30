"""Session / memory / artifact managers, ADK-style (offline)."""

import asyncio

from yaab import ArtifactManager, MemoryManager, SessionManager
from yaab.types import Role


async def main():
    # Sessions scoped by app + user.
    sessions = SessionManager()
    s = await sessions.create_session(app_name="bank", user_id="alice", state={"tier": "gold"})
    await sessions.append_text(s.id, Role.USER, "What's my balance?")
    print("sessions:", await sessions.list_sessions(app_name="bank", user_id="alice"))
    print("state:", await sessions.get_state(s.id))

    # Long-term memory with namespacing + retrieval.
    memory = MemoryManager()
    await memory.add("Alice prefers email contact", app_name="bank", user_id="alice")
    await memory.add("Alice lives in Paris", app_name="bank", user_id="alice")
    hits = await memory.search("how to reach Alice?", app_name="bank", user_id="alice", k=1)
    print("memory recall:", hits[0][0].text)

    # Versioned artifacts.
    artifacts = ArtifactManager()
    await artifacts.save("statement.txt", b"January statement", session_id=s.id)
    await artifacts.save("statement.txt", b"February statement", session_id=s.id)
    print("artifact versions:", await artifacts.list_versions("statement.txt", session_id=s.id))
    print("latest:", (await artifacts.load("statement.txt", session_id=s.id)).decode())


asyncio.run(main())
