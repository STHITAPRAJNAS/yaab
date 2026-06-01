"""Session, memory, and artifact managers with app/user scoping (offline)."""

import asyncio

from yaab import ArtifactManager, MemoryManager, SessionManager
from yaab.types import Role


async def main() -> dict:
    """Exercise the three managers and return what each one stored."""
    # Sessions scoped by app + user.
    sessions = SessionManager()
    s = await sessions.create_session(app_name="bank", user_id="alice", state={"tier": "gold"})
    await sessions.append_text(s.id, Role.USER, "What's my balance?")
    listed = await sessions.list_sessions(app_name="bank", user_id="alice")
    state = await sessions.get_state(s.id)
    print("sessions:", listed)
    print("state:", state)

    # Long-term memory with namespacing + retrieval.
    memory = MemoryManager()
    await memory.add("Alice prefers email contact", app_name="bank", user_id="alice")
    await memory.add("Alice lives in Paris", app_name="bank", user_id="alice")
    hits = await memory.search("how to reach Alice?", app_name="bank", user_id="alice", k=1)
    recall = hits[0][0].text
    print("memory recall:", recall)

    # Versioned artifacts.
    artifacts = ArtifactManager()
    await artifacts.save("statement.txt", b"January statement", session_id=s.id)
    await artifacts.save("statement.txt", b"February statement", session_id=s.id)
    versions = await artifacts.list_versions("statement.txt", session_id=s.id)
    latest = (await artifacts.load("statement.txt", session_id=s.id)).decode()
    print("artifact versions:", versions)
    print("latest:", latest)

    return {
        "sessions": listed,
        "state": state,
        "recall": recall,
        "versions": versions,
        "latest": latest,
    }


if __name__ == "__main__":
    asyncio.run(main())
