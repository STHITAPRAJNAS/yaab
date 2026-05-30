# State: sessions, memory & artifacts

YAAB separates three kinds of state, each with a low-level **service** (the
storage protocol) and a high-level **manager** (scoped, ADK-style operations):

| Concern | What it holds | Service | Manager |
|---|---|---|---|
| Session | conversation history + structured KV state | `SessionService` | `SessionManager` |
| Memory | long-term, semantic (vector) recall | `MemoryService` | `MemoryManager` |
| Artifacts | binary/file blobs, versioned | `ArtifactService` | `ArtifactManager` |

Managers add `(app_name, user_id, session_id)` scoping; services are the
swappable backends.

## Sessions

```python
from yaab import SessionManager
from yaab.sessions import SQLiteSessionService

sessions = SessionManager(SQLiteSessionService("sessions.db"))

s = await sessions.create_session(app_name="bank", user_id="alice", state={"tier": "gold"})
await sessions.append_text(s.id, Role.USER, "Hello")
await sessions.update_state(s.id, last_seen="2026-01-01")
ids = await sessions.list_sessions(app_name="bank", user_id="alice")
```

Pass a `session_id` to `agent.run(...)` to make a conversation durable and
multi-turn — prior history is replayed automatically:

```python
await agent.run("My name is Alice.", session_id=s.id)
await agent.run("What's my name?", session_id=s.id)   # remembers
```

Backends: `InMemorySessionService` (default), `SQLiteSessionService`, and your
own (Postgres/Redis) implementing the `SessionService` protocol.

## Memory (long-term, vector)

```python
from yaab import MemoryManager
from yaab.memory import InMemoryVectorMemory
from yaab.memory.embedders import LiteLLMEmbedder

memory = MemoryManager(InMemoryVectorMemory(embedder=LiteLLMEmbedder("openai/text-embedding-3-small")))

await memory.add("Alice prefers email contact", app_name="bank", user_id="alice")
hits = await memory.search("how should we reach Alice?", app_name="bank", user_id="alice", k=3)
for record, score in hits:
    print(score, record.text)
```

Retrieval uses the Rust-accelerated cosine/top-k (`yaab._core`), with a
pure-Python fallback. The default embedder is a deterministic hashing stub for
offline use; swap in `LiteLLMEmbedder` (or any `Callable[[str], list[float]]`)
for production.

### Ingest a session into memory

```python
session = await sessions.get_session(app_name="bank", user_id="alice", session_id=s.id)
await memory.add_session_to_memory(session, app_name="bank", user_id="alice")
```

Attach a `MemoryService` to a `Runner` to fold retrieved memories into the system
prompt automatically:

```python
runner = Runner(memory_service=InMemoryVectorMemory())
```

## Artifacts (versioned blobs)

```python
from yaab import ArtifactManager

artifacts = ArtifactManager()
v1 = await artifacts.save("report.pdf", pdf_bytes, mime_type="application/pdf", session_id=s.id)
v2 = await artifacts.save("report.pdf", new_bytes, session_id=s.id)   # version 2
latest = await artifacts.load("report.pdf", session_id=s.id)
first  = await artifacts.load("report.pdf", version=1, session_id=s.id)
```

All three managers are backend-agnostic — the in-memory defaults are for dev;
production swaps the service while the manager API stays identical.
