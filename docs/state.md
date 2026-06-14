# State: sessions, memory & artifacts

YAAB separates three kinds of state, each with a low-level **service** (the
storage protocol) and a high-level **manager** (scoped operations):

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
from yaab.types import Role

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

### Rewind & migrate

Roll a conversation back to an earlier point, or copy it to another backend. A
*turn* starts at a user message and runs until the next one; `rewind` keeps the
first N turns, `rewind_last` drops the most recent N. The structured `state` is
preserved — only the message history is truncated — and the result is persisted.

```python
session = await sessions.rewind(s.id, keep_turns=2)     # keep the first 2 turns
session = await sessions.rewind_last(s.id, turns=1)      # undo the last exchange
```

`migrate_session` copies a session (messages **and** state) into another
`SessionService` under the same id, leaving the source untouched — for moving a
conversation across stores or schema versions:

```python
from yaab.sessions import InMemorySessionService

await sessions.migrate_session(s.id, to_service=InMemorySessionService())
```

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
from yaab import Runner

runner = Runner(memory_service=InMemoryVectorMemory())
```

When the backend is namespace-aware (a `MemoryManager`), the Runner threads the
run's `identity` into the search as `user_id`, and a `memory_app_name` as
`app_name`, so **scoped** memory is reachable from the Agent path — and stays
isolated across users:

```python
memory = MemoryManager()
await memory.add("Alice's deadline is March 15.", app_name="bank", user_id="alice")

runner = Runner(memory_service=memory, memory_app_name="bank")
await runner.run(agent, "When is my deadline?", identity="alice")  # recalls it
await runner.run(agent, "When is my deadline?", identity="bob")    # sees nothing
```

Without `identity`/`memory_app_name` the search targets the `default` namespace.

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
