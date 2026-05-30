# Storage & backends

Every stateful concern in YAAB — **sessions, long-term memory, vector stores,
checkpoints, the registry, and the audit log** — is defined by a small
`typing.Protocol` and selected through the component registry. A
dependency-free **in-memory default** ships for each, and concrete backends for
the databases and cloud services teams actually run are included. Swapping a
backend is a one-line change; adding your own is a plug-in, not a fork.

## The pattern

```python
from yaab import Runner
from yaab.sessions import PostgresSessionService

# Default: in-memory. Production: pass a concrete backend — agent code is unchanged.
runner = Runner(session_service=PostgresSessionService("postgresql://…@aurora-endpoint/db"))
```

Or resolve by name from the registry (great for config-driven deploys):

```python
from yaab import get_component, available_components

available_components("session")        # ['memory', 'sqlite', 'postgres', 'aurora', 'redis']
available_components("vectorstore")    # ['memory','pgvector','aurora','chroma','qdrant','opensearch','oracle']

store = get_component("vectorstore", "opensearch", index="kb", hosts=[...])
```

## Sessions

Conversation history + structured KV state. Protocol: `SessionService`.

| Name | Class | Install | Use for |
|---|---|---|---|
| `memory` | `InMemorySessionService` | — (default) | dev, tests |
| `sqlite` | `SQLiteSessionService` | — | single-node, local persistence |
| `postgres` | `PostgresSessionService` | `yaab[postgres]` | Postgres |
| `aurora` | `PostgresSessionService` | `yaab[postgres]` | **Amazon Aurora PostgreSQL** (same driver; point the DSN at the cluster) |
| `redis` | `RedisSessionService` | `yaab[redis]` | **ElastiCache / MemoryDB / Azure Cache**, low-latency distributed sessions |

```python
from yaab.sessions import RedisSessionService
svc = RedisSessionService("rediss://my-elasticache:6379/0", ttl_seconds=86400)
```

## Long-term memory

Semantic, cross-session recall. Protocol: `MemoryService` (with a `MemoryManager`
on top for app/user scoping and session→memory ingestion). The default
`InMemoryVectorMemory` uses the Rust-accelerated top-k; for durable memory, back
it with any vector store below via a `KnowledgeBase`, or implement `MemoryService`
against your store of choice.

```python
from yaab import MemoryManager
from yaab.memory import InMemoryVectorMemory
from yaab.memory.embedders import LiteLLMEmbedder

memory = MemoryManager(InMemoryVectorMemory(embedder=LiteLLMEmbedder("openai/text-embedding-3-small")))
```

## Vector stores (RAG)

Embedded-chunk storage + similarity search with metadata filtering. Protocol:
`VectorStore`.

| Name | Class | Install | Notes |
|---|---|---|---|
| `memory` | `InMemoryVectorStore` | — (default) | dev/tests; Rust top-k |
| `pgvector` | `PgVectorStore` | `yaab[postgres]` | Postgres + pgvector |
| `aurora` | `PgVectorStore` | `yaab[postgres]` | **Aurora PostgreSQL** with pgvector |
| `chroma` | `ChromaVectorStore` | `yaab[chroma]` | local/embedded or server |
| `qdrant` | `QdrantVectorStore` | `yaab[qdrant]` | in-memory, server, or Qdrant Cloud |
| `opensearch` | `OpenSearchVectorStore` | `yaab[opensearch]` | **Amazon OpenSearch Service / Serverless**, self-managed |
| `oracle` | `OracleVectorStore` | `yaab[oracle]` | **Oracle Database 23ai** AI Vector Search |
| `pinecone` | `PineconeVectorStore` | `yaab[pinecone]` | Pinecone serverless/pod |
| `weaviate` | `WeaviateVectorStore` | `yaab[weaviate]` | Weaviate (local or Cloud) |

```python
from yaab.rag import KnowledgeBase
from yaab.rag import OpenSearchVectorStore        # yaab[opensearch]

kb = KnowledgeBase(store=OpenSearchVectorStore(index="kb", hosts=[{"host": "...", "port": 443}],
                                               http_auth=(...), use_ssl=True))
```

All stores honor metadata `where` filters, so [per-user/document access
control](rag.md#per-user--document-level-access-control) pushes down to the
database/cluster (JSONB containment on Postgres, `JSON_VALUE` on Oracle, term
filters on OpenSearch).

## Checkpoints (durable graphs)

Durable graph state for crash recovery, resume, and time-travel. Protocol:
`Checkpointer` (`put`/`get`/`history`).

| Name | Class | Install |
|---|---|---|
| `memory` | `MemorySaver` | — (default) |
| `sqlite` | `SQLiteSaver` | — |
| `postgres` / `aurora` | `PostgresSaver` | `yaab[postgres]` |
| `redis` | `RedisSaver` | `yaab[redis]` |

```python
from yaab.graph import StateGraph, PostgresSaver
app = graph.compile(checkpointer=PostgresSaver("postgresql://…@aurora-endpoint/db"))
```

See [Graph orchestration](graph.md#durable-execution--checkpoints).

## Audit sinks

Protocol: `AuditSink` (`write(event)`). In-memory + SQLite ship; forward to
Langfuse, Logfire, an OTel collector, or a callback via
[`yaab.observability.sinks`](platform.md#deeper-observability--eval).

## Implement your own

Any object matching the protocol works anywhere the protocol is accepted —
register it to make it selectable by name:

```python
from yaab.extensions import register
from yaab.rag.store import VectorStore   # the Protocol

class PineconeVectorStore:               # implement add/query/delete/count
    def add(self, chunks): ...
    def query(self, embedding, *, k=5, where=None): ...
    def delete(self, *, where=None): ...
    def count(self): ...

register("vectorstore", "pinecone", lambda **kw: PineconeVectorStore(**kw))
```

Or ship it as a package via an entry point (`yaab.vectorstores`,
`yaab.sessions`, `yaab.memory`, …) — see [Extending YAAB](extending.md). Import
the client library **lazily** inside `__init__` and raise a clear "install X"
error if missing, so your backend stays an optional dependency.
