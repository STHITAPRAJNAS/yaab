# Extending YAAB

YAAB is built to be extended. Every swappable concern is a `typing.Protocol`,
and a central **component registry** lets third-party developers add features
either in-process or as installable packages — without ever forking the core.

## Protocols (implement to swap)

| Protocol | Swap to change… | Module |
|---|---|---|
| `ModelProvider` | the LLM backend | `yaab.models.base` |
| `Tool` | how a capability is exposed | `yaab.tools.base` |
| `SessionService` | session storage (memory/SQLite/Postgres/Aurora/Redis/…) | `yaab.sessions.base` |
| `MemoryService` | long-term memory | `yaab.memory` |
| `VectorStore` | RAG vector storage (memory/pgvector/OpenSearch/Oracle/…) | `yaab.rag.store` |
| `Reranker` | retrieval reranking | `yaab.rag.rerank` |
| `Embedder` | embeddings | `yaab.memory.embedders` |
| `Chunker` | document splitting | `yaab.rag.chunking` |
| `ArtifactService` | blob storage | `yaab.artifacts` |
| `Checkpointer` | graph durability | `yaab.graph.checkpoint` |
| `GuardrailScanner` | a policy check | `yaab.governance.policy` |
| `ToolAuthorizer` | tool authorization | `yaab.governance.authorization` |
| `AuditSink` | where audit events go | `yaab.governance.audit` |
| `RegistryBackend` | registry storage | `yaab.governance.registry` |
| `ComplianceMapper` | a regulatory regime | `yaab.governance.compliance` |
| eval metric | a scoring metric (RAGAS/DeepEval/custom) | `yaab.eval` |
| `Optimizer` | a compile strategy | `yaab.optimize.optimizer` |
| `ContextStrategy` | context-window management | `yaab.context` |
| `Sandbox` | code-execution isolation | `yaab.tools.sandbox` |
| `Plugin` | cross-cutting hooks | `yaab.plugins` |
| `AuthScheme` | how requests authenticate | `yaab.auth` |

Anything matching the protocol works anywhere the protocol is accepted — no
registration required. Registering it (below) additionally makes it
**selectable by name** and discoverable.

## The component registry

For discoverable, name-addressable components, use `yaab.extensions`.

### Register in-process

```python
from yaab.extensions import register, get, available

@register("embedder", "myco")
def _make(**kwargs):
    return MyEmbedder(**kwargs)

available("embedder")              # [..., "myco", ...]
embedder = get("embedder", "myco", dim=256)
```

Component kinds include: `model`, `tool`, `session`, `memory`, `artifact`,
`checkpointer`, `guardrail`, `embedder`, `vectorstore`, `reranker`, `metric`,
`plugin`, `compliance`, `skill`.

### Worked example: add a vector-store backend

Implement the four-method `VectorStore` protocol, import the client lazily, and
register it — then it's selectable by name and drops into any `KnowledgeBase`:

```python
from yaab.extensions import register
from yaab.rag.types import Chunk, RetrievedChunk

class PineconeVectorStore:
    def __init__(self, *, index: str, **kw):
        try:
            from pinecone import Pinecone          # lazy: optional dependency
        except ImportError as exc:
            raise RuntimeError("pinecone is required. `pip install pinecone`.") from exc
        self._index = Pinecone(**kw).Index(index)

    def add(self, chunks: list[Chunk]) -> None: ...
    def query(self, embedding, *, k=5, where=None) -> list[RetrievedChunk]: ...
    def delete(self, *, where=None) -> int: ...
    def count(self) -> int: ...

register("vectorstore", "pinecone", lambda **kw: PineconeVectorStore(**kw))
```

```python
from yaab.rag import KnowledgeBase
from yaab import get_component
kb = KnowledgeBase(store=get_component("vectorstore", "pinecone", index="kb"))
```

The same recipe applies to a `SessionService` (kind `session`), `MemoryService`
(`memory`), `Reranker` (`reranker`), eval metric (`metric`), and so on. Honor
the protocol's metadata-filter semantics so per-tenant isolation keeps working.

### Register as an installable package (entry points)

Ship a package that advertises an entry point in the matching `yaab.<kind>s`
group; it is discovered lazily on first lookup. A broken plugin never breaks
`import yaab`.

```toml
# pyproject.toml
[project.entry-points."yaab.embedders"]
myco = "my_pkg.embedders:MyEmbedder"

[project.entry-points."yaab.compliance"]
my_regime = "my_pkg.compliance:MyRegimeMapper"

[project.entry-points."yaab.skills"]
research = "my_pkg.skills:research_skill"
```

## Plugins (cross-cutting hooks)

Plugins register on the `Runner` and fire on lifecycle callbacks that apply
across every agent the runner drives. A hook can **observe** (return `None`),
**intervene** (return a value to short-circuit), or **amend** (mutate the
context).

```python
from yaab.plugins import Plugin

class LoggingPlugin(Plugin):
    name = "logging"
    async def before_model(self, ctx, agent, messages):
        print(f"[{agent}] calling model with {len(messages)} messages")
        return None        # observe

runner = Runner(plugins=[LoggingPlugin()])
```

Built-ins (`yaab.plugins.builtins`): `AuditPlugin`, `CostBudgetPlugin`,
`CachingPlugin`. Hooks available: `before/after_run`, `on_user_message`,
`before/after_model`, `before/after_tool`.

## Compliance mappers (add a regime)

```python
from yaab.governance.compliance.base import ComplianceReport, ControlResult, ControlStatus

class MyRegimeMapper:
    regime = "my_regime"
    def map(self, registry, audit, agent_id=None) -> ComplianceReport:
        return ComplianceReport(regime=self.regime, agent_id=agent_id, controls=[
            ControlResult(id="X.1", title="...", status=ControlStatus.SATISFIED),
        ])
```

Register it under `yaab.compliance` and it appears in `yaab compliance report`
and `available_mappers()` — no core change.

## Cross-language

`yaab-core` is built as both `cdylib` and `rlib`, so the Rust engine can back
other language bindings (a TypeScript SDK is the planned next binding). Keep the
engine language-neutral; language SDKs stay thin.
