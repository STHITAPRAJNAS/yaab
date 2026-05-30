# RAG (retrieval-augmented generation)

YAAB ships RAG **built-in and provider-neutral** — not delegated to a managed
cloud service (as ADK→Vertex, OpenAI→hosted vector stores, and Strands→Bedrock
do) and not left entirely to you to assemble (as Pydantic AI does). The pipeline
mirrors the de-facto standard — `Document → Chunk → Embedder → VectorStore →
Retriever → Reranker` — wrapped in one `KnowledgeBase` object, and adds the
governance pieces the ecosystem still lacks: **per-user/document access control
at retrieval, source citations, embedding caching, incremental dedup indexing,
retrieval guardrails, and faithfulness evaluation.**

## Quickstart

```python
from yaab import Agent, Document, KnowledgeBase

kb = KnowledgeBase()
kb.add(Document(text="Paris is the capital of France.", source="geo.md"))

# Use it as an agent tool — the agent retrieves on demand:
agent = Agent("assistant", model="openai/gpt-4o", tools=[kb.as_tool()])
print(agent.run_sync("What is the capital of France?").output)
```

Or retrieve directly (classic context-stuffing), with citations:

```python
block, chunks = await kb.augment("capital of France?", k=3)
# block: "[geo.md#0] Paris is the capital of France."
```

## The pipeline (all swappable)

```python
from yaab.rag import KnowledgeBase, SentenceChunker, InMemoryVectorStore, KeywordReranker
from yaab.memory.embedders import LiteLLMEmbedder, CachingEmbedder

kb = KnowledgeBase(
    chunker=SentenceChunker(chunk_size=800),
    embedder=CachingEmbedder(LiteLLMEmbedder("openai/text-embedding-3-small")),
    store=InMemoryVectorStore(),
    reranker=KeywordReranker(weight=0.4),
)
```

Every component is a `typing.Protocol`, so Chroma/Qdrant/Pinecone stores or
cross-encoder rerankers drop in behind `VectorStore` / `Reranker`. Built-ins:

| Concern | Ships |
|---|---|
| Chunkers | `CharacterChunker`, `SentenceChunker`, `ParagraphChunker` |
| Embedders | `hashing_embedder` (offline), `LiteLLMEmbedder` (any provider), `CachingEmbedder` |
| Vector stores | in-memory · pgvector/Aurora · Chroma · Qdrant · OpenSearch · Oracle 23ai |
| Rerankers | `KeywordReranker` (lexical hybrid), `LLMReranker`, `CrossEncoderReranker` |

## Production vector stores

All stores satisfy one `VectorStore` protocol and honor metadata `where` filters
(per-tenant isolation pushes down to the DB/cluster). Pick by class or by name —
see the full matrix in [Storage & backends](storage-backends.md).

```python
from yaab.rag import KnowledgeBase, PgVectorStore           # yaab[postgres]

# Postgres / Amazon Aurora PostgreSQL with pgvector:
kb = KnowledgeBase(store=PgVectorStore("postgresql://…@aurora-endpoint/db", dim=1536))

# Amazon OpenSearch Service / Serverless:           yaab[opensearch]
from yaab.rag import OpenSearchVectorStore
kb = KnowledgeBase(store=OpenSearchVectorStore(index="kb", hosts=[{"host": "...", "port": 443}]))

# Oracle Database 23ai AI Vector Search:            yaab[oracle]
from yaab.rag import OracleVectorStore
kb = KnowledgeBase(store=OracleVectorStore(dsn="...", user="...", password="..."))

# Chroma (yaab[chroma]) and Qdrant (yaab[qdrant]) likewise.
```

## Governance features

### Per-user / document-level access control

Tag documents with metadata, then filter at retrieval — so an agent run for one
user never retrieves another's documents:

```python
kb.add(Document(text="Alice's note", source="a", metadata={"user": "alice"}))
results = await kb.retrieve("note", where={"user": "alice"})   # only alice's
```

Wire it to the agent automatically with `scope_from_deps`:

```python
tool = kb.as_tool(scope_from_deps="user")   # reads ctx.deps.user as the filter
```

### Source citations

Every `RetrievedChunk` carries a citation (`source#index`); `augment()` prepends
them so answers can attribute their context.

### Incremental, dedup indexing

Re-ingesting unchanged content is a cheap no-op; update a source in place:

```python
kb.add(docs)                                   # dedups by content hash
kb.reindex(new_docs, source="policy.md")       # replace one source's chunks
kb.delete(source="policy.md")                  # remove a source
```

### Retrieval guardrails

Filter weak or unsafe context *before* it reaches the model (context-poisoning /
leakage defense):

```python
kb = KnowledgeBase(
    min_score=0.2,                               # drop weak recall
    context_guard=lambda rc: "secret" not in rc.text.lower(),
)
```

### Faithfulness evaluation

Is the answer grounded in the retrieved context? RAGAS-style metrics, native:

```python
from yaab.rag import faithfulness, context_relevance, FaithfulnessEvaluator

faithfulness(answer, chunks)        # deterministic 0–1 groundedness proxy
context_relevance(query, chunks)    # deterministic 0–1 retrieval-recall proxy
await FaithfulnessEvaluator("openai/gpt-4o").ascore(answer, chunks)  # LLM judge
```

These plug into the [governance eval framework](governance.md#evaluation) and the
[drift monitor](governance.md#drift-detection--trust-scoring) for ongoing RAG
quality tracking.

### Embedding cache

`CachingEmbedder` wraps any embedder to avoid re-embedding identical text
(re-indexing, repeated queries) — a recurring RAG cost sink:

```python
emb = CachingEmbedder(LiteLLMEmbedder("openai/text-embedding-3-small"))
print(emb.hits, emb.misses)
```
