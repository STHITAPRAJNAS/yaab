# Platform extensions

Loaders, more stores, a real sandbox, structured streaming, batch inference, a
dev UI, and deeper observability — the pieces that turn the building blocks into
a complete platform.

## Document loaders

Point YAAB at files instead of pre-extracting text:

```python
from yaab.rag import load, load_directory, KnowledgeBase

docs = load("handbook.pdf") + load_directory("./knowledge", glob="**/*.md")
KnowledgeBase().add(docs)
```

| Format | Loader | Notes |
|---|---|---|
| `.txt` / `.md` | `load_text` / `load_markdown` | dependency-free |
| `.html` / `.htm` | `load_html` | BeautifulSoup if present, else regex strip |
| `.pdf` | `load_pdf` | one Document per page; needs `pypdf` |
| `.csv` | `load_csv` | one Document per row |
| `.json` | `load_json` | one Document per array element |

`load(path)` dispatches on extension; `load_directory(dir, glob=...)` walks a
tree and skips unknown types; `load_bytes(data, source=..., fmt=...)` handles
uploads. Install parsers as needed: `pip install 'yaab[rag]'` (pypdf +
beautifulsoup4).

## More vector stores & a cross-encoder reranker

Beyond in-memory and pgvector — Chroma and Qdrant adapters (same `VectorStore`
protocol), plus a cross-encoder reranker (the precision standard):

```python
from yaab.rag import KnowledgeBase, CrossEncoderReranker
from yaab.rag import ChromaVectorStore        # pip install chromadb
# from yaab.rag import QdrantVectorStore       # pip install qdrant-client

kb = KnowledgeBase(
    store=ChromaVectorStore(path="./chroma"),
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
```

All stores are registered components: `yaab.get_component("vectorstore", "qdrant", url=...)`.

## A real sandbox for `python_exec`

The default `python_exec` is subprocess-isolated (crash/hang protection, **not**
security). For untrusted code, switch to the Docker sandbox — no network,
read-only root, dropped capabilities, CPU/memory/time caps:

```python
from yaab.tools.sandbox import DockerSandbox, set_default_sandbox

set_default_sandbox(DockerSandbox(image="python:3.11-slim", memory="256m", network=False))
```

Combine with [tool approval](robustness.md#human-in-the-loop-tool-approval-fast-path)
for defense in depth.

## Structured-output streaming

Render partial typed objects as the model generates them:

```python
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    findings: list[str]

async for partial in agent.stream_structured("write a report", output_type=Report):
    render(partial)     # successive partials; the final yield is fully validated
```

A tolerant JSON parser closes incomplete fragments so each yield is the
best-effort object so far.

## Batch / offline inference

High-throughput jobs over many inputs with bounded concurrency and
partial-failure tolerance:

```python
from yaab import batch_run, batch_embed
from yaab.batch import batch_map

result = await batch_run(agent, prompts, concurrency=16, on_progress=print)
print(result.succeeded, result.failed, result.outputs)

vectors = await batch_embed(embedder, texts, concurrency=32)
out = await batch_map(my_async_fn, items, concurrency=8)   # generic
```

Failures are recorded per item (`ok=False`, `error=...`) instead of aborting the
batch; order is preserved.

## `yaab web` — local dev playground

A zero-build browser playground that streams your agent's responses:

```bash
yaab web mymodule:agent          # http://127.0.0.1:8080
```

```python
from yaab.web import web_app
app = web_app(agent)             # mount in any ASGI server
```

It serves a self-contained HTML page over the agent's `/chat/stream` SSE
endpoint, alongside the full agent API (`/run`, `/a2a/tasks`, the agent card).

## Deeper observability & eval

Forward the audit log to popular backends:

```python
from yaab.governance import AuditLog
from yaab.observability.sinks import LangfuseSink, LogfireSink, OTelSpanSink, CallbackSink

audit = AuditLog(sinks=[
    LangfuseSink(),                       # pip install langfuse
    LogfireSink(),                        # pip install logfire
    OTelSpanSink(),                       # emits a span per event
    CallbackSink(lambda e: print(e.kind)) # custom
])
```

More eval metrics for CI and the [drift monitor](governance.md#drift-detection--trust-scoring):

```python
from yaab.governance import Regex, JSONMatch, NumericTolerance, Levenshtein, LLMJudge

# deterministic metrics (Evaluator protocol):
Regex(); JSONMatch(); NumericTolerance(tol=0.01); Levenshtein()
# LLM-judge (async):
await LLMJudge("openai/gpt-4o", criteria="factually correct").ascore(case, output)
```
