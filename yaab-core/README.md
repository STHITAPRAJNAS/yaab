# yaab-core

The Rust performance core for [YAAB](../README.md), exposed to Python via PyO3.

It provides the hot-path primitives the Python layer calls into:

- **vector** — cosine similarity + top-k for memory retrieval
- **checkpoint** — framed (de)serialization of graph state
- **channels** — state-channel reducers (last-value / append / add)
- **scheduler** — BSP superstep planning for the orchestration engine
- **actors** — tamper-evident hash-chaining + cost aggregation for the audit log

Every function has a pure-Python fallback in `yaab._core`, so the SDK installs
and runs without this extension. Build it for the accelerated path:

```bash
maturin develop -m yaab-core/Cargo.toml --release
```
