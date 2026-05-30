"""Performance-core shim.

Prefers the compiled Rust extension (:mod:`yaab_core`) and transparently falls
back to pure-Python implementations when the wheel is unavailable. The public
functions here are the *only* way the rest of the SDK touches the hot paths, so
the Rust/Python split is invisible to callers.

Inspect :data:`RUST` to know which backend is active.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

# Set YAAB_NO_RUST=1 to force the pure-Python fallback (used to keep the
# fallback path green in CI even when the wheel is installed).
if os.environ.get("YAAB_NO_RUST") == "1":
    _rust = None
    RUST = False
else:
    try:  # pragma: no cover - exercised by whichever backend is installed
        import yaab_core as _rust

        RUST = True
    except ImportError:  # pragma: no cover
        _rust = None
        RUST = False

__all__ = [
    "RUST",
    "backend",
    "cosine_similarity",
    "top_k",
    "encode_checkpoint",
    "decode_checkpoint",
    "reduce_channel",
    "plan_supersteps",
    "hash_event",
    "verify_chain",
    "aggregate_cost",
]

_MAGIC = b"YAAB"
_VERSION = 1


def backend() -> str:
    """Return ``"rust"`` or ``"python"`` for diagnostics and tests."""
    return "rust" if RUST else "python"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if RUST:
        return _rust.cosine_similarity(list(a), list(b))
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def top_k(query: list[float], matrix: list[list[float]], k: int) -> list[tuple[int, float]]:
    if RUST:
        return _rust.top_k(list(query), [list(r) for r in matrix], k)
    scored = [(i, cosine_similarity(query, row)) for i, row in enumerate(matrix)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def encode_checkpoint(state: Any) -> bytes:
    """Serialize a JSON-compatible state object into a framed checkpoint blob."""
    json_str = json.dumps(state, separators=(",", ":"), sort_keys=True)
    if RUST:
        return bytes(_rust.encode_checkpoint(json_str))
    return _MAGIC + bytes([_VERSION]) + json_str.encode("utf-8")


def decode_checkpoint(blob: bytes) -> Any:
    """Decode a framed checkpoint blob back into a Python object."""
    if RUST:
        return json.loads(_rust.decode_checkpoint(bytes(blob)))
    if len(blob) < 5 or blob[0:4] != _MAGIC:
        raise ValueError("invalid checkpoint: bad magic header")
    if blob[4] != _VERSION:
        raise ValueError(f"unsupported checkpoint version: {blob[4]}")
    return json.loads(blob[5:].decode("utf-8"))


def reduce_channel(reducer: str, current: Any, update: Any) -> Any:
    """Merge ``update`` into ``current`` under the named channel reducer."""
    if RUST:
        out = _rust.reduce_channel(
            reducer,
            json.dumps(current, separators=(",", ":")),
            json.dumps(update, separators=(",", ":")),
        )
        return json.loads(out)
    if reducer == "last_value":
        return update
    if reducer == "append":
        base = list(current) if isinstance(current, list) else ([] if current is None else [current])
        if isinstance(update, list):
            base.extend(update)
        else:
            base.append(update)
        return base
    if reducer == "add":
        a = current if isinstance(current, (int, float)) else 0
        b = update if isinstance(update, (int, float)) else 0
        return a + b
    return update


def plan_supersteps(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Group nodes into BSP supersteps; trailing cycle members form a final step."""
    if RUST:
        return _rust.plan_supersteps(list(nodes), [tuple(e) for e in edges])
    node_set = set(nodes)
    indegree = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src not in node_set or dst not in node_set or src == dst:
            continue
        adj[src].append(dst)
        indegree[dst] += 1
    layers: list[list[str]] = []
    ready = [n for n, d in indegree.items() if d == 0]
    visited: set[str] = set()
    while ready:
        layer = sorted(n for n in ready if n not in visited)
        visited.update(layer)
        nxt: list[str] = []
        for node in ready:
            for child in adj.get(node, []):
                indegree[child] -= 1
                if indegree[child] == 0 and child not in visited:
                    nxt.append(child)
        if layer:
            layers.append(layer)
        ready = nxt
    remaining = sorted(n for n in nodes if n not in visited)
    if remaining:
        layers.append(remaining)
    return layers


def hash_event(prev_hash: str, payload: str) -> str:
    """Chained audit hash: ``sha256(prev_hash || payload)``."""
    if RUST:
        return _rust.hash_event(prev_hash, payload)
    import hashlib

    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(b"\x1f")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def verify_chain(genesis: str, entries: list[tuple[str, str]]) -> int | None:
    """Return the index of the first broken hash-chain link, or ``None``."""
    if RUST:
        return _rust.verify_chain(genesis, [tuple(e) for e in entries])
    prev = genesis
    for i, (payload, recorded) in enumerate(entries):
        expected = hash_event(prev, payload)
        if expected != recorded:
            return i
        prev = expected
    return None


def aggregate_cost(values: list[float]) -> float:
    if RUST:
        return _rust.aggregate_cost([float(v) for v in values])
    return float(sum(values))
