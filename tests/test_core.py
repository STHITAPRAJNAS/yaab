"""Tests for the performance core (Rust or pure-Python fallback)."""

from __future__ import annotations

from yaab import _core


def test_backend_reports():
    assert _core.backend() in ("rust", "python")


def test_cosine_similarity():
    assert _core.cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
    assert abs(_core.cosine_similarity([1, 0], [0, 1])) < 1e-6
    assert _core.cosine_similarity([], []) == 0.0


def test_top_k():
    matrix = [[1, 0], [0, 1], [0.9, 0.1]]
    hits = _core.top_k([1, 0], matrix, 2)
    assert len(hits) == 2
    assert hits[0][0] == 0  # exact match ranks first


def test_checkpoint_roundtrip():
    state = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    blob = _core.encode_checkpoint(state)
    assert isinstance(blob, (bytes, bytearray))
    assert _core.decode_checkpoint(blob) == state


def test_reduce_channel():
    assert _core.reduce_channel("last_value", 1, 2) == 2
    assert _core.reduce_channel("append", [1], 2) == [1, 2]
    assert _core.reduce_channel("append", [1], [2, 3]) == [1, 2, 3]
    assert _core.reduce_channel("add", 2, 3) == 5


def test_plan_supersteps_linear():
    layers = _core.plan_supersteps(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert layers == [["a"], ["b"], ["c"]]


def test_plan_supersteps_parallel():
    layers = _core.plan_supersteps(["a", "b", "c"], [("a", "b"), ("a", "c")])
    assert layers[0] == ["a"]
    assert sorted(layers[1]) == ["b", "c"]


def test_hash_chain_and_verify():
    g = "0" * 64
    h1 = _core.hash_event(g, "one")
    h2 = _core.hash_event(h1, "two")
    assert _core.verify_chain(g, [("one", h1), ("two", h2)]) is None
    # Tamper with the second payload -> chain breaks at index 1.
    assert _core.verify_chain(g, [("one", h1), ("TWO", h2)]) == 1


def test_aggregate_cost():
    assert abs(_core.aggregate_cost([0.1, 0.2, 0.3]) - 0.6) < 1e-9
