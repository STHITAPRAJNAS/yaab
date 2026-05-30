//! `yaab_core` — the Rust performance core for the YAAB agent SDK.
//!
//! This crate exposes a small, stable surface of hot-path primitives to
//! Python via PyO3: vector similarity for memory retrieval, framed checkpoint
//! (de)serialization for durable graph execution, channel reducers for graph
//! state, BSP superstep planning for the orchestration engine, and
//! tamper-evident hash-chaining for the audit log.
//!
//! Every function here has a pure-Python fallback in `yaab._core`, so the SDK
//! installs and runs even when this extension is unavailable. The Rust path is
//! an accelerator, never a hard dependency.

use pyo3::prelude::*;

mod actors;
mod channels;
mod checkpoint;
mod scheduler;
mod vector;

/// Cosine similarity between two equal-length vectors.
#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    vector::cosine(&a, &b)
}

/// Indices and scores of the `k` rows most similar to `query`.
#[pyfunction]
fn top_k(query: Vec<f32>, matrix: Vec<Vec<f32>>, k: usize) -> Vec<(usize, f32)> {
    vector::top_k(&query, &matrix, k)
}

/// Encode a JSON document string into a framed checkpoint blob.
#[pyfunction]
fn encode_checkpoint(json_str: &str) -> PyResult<Vec<u8>> {
    checkpoint::encode(json_str).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Decode a framed checkpoint blob back into a JSON string.
#[pyfunction]
fn decode_checkpoint(blob: Vec<u8>) -> PyResult<String> {
    checkpoint::decode(&blob).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Reduce `current` with `update` under the named channel reducer.
#[pyfunction]
fn reduce_channel(reducer: &str, current_json: &str, update_json: &str) -> PyResult<String> {
    channels::reduce(reducer, current_json, update_json)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Apply a whole superstep's node updates to graph state in one call.
#[pyfunction]
fn advance_superstep(
    state_json: &str,
    reducers_json: &str,
    updates_json: &str,
) -> PyResult<String> {
    channels::advance_superstep(state_json, reducers_json, updates_json)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Plan BSP supersteps for the given nodes and edges.
#[pyfunction]
fn plan_supersteps(nodes: Vec<String>, edges: Vec<(String, String)>) -> Vec<Vec<String>> {
    scheduler::plan(&nodes, &edges)
}

/// Chained hash for an audit entry: `sha256(prev_hash || payload)`.
#[pyfunction]
fn hash_event(prev_hash: &str, payload: &str) -> String {
    actors::hash_event(prev_hash, payload)
}

/// Verify a hash chain; returns the index of the first broken link or `None`.
#[pyfunction]
fn verify_chain(genesis: &str, entries: Vec<(String, String)>) -> Option<usize> {
    actors::verify_chain(genesis, &entries)
}

/// Sum per-call costs into a single total.
#[pyfunction]
fn aggregate_cost(values: Vec<f64>) -> f64 {
    actors::aggregate_cost(&values)
}

#[pymodule]
fn yaab_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RUST", true)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(top_k, m)?)?;
    m.add_function(wrap_pyfunction!(encode_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(decode_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_channel, m)?)?;
    m.add_function(wrap_pyfunction!(advance_superstep, m)?)?;
    m.add_function(wrap_pyfunction!(plan_supersteps, m)?)?;
    m.add_function(wrap_pyfunction!(hash_event, m)?)?;
    m.add_function(wrap_pyfunction!(verify_chain, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_cost, m)?)?;
    Ok(())
}
