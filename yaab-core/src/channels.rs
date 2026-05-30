//! State channel reducers and whole-superstep state advancement.
//!
//! Channels define how a node's write to a key is merged into graph state.
//! The semantics mirror LangGraph channels: last-value (overwrite),
//! append (accumulate into a list), and add (numeric aggregation).
//!
//! `advance_superstep` applies an entire superstep's node updates against the
//! current state in a single call — the compute-bound part of graph
//! orchestration the Python engine can offload to Rust when the developer
//! opts into the native engine.

use serde_json::{Map, Value};
use std::collections::HashMap;

/// Reduce a single `current`/`update` pair at the `Value` level.
fn reduce_value(reducer: &str, current: &Value, update: Value) -> Value {
    match reducer {
        "append" => {
            let mut list = match current {
                Value::Array(items) => items.clone(),
                Value::Null => Vec::new(),
                other => vec![other.clone()],
            };
            match update {
                Value::Array(items) => list.extend(items),
                other => list.push(other),
            }
            Value::Array(list)
        }
        "add" => {
            let a = current.as_f64().unwrap_or(0.0);
            let b = update.as_f64().unwrap_or(0.0);
            let sum = a + b;
            if sum.fract() == 0.0 {
                Value::from(sum as i64)
            } else {
                Value::from(sum)
            }
        }
        // "last_value" and any unknown reducer overwrite.
        _ => update,
    }
}

/// Reduce `current` with `update` according to `reducer` (JSON-string API).
pub fn reduce(reducer: &str, current_json: &str, update_json: &str) -> Result<String, String> {
    let update: Value = serde_json::from_str(update_json).map_err(|e| e.to_string())?;
    let current: Value = serde_json::from_str(current_json).map_err(|e| e.to_string())?;
    let result = reduce_value(reducer, &current, update);
    serde_json::to_string(&result).map_err(|e| e.to_string())
}

/// Apply a whole superstep's node updates to `state` in one pass.
///
/// * `state_json` — the current graph state object (JSON object).
/// * `reducers_json` — a JSON object mapping channel key -> reducer name.
///   Keys absent from this map use last-value semantics.
/// * `updates_json` — a JSON array of update objects, one per node that ran in
///   the superstep, applied in order.
///
/// Returns the new state as a JSON object string. Doing the full fold in Rust
/// avoids one Python<->Rust round-trip per (key, node) pair.
pub fn advance_superstep(
    state_json: &str,
    reducers_json: &str,
    updates_json: &str,
) -> Result<String, String> {
    let mut state: Map<String, Value> =
        match serde_json::from_str(state_json).map_err(|e| e.to_string())? {
            Value::Object(m) => m,
            _ => return Err("state must be a JSON object".to_string()),
        };
    let reducers: HashMap<String, String> =
        serde_json::from_str(reducers_json).map_err(|e| e.to_string())?;
    let updates: Vec<Value> = serde_json::from_str(updates_json).map_err(|e| e.to_string())?;

    for update in updates {
        let obj = match update {
            Value::Object(m) => m,
            Value::Null => continue, // a node may return no update
            _ => return Err("each node update must be a JSON object or null".to_string()),
        };
        for (key, value) in obj {
            let reducer = reducers
                .get(&key)
                .map(|s| s.as_str())
                .unwrap_or("last_value");
            let current = state.get(&key).cloned().unwrap_or(Value::Null);
            state.insert(key, reduce_value(reducer, &current, value));
        }
    }

    serde_json::to_string(&Value::Object(state)).map_err(|e| e.to_string())
}
