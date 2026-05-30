//! State channel reducers.
//!
//! Channels define how a node's write to a key is merged into graph state.
//! The semantics mirror LangGraph channels: last-value (overwrite),
//! append (accumulate into a list), and add (numeric aggregation).

use serde_json::Value;

/// Reduce `current` with `update` according to `reducer`.
///
/// Inputs and output are JSON strings so the Rust core stays agnostic to the
/// Python value types. Unknown reducers fall back to last-value semantics.
pub fn reduce(reducer: &str, current_json: &str, update_json: &str) -> Result<String, String> {
    let update: Value = serde_json::from_str(update_json).map_err(|e| e.to_string())?;
    let result = match reducer {
        "last_value" => update,
        "append" => {
            let current: Value = serde_json::from_str(current_json).map_err(|e| e.to_string())?;
            let mut list = match current {
                Value::Array(items) => items,
                Value::Null => Vec::new(),
                other => vec![other],
            };
            match update {
                Value::Array(items) => list.extend(items),
                other => list.push(other),
            }
            Value::Array(list)
        }
        "add" => {
            let current: Value = serde_json::from_str(current_json).map_err(|e| e.to_string())?;
            let a = current.as_f64().unwrap_or(0.0);
            let b = update.as_f64().unwrap_or(0.0);
            let sum = a + b;
            // Preserve integers where possible for cleaner round-tripping.
            if sum.fract() == 0.0 {
                Value::from(sum as i64)
            } else {
                Value::from(sum)
            }
        }
        _ => update,
    };
    serde_json::to_string(&result).map_err(|e| e.to_string())
}
