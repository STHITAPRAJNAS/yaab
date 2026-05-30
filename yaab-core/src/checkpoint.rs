//! Checkpoint (de)serialization for durable graph execution.
//!
//! State snapshots are framed with a small magic header + version byte so the
//! format can evolve without ambiguity. The payload is compact JSON; the Rust
//! path validates structure on the way in and out.

use serde_json::Value;

const MAGIC: &[u8] = b"YAAB";
const VERSION: u8 = 1;

/// Encode a JSON document (passed as a string) into a framed checkpoint blob.
pub fn encode(json_str: &str) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_str(json_str).map_err(|e| e.to_string())?;
    let payload = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(payload.len() + 5);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Decode a framed checkpoint blob back into a compact JSON string.
pub fn decode(blob: &[u8]) -> Result<String, String> {
    if blob.len() < 5 || &blob[0..4] != MAGIC {
        return Err("invalid checkpoint: bad magic header".to_string());
    }
    if blob[4] != VERSION {
        return Err(format!("unsupported checkpoint version: {}", blob[4]));
    }
    let value: Value = serde_json::from_slice(&blob[5..]).map_err(|e| e.to_string())?;
    serde_json::to_string(&value).map_err(|e| e.to_string())
}
