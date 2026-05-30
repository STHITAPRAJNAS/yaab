//! Tamper-evident audit hashing and cost aggregation.
//!
//! The audit log is hash-chained: each entry's hash folds in the previous
//! entry's hash plus the current payload, so any retroactive edit breaks the
//! chain. Hashing the chain in Rust keeps the audit hot-path cheap.

use sha2::{Digest, Sha256};

/// Compute the chained hash for an audit entry: `sha256(prev_hash || payload)`.
pub fn hash_event(prev_hash: &str, payload: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.as_bytes());
    hasher.update(b"\x1f"); // unit separator, avoids boundary ambiguity
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

/// Verify a chain of `(payload, recorded_hash)` entries against a genesis hash.
/// Returns the index of the first broken link, or `None` if the chain is intact.
pub fn verify_chain(genesis: &str, entries: &[(String, String)]) -> Option<usize> {
    let mut prev = genesis.to_string();
    for (i, (payload, recorded)) in entries.iter().enumerate() {
        let expected = hash_event(&prev, payload);
        if &expected != recorded {
            return Some(i);
        }
        prev = expected;
    }
    None
}

/// Sum a set of per-call costs into a single total. Trivial today, but a
/// stable seam for moving cost aggregation fully into Rust later.
pub fn aggregate_cost(values: &[f64]) -> f64 {
    values.iter().sum()
}
