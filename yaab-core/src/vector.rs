//! Vector similarity operations used by `MemoryService` retrieval.
//!
//! These run outside the GIL on the Python side and operate on plain
//! `f32` slices, which keeps memory-retrieval cheap even for large stores.

/// Cosine similarity between two equal-length vectors.
///
/// Returns 0.0 for mismatched lengths or zero-magnitude vectors so callers
/// never have to special-case degenerate input.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Return the indices and scores of the `k` rows most similar to `query`,
/// sorted by descending cosine similarity.
pub fn top_k(query: &[f32], matrix: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| (i, cosine(query, row)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
