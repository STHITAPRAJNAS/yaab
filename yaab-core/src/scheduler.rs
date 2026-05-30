//! Superstep scheduler (Bulk Synchronous Parallel).
//!
//! Given a node set and directed edges, group nodes into supersteps: each
//! superstep is a set of nodes with no unresolved upstream dependency, so all
//! nodes in a superstep can execute in parallel. This mirrors the Pregel /
//! Apache Beam execution model LangGraph is built on.
//!
//! Cycles are expected (graphs support loops), so any nodes that remain after
//! the acyclic layering are emitted as a final superstep rather than treated
//! as an error — the runtime drives cycles via conditional edges at run time.

use std::collections::{HashMap, HashSet, VecDeque};

/// Plan the supersteps for a graph.
pub fn plan(nodes: &[String], edges: &[(String, String)]) -> Vec<Vec<String>> {
    let node_set: HashSet<&String> = nodes.iter().collect();

    let mut indegree: HashMap<&String, usize> = nodes.iter().map(|n| (n, 0usize)).collect();
    let mut adj: HashMap<&String, Vec<&String>> = HashMap::new();
    for (src, dst) in edges {
        if !node_set.contains(src) || !node_set.contains(dst) || src == dst {
            continue;
        }
        adj.entry(src).or_default().push(dst);
        *indegree.entry(dst).or_insert(0) += 1;
    }

    let mut layers: Vec<Vec<String>> = Vec::new();
    let mut ready: VecDeque<&String> = indegree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    let mut visited: HashSet<&String> = HashSet::new();

    while !ready.is_empty() {
        let mut layer: Vec<String> = Vec::new();
        let current: Vec<&String> = ready.drain(..).collect();
        for node in &current {
            if visited.contains(node) {
                continue;
            }
            visited.insert(node);
            layer.push((*node).clone());
        }
        layer.sort();
        let mut next: Vec<&String> = Vec::new();
        for node in &current {
            if let Some(children) = adj.get(node) {
                for child in children {
                    if let Some(d) = indegree.get_mut(*child) {
                        *d = d.saturating_sub(1);
                        if *d == 0 && !visited.contains(*child) {
                            next.push(child);
                        }
                    }
                }
            }
        }
        if !layer.is_empty() {
            layers.push(layer);
        }
        ready.extend(next);
    }

    // Any nodes left participate in a cycle; emit them as a trailing superstep.
    let mut remaining: Vec<String> = nodes
        .iter()
        .filter(|n| !visited.contains(n))
        .cloned()
        .collect();
    if !remaining.is_empty() {
        remaining.sort();
        layers.push(remaining);
    }
    layers
}
