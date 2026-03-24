use instant_distance::{Builder, HnswMap, Search};
use std::collections::HashSet;

use crate::types::NodeId;

// ─── Point implementation ───────────────────────────────

/// Wrapper around a f32 vector that implements `instant_distance::Point`
/// using cosine distance (1 - cosine_similarity).
#[derive(Clone, Debug)]
pub struct EmbeddingPoint(pub Vec<f32>);

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        cosine_distance(&self.0, &other.0)
    }
}

/// Cosine distance = 1 - cosine_similarity. Range [0, 2].
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot / denom)
}

/// Cosine similarity in [0, 1] (clamped).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    (1.0 - cosine_distance(a, b)).max(0.0)
}

// ─── Two-tier vector index ──────────────────────────────

/// The primary HNSW index is immutable (built once from SQLite on open, or on
/// periodic rebuild). New nodes added mid-session go into a linear buffer that
/// is brute-force scanned alongside the HNSW query. When the buffer exceeds
/// `rebuild_threshold`, a full rebuild is triggered.
pub struct VectorIndex {
    /// Frozen HNSW graph. `None` when the graph is empty.
    map: Option<HnswMap<EmbeddingPoint, NodeId>>,
    /// Number of items in the HNSW map (for len calculation).
    map_len: usize,
    /// Recent nodes not yet in the HNSW — searched via linear scan.
    buffer: Vec<(NodeId, Vec<f32>)>,
    /// IDs deleted at runtime — filtered out of search results.
    deleted: HashSet<NodeId>,
}

impl VectorIndex {
    /// Build from a complete set of embeddings (typically loaded from SQLite).
    pub fn build(items: Vec<(NodeId, Vec<f32>)>) -> Self {
        if items.is_empty() {
            return Self {
                map: None,
                map_len: 0,
                buffer: vec![],
                deleted: HashSet::new(),
            };
        }

        let len = items.len();
        let points: Vec<EmbeddingPoint> =
            items.iter().map(|(_, v)| EmbeddingPoint(v.clone())).collect();
        let values: Vec<NodeId> = items.iter().map(|(id, _)| id.clone()).collect();

        let map = Builder::default().build(points, values);

        Self {
            map: Some(map),
            map_len: len,
            buffer: vec![],
            deleted: HashSet::new(),
        }
    }

    /// Create an empty index.
    pub fn empty() -> Self {
        Self {
            map: None,
            map_len: 0,
            buffer: vec![],
            deleted: HashSet::new(),
        }
    }

    /// Add a new node to the linear buffer (not yet in HNSW).
    pub fn insert(&mut self, id: NodeId, embedding: Vec<f32>) {
        // If previously deleted, un-delete
        self.deleted.remove(&id);
        // Remove any previous buffer entry for this ID
        self.buffer.retain(|(eid, _)| eid != &id);
        self.buffer.push((id, embedding));
    }

    /// Mark a node as deleted. It will be filtered out of search results.
    /// Buffer entries are removed immediately; HNSW entries are skipped.
    pub fn remove(&mut self, id: &str) {
        self.buffer.retain(|(eid, _)| eid != id);
        self.deleted.insert(id.to_string());
    }

    /// How many items are waiting in the linear buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Total number of indexed items (HNSW + buffer).
    pub fn len(&self) -> usize {
        self.map_len + self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Search for the `k` most similar nodes. Queries the HNSW index and
    /// linearly scans the buffer, merging results.
    /// Returns `(NodeId, cosine_similarity)` pairs sorted descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        let mut results: Vec<(NodeId, f32)> = Vec::new();
        let mut seen = HashSet::new();

        // 1. HNSW search
        if let Some(ref map) = self.map {
            let q = EmbeddingPoint(query.to_vec());
            let mut search = Search::default();
            // Over-fetch to compensate for deleted entries
            for item in map.search(&q, &mut search).take(k + self.deleted.len()) {
                let node_id: &NodeId = item.value;
                if self.deleted.contains(node_id) {
                    continue;
                }
                let sim = 1.0 - item.distance;
                if seen.insert(node_id.clone()) {
                    results.push((node_id.clone(), sim));
                }
            }
        }

        // 2. Linear scan of buffer
        for (id, emb) in &self.buffer {
            if seen.contains(id) || self.deleted.contains(id) {
                continue;
            }
            let sim = cosine_similarity(query, emb);
            seen.insert(id.clone());
            results.push((id.clone(), sim));
        }

        // Sort descending by similarity, take top-k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Full rebuild from all embeddings. Clears the linear buffer.
    /// Call this when `buffer_len()` exceeds the configured threshold.
    pub fn rebuild(&mut self, all_items: Vec<(NodeId, Vec<f32>)>) {
        *self = Self::build(all_items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_empty_index_search() {
        let idx = VectorIndex::empty();
        let results = idx.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_buffer_only_search() {
        let mut idx = VectorIndex::empty();
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]);
        idx.insert("c".into(), vec![0.7, 0.7, 0.0]);

        let results = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a"); // exact match
    }

    #[test]
    fn test_build_and_search() {
        let items = vec![
            ("n1".into(), vec![1.0, 0.0, 0.0]),
            ("n2".into(), vec![0.0, 1.0, 0.0]),
            ("n3".into(), vec![0.0, 0.0, 1.0]),
            ("n4".into(), vec![0.9, 0.1, 0.0]),
        ];
        let idx = VectorIndex::build(items);

        let results = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // n1 and n4 should be the closest to [1,0,0]
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"n1"));
        assert!(ids.contains(&"n4"));
    }

    #[test]
    fn test_hybrid_search_hnsw_plus_buffer() {
        let items = vec![
            ("n1".into(), vec![1.0, 0.0, 0.0]),
            ("n2".into(), vec![0.0, 1.0, 0.0]),
        ];
        let mut idx = VectorIndex::build(items);
        // Add to buffer after build
        idx.insert("n3".into(), vec![0.95, 0.05, 0.0]);

        let results = idx.search(&[1.0, 0.0, 0.0], 3);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"n1"));
        assert!(ids.contains(&"n3")); // from buffer
    }
}
