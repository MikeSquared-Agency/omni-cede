/// Runtime configuration for a CortexEmbedded instance.
#[derive(Debug, Clone)]
pub struct Config {
    /// Message count threshold before context compaction triggers.
    pub compaction_threshold: usize,
    /// Maximum agent loop iterations before forced stop.
    pub max_iterations: usize,
    /// Seconds between background decay sweeps.
    pub decay_interval_secs: u64,
    /// Minimum cosine similarity to create a `RelatesTo` auto-link edge.
    pub auto_link_cosine_threshold: f64,
    /// Minimum cosine similarity (with negation pattern) to flag as contradiction.
    pub contradiction_cosine_threshold: f64,
    /// When the HNSW linear buffer exceeds this count, trigger a full rebuild.
    pub hnsw_rebuild_threshold: usize,
    /// Maximum entries in the embedding LRU cache.
    pub embedding_cache_size: usize,
    /// Default number of nearest-neighbours returned by recall().
    pub default_recall_top_k: usize,
    /// Default BFS depth for graph walk during recall().
    pub default_graph_depth: usize,
    /// Lambda for recency weighting: exp(-lambda * hours_since_access).
    pub decay_lambda: f64,
    /// Number of HNSW neighbours to fetch for auto-link analysis.
    pub auto_link_candidates: usize,
    /// Number of most-recent session nodes (UserInput + Fact) always included
    /// in a chat turn's briefing, regardless of semantic similarity.
    pub session_recency_window: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            compaction_threshold: 20,
            max_iterations: 10,
            decay_interval_secs: 60,
            auto_link_cosine_threshold: 0.75,
            contradiction_cosine_threshold: 0.85,
            hnsw_rebuild_threshold: 200,
            embedding_cache_size: 10_000,
            default_recall_top_k: 10,
            default_graph_depth: 2,
            decay_lambda: 0.01,
            auto_link_candidates: 20,
            session_recency_window: 7,
        }
    }
}
