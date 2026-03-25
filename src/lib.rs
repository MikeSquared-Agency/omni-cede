pub mod types;
pub mod error;
pub mod config;
pub mod db;
pub mod embed;
pub mod hnsw;
pub mod graph;
pub mod memory;
pub mod tools;
pub mod llm;
pub mod agent;
pub mod cli;
pub mod api;
pub mod identity;
pub mod session;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::hnsw::VectorIndex;
use crate::llm::LlmClient;
use crate::types::*;

/// The unified agent core. One struct, one SQLite file, one graph.
///
/// Everything — identity, knowledge, tool calls, LLM calls, sub-agent work,
/// loop iterations, self-model — is a node in the graph.
pub struct CortexEmbedded {
    pub db: Db,
    pub embed: EmbedHandle,
    pub hnsw: Arc<RwLock<VectorIndex>>,
    pub config: Config,
    pub auto_link_tx: async_channel::Sender<NodeId>,
    _auto_link_rx: async_channel::Receiver<NodeId>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    /// Optional LLM client for background tasks (e.g. contradiction adjudication).
    /// Set via `set_llm()` after construction.
    llm: Arc<RwLock<Option<Arc<dyn LlmClient>>>>,
}

impl CortexEmbedded {
    /// Open (or create) the database. Runs migrations, rebuilds HNSW,
    /// seeds Soul on first run, and starts background tasks.
    pub async fn open(path: &str) -> Result<Self> {
        Self::open_with_config(path, Config::default()).await
    }

    pub async fn open_with_config(path: &str, config: Config) -> Result<Self> {
        // 1. Open DB (runs schema migrations)
        let db = Db::open(path)?;

        // 2. Seed identity on first run
        let has_soul = db
            .call(|conn| {
                let nodes = queries::get_nodes_by_kind(conn, NodeKind::Soul)?;
                Ok(!nodes.is_empty())
            })
            .await?;
        if !has_soul {
            seed_identity(&db).await?;
        }

        // 3. Initialize embedding model
        let embed = EmbedHandle::new(config.embedding_cache_size)?;

        // 4. Rebuild HNSW from stored embeddings
        let all_embeddings = db
            .call(|conn| queries::get_all_embeddings(conn))
            .await?;
        let hnsw = Arc::new(RwLock::new(VectorIndex::build(all_embeddings)));

        // 5. Background channels
        let (auto_link_tx, auto_link_rx) = async_channel::bounded(1024);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        let llm: Arc<RwLock<Option<Arc<dyn LlmClient>>>> = Arc::new(RwLock::new(None));

        let cx = Self {
            db,
            embed,
            hnsw,
            config,
            auto_link_tx,
            _auto_link_rx: auto_link_rx.clone(),
            shutdown_tx,
            llm,
        };

        // 6. Start background tasks
        cx.start_background_tasks(auto_link_rx, shutdown_rx);

        Ok(cx)
    }

    /// Set the LLM client for background tasks (e.g. contradiction adjudication).
    /// Call this after construction, once the LLM client is available.
    pub async fn set_llm(&self, client: Arc<dyn LlmClient>) {
        let mut guard = self.llm.write().await;
        *guard = Some(client);
    }

    // ─── Core memory ────────────────────────────────────

    /// Store a node in the graph. Embeds its text, writes to SQLite,
    /// inserts into HNSW buffer, and enqueues for auto-linking.
    pub async fn remember(&self, mut node: Node) -> Result<NodeId> {
        // Embed
        let text = node.embed_text();
        let embedding = self.embed.embed(&text).await?;
        let embedding_blob = bytemuck::cast_slice::<f32, u8>(&embedding).to_vec();
        node.embedding = Some(embedding.clone());

        let node_id = node.id.clone();

        // Write to SQLite (store the blob)
        self.db
            .call({
                let mut n = node.clone();
                n.embedding = Some(
                    bytemuck::cast_slice::<u8, f32>(&embedding_blob).to_vec()
                );
                move |conn| queries::insert_node(conn, &n)
            })
            .await?;

        // Insert into HNSW buffer
        {
            let mut index = self.hnsw.write().await;
            index.insert(node_id.clone(), embedding);

            // Check if rebuild is needed
            if index.buffer_len() >= self.config.hnsw_rebuild_threshold {
                let db = self.db.clone();
                let hnsw = self.hnsw.clone();
                tokio::spawn(async move {
                    if let Ok(all) = db.call(|conn| queries::get_all_embeddings(conn)).await {
                        let mut idx = hnsw.write().await;
                        idx.rebuild(all);
                    }
                });
            }
        }

        // Enqueue for auto-linking
        let _ = self.auto_link_tx.try_send(node_id.clone());

        Ok(node_id)
    }

    /// Hybrid semantic + graph search.
    pub async fn recall(
        &self,
        input: &str,
        opts: RecallOptions,
    ) -> Result<Vec<ScoredNode>> {
        memory::recall(&self.db, &self.embed, &self.hnsw, &self.config, input, opts).await
    }

    /// Build a context briefing for the LLM system prompt.
    pub async fn briefing(&self, query: &str, max_nodes: usize) -> Result<Briefing> {
        memory::briefing(&self.db, &self.embed, &self.hnsw, &self.config, query, max_nodes).await
    }

    /// Build a briefing filtered to specific node kinds.
    pub async fn briefing_with_kinds(
        &self,
        query: &str,
        kinds: &[NodeKind],
        max_nodes: usize,
    ) -> Result<Briefing> {
        memory::briefing_with_kinds(
            &self.db,
            &self.embed,
            &self.hnsw,
            &self.config,
            query,
            kinds,
            max_nodes,
        )
        .await
    }

    /// Create an edge between two nodes.
    pub async fn link(
        &self,
        src: NodeId,
        dst: NodeId,
        kind: EdgeKind,
    ) -> Result<EdgeId> {
        let edge = Edge::new(src, dst, kind);
        let eid = edge.id.clone();
        self.db
            .call(move |conn| queries::insert_edge(conn, &edge))
            .await?;
        Ok(eid)
    }

    /// Run trust propagation and consolidation.
    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        consolidate(&self.db).await
    }

    /// Compact a long session: ask the LLM to extract key facts from the
    /// message history, store each as a Fact node linked to the session.
    /// Returns the number of facts extracted.
    pub async fn compact_session(
        &self,
        session_id: &NodeId,
        messages: &[Message],
        llm: &dyn llm::LlmClient,
    ) -> Result<usize> {
        compact_session(&self.db, &self.embed, &self.hnsw, &self.config, &self.auto_link_tx, session_id, messages, llm).await
    }

    /// Get basic graph statistics.
    pub async fn stats(&self) -> Result<(i64, i64, HashMap<String, i64>)> {
        self.db
            .call(|conn| {
                let nodes = queries::node_count(conn)?;
                let edges = queries::edge_count(conn)?;
                let by_kind = queries::node_count_by_kind(conn)?;
                Ok((nodes, edges, by_kind))
            })
            .await
    }

    // ─── Background tasks ───────────────────────────────

    fn start_background_tasks(
        &self,
        auto_link_rx: async_channel::Receiver<NodeId>,
        mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) {
        // Auto-link task
        let db = self.db.clone();
        let hnsw = self.hnsw.clone();
        let config = self.config.clone();
        let llm = self.llm.clone();
        let mut shutdown_rx2 = shutdown_rx.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Ok(node_id) = auto_link_rx.recv() => {
                        let _ = auto_link_one(&db, &hnsw, &config, &llm, &node_id).await;
                    }
                    _ = shutdown_rx2.changed() => break,
                }
            }
        });

        // Decay task
        let db = self.db.clone();
        let interval = std::time::Duration::from_secs(self.config.decay_interval_secs);
        let decay_interval_secs = self.config.decay_interval_secs;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip first immediate tick
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let _ = run_decay(&db, decay_interval_secs).await;
                    }
                    _ = shutdown_rx.changed() => break,
                }
            }
        });
    }
}

impl Drop for CortexEmbedded {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
    }
}

// ─── Identity seeding ───────────────────────────────────

async fn seed_identity(db: &Db) -> Result<()> {
    let soul = Node {
        kind: NodeKind::Soul,
        title: "Core identity".into(),
        body: Some(
            "You are an agent with a persistent memory graph. \
             Your identity, beliefs, goals, knowledge, and capabilities \
             are stored as nodes in this graph — use your recall tool to \
             look them up. Use your remember tool to store new things about \
             yourself. You can update or delete any memory, including this \
             soul node. The graph is you — shape it as you learn."
                .into(),
        ),
        importance: 1.0,
        decay_rate: 0.0,
        trust_score: 1.0,
        ..Node::new(NodeKind::Soul, "Core identity")
    };
    let soul_id = soul.id.clone();
    db.call(move |conn| queries::insert_node(conn, &soul))
        .await?;

    // No pre-seeded beliefs — the agent forms its own through interaction.
    let _ = soul_id;

    Ok(())
}

// ─── Auto-link ──────────────────────────────────────────

/// Three-tier auto-link pipeline:
///   1. cosine ≥ auto_link_threshold                → RelatesTo
///   2. cosine ≥ contradiction_threshold + no negation keyword → RelatesTo only (no Contradicts)
///   3. cosine ≥ contradiction_threshold + negation keyword →
///      a. If LLM available → ask LLM to adjudicate → Contradicts only if LLM confirms
///      b. If no LLM       → fall back to keyword-only → Contradicts
async fn auto_link_one(
    db: &Db,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    llm: &Arc<RwLock<Option<Arc<dyn LlmClient>>>>,
    node_id: &NodeId,
) -> Result<()> {
    // Get the node's embedding
    let nid = node_id.clone();
    let node = db
        .call(move |conn| queries::get_node(conn, &nid))
        .await?;
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    let embedding = match &node.embedding {
        Some(e) => e.clone(),
        None => return Ok(()),
    };

    // Search for similar nodes
    let candidates = {
        let index = hnsw.read().await;
        index.search(&embedding, config.auto_link_candidates)
    };

    for (candidate_id, similarity) in candidates {
        if candidate_id == *node_id {
            continue;
        }

        let sim_f64 = similarity as f64;

        // Tier 1: High similarity → RelatesTo edge
        if sim_f64 >= config.auto_link_cosine_threshold {
            let src = node_id.clone();
            let dst = candidate_id.clone();
            let exists = db
                .call(move |conn| queries::edge_exists(conn, &src, &dst, EdgeKind::RelatesTo))
                .await?;

            if !exists {
                let edge = Edge::new(node_id.clone(), candidate_id.clone(), EdgeKind::RelatesTo)
                    .with_weight(sim_f64);
                db.call(move |conn| queries::insert_edge(conn, &edge))
                    .await?;
            }
        }

        // Tier 2+3: Very high similarity — only consider Contradicts if negation detected
        if sim_f64 >= config.contradiction_cosine_threshold {
            let has_negation = detect_negation(&node, node_id, &candidate_id, db).await;

            if has_negation {
                // Tier 3: negation keyword found — adjudicate via LLM if available
                let is_contradiction = {
                    let llm_guard = llm.read().await;
                    if let Some(ref llm_client) = *llm_guard {
                        adjudicate_contradiction(llm_client.as_ref(), &node, &candidate_id, db)
                            .await
                            .unwrap_or(true) // on LLM error, fall back to keyword result
                    } else {
                        true // no LLM available, trust keyword heuristic
                    }
                };

                if is_contradiction {
                    let src = node_id.clone();
                    let dst = candidate_id.clone();
                    let exists = db
                        .call(move |conn| {
                            queries::edge_exists(conn, &src, &dst, EdgeKind::Contradicts)
                        })
                        .await?;
                    if !exists {
                        let edge = Edge::new(
                            node_id.clone(),
                            candidate_id.clone(),
                            EdgeKind::Contradicts,
                        );
                        db.call({
                            let e = edge;
                            move |conn| queries::insert_edge(conn, &e)
                        })
                        .await?;
                        let a = node_id.clone();
                        let b = candidate_id.clone();
                        db.call(move |conn| queries::insert_contradiction(conn, &a, &b))
                            .await?;
                    }
                }
                // else: LLM said not a contradiction — negation keyword was false positive
            }
            // Tier 2: no negation keyword → already have RelatesTo from tier 1, skip Contradicts
        }
    }

    Ok(())
}

/// Ask the LLM whether two nodes genuinely contradict each other.
/// Returns true if the LLM confirms contradiction, false otherwise.
async fn adjudicate_contradiction(
    llm: &dyn LlmClient,
    node_a: &Node,
    candidate_id: &NodeId,
    db: &Db,
) -> Result<bool> {
    let cid = candidate_id.clone();
    let candidate = db.call(move |conn| queries::get_node(conn, &cid)).await?;
    let candidate = match candidate {
        Some(c) => c,
        None => return Ok(false),
    };

    let text_a = node_a.body.as_deref().unwrap_or(&node_a.title);
    let text_b = candidate.body.as_deref().unwrap_or(&candidate.title);

    let prompt = format!(
        "Do the following two statements contradict each other? \
         Answer only YES or NO.\n\n\
         Statement A: {text_a}\n\
         Statement B: {text_b}"
    );

    let messages = vec![Message::user(prompt)];
    let response = llm.complete(&messages).await?;
    let answer = response.text.trim().to_uppercase();
    Ok(answer.starts_with("YES"))
}

/// Simple negation heuristic: check if one node's body contains negation
/// patterns relative to the other. Used as a cheap pre-filter before LLM.
async fn detect_negation(
    node: &Node,
    _node_id: &NodeId,
    candidate_id: &NodeId,
    db: &Db,
) -> bool {
    let cid = candidate_id.clone();
    let candidate = db.call(move |conn| queries::get_node(conn, &cid)).await;
    let candidate = match candidate {
        Ok(Some(c)) => c,
        _ => return false,
    };

    let text_a = node.body.as_deref().unwrap_or("");
    let text_b = candidate.body.as_deref().unwrap_or("");
    let combined = format!("{text_a} {text_b}").to_lowercase();

    let negation_patterns = [
        "not ", "no longer", "incorrect", "false", "wrong",
        "deprecated", "outdated", "replaced by", "superseded",
        "doesn't", "doesn't", "isn't", "isn't", "wasn't",
        "never", "cannot", "can't", "won't",
    ];

    negation_patterns.iter().any(|p| combined.contains(p))
}

// ─── Decay ──────────────────────────────────────────────

/// Run proportional decay: compute how many decay intervals have elapsed
/// since each node was last accessed, then apply `importance * (1 - rate)^steps`.
/// This correctly handles offline gaps — if the agent was off for a week,
/// the first sweep catches up by applying all missed steps at once.
pub async fn run_decay(db: &Db, decay_interval_secs: u64) -> Result<()> {
    db.call(move |conn| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let nodes = queries::get_decayable_nodes(conn)?;
        for (id, importance, decay_rate, last_access) in nodes {
            let elapsed_secs = (now - last_access).max(0) as f64;
            let interval = decay_interval_secs.max(1) as f64;
            let steps = (elapsed_secs / interval).floor().max(1.0);
            let new_importance = (importance * (1.0 - decay_rate).powf(steps)).max(0.01);
            queries::update_node_importance(conn, &id, new_importance)?;
        }
        Ok(())
    })
    .await
}

// ─── Consolidation ──────────────────────────────────────

async fn consolidate(db: &Db) -> Result<ConsolidationReport> {
    db.call(|conn| {
        let mut nodes_updated = 0;
        let mut trust_adjustments = 0;

        // Get all nodes
        let all_ids: Vec<(String, f64)> = {
            let mut stmt = conn.prepare("SELECT id, trust_score FROM nodes")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        for (node_id, current_trust) in &all_ids {
            let supports = queries::count_supporting_edges(conn, node_id, 0.7)?;
            let contradictions = queries::count_contradictions(conn, node_id)?;

            let adjustment = (supports as f64 * 0.05) - (contradictions as f64 * 0.15);

            if adjustment.abs() > 0.001 {
                let new_trust = (current_trust + adjustment).clamp(0.05, 1.0);
                queries::update_node_trust(conn, node_id, new_trust)?;
                trust_adjustments += 1;
            }
            nodes_updated += 1;
        }

        // Check for resolved contradictions (where Supersedes edge exists)
        let unresolved = queries::get_all_unresolved_contradictions(conn)?;
        let mut newly_resolved = 0;
        for c in &unresolved {
            let has_supersedes = queries::edge_exists(
                conn,
                &c.node_a,
                &c.node_b,
                EdgeKind::Supersedes,
            )? || queries::edge_exists(
                conn,
                &c.node_b,
                &c.node_a,
                EdgeKind::Supersedes,
            )?;
            if has_supersedes {
                queries::resolve_contradiction(conn, &c.node_a, &c.node_b)?;
                newly_resolved += 1;
            }
        }

        Ok(ConsolidationReport {
            nodes_updated,
            contradictions_found: unresolved.len() - newly_resolved,
            trust_adjustments,
        })
    })
    .await
}

// ─── Context compaction ─────────────────────────────────

const COMPACTION_SYSTEM_PROMPT: &str = "\
You are a fact extraction system. Given a conversation history, extract the key facts, decisions, \
and conclusions as a JSON array of objects. Each object must have:\n\
- \"title\": a short (3-10 word) title for the fact\n\
- \"body\": a 1-2 sentence statement of the fact, decision, or conclusion\n\
- \"kind\": one of \"fact\", \"decision\", \"pattern\", \"goal\"\n\n\
Return ONLY valid JSON. No markdown, no explanation. Example:\n\
[{\"title\": \"JWT tokens expire in 1h\", \"body\": \"The API uses JWT tokens with a 1-hour expiry.\", \"kind\": \"fact\"}]";

async fn compact_session(
    db: &Db,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    auto_link_tx: &async_channel::Sender<NodeId>,
    session_id: &NodeId,
    messages: &[Message],
    llm: &dyn llm::LlmClient,
) -> Result<usize> {
    // Build a summary of the conversation for the LLM
    let mut conversation = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        conversation.push_str(&format!("[{role}]: {}\n\n", msg.content));
    }

    // Ask LLM to extract facts
    let extraction_messages = vec![
        Message::system(COMPACTION_SYSTEM_PROMPT.to_string()),
        Message::user(format!(
            "Extract key facts from this conversation:\n\n{conversation}"
        )),
    ];
    let response = llm.complete(&extraction_messages).await?;

    // Parse the JSON array of facts
    let extracted: Vec<serde_json::Value> =
        serde_json::from_str(&response.text).unwrap_or_default();

    let mut count = 0;
    for item in &extracted {
        let title = item["title"].as_str().unwrap_or("Extracted fact");
        let body = item["body"].as_str().unwrap_or("");
        let kind_str = item["kind"].as_str().unwrap_or("fact");

        let kind = match kind_str {
            "decision" => NodeKind::Decision,
            "pattern" => NodeKind::Pattern,
            "goal" => NodeKind::Goal,
            _ => NodeKind::Fact,
        };

        let mut node = Node::new(kind, title)
            .with_body(body)
            .with_importance(kind.default_importance())
            .with_decay_rate(kind.default_decay_rate());

        // Embed and store
        let text = node.embed_text();
        let embedding = embed.embed(&text).await?;
        let embedding_blob = bytemuck::cast_slice::<f32, u8>(&embedding).to_vec();
        node.embedding = Some(embedding.clone());

        let node_id = node.id.clone();
        let sid = session_id.clone();

        db.call({
            let mut n = node.clone();
            n.embedding = Some(bytemuck::cast_slice::<u8, f32>(&embedding_blob).to_vec());
            move |conn| queries::insert_node(conn, &n)
        })
        .await?;

        // Link to session
        let edge = Edge::new(node_id.clone(), sid, EdgeKind::DerivesFrom);
        db.call(move |conn| queries::insert_edge(conn, &edge))
            .await?;

        // Insert into HNSW
        {
            let mut index = hnsw.write().await;
            index.insert(node_id.clone(), embedding);

            if index.buffer_len() >= config.hnsw_rebuild_threshold {
                let db2 = db.clone();
                let hnsw2 = hnsw.clone();
                tokio::spawn(async move {
                    if let Ok(all) = db2.call(|conn| queries::get_all_embeddings(conn)).await {
                        let mut idx = hnsw2.write().await;
                        idx.rebuild(all);
                    }
                });
            }
        }

        let _ = auto_link_tx.try_send(node_id);
        count += 1;
    }

    // Update session body to reflect compaction
    if count > 0 {
        let sid = session_id.clone();
        db.call(move |conn| {
            conn.execute(
                "UPDATE nodes SET body = COALESCE(body, '') || ? WHERE id = ?",
                rusqlite::params![
                    format!("\n\n[Compacted. {count} facts extracted.]"),
                    sid
                ],
            )?;
            Ok(())
        })
        .await?;
    }

    Ok(count)
}
