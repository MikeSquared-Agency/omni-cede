use std::collections::HashSet;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::graph;
use crate::hnsw::VectorIndex;
use crate::types::*;

use std::sync::Arc;
use tokio::sync::RwLock;

// ─── recall ─────────────────────────────────────────────

/// Hybrid semantic + graph search.
///
/// 1. Embed input
/// 2. HNSW k-NN
/// 3. BFS graph walk from candidates
/// 4. Score, rank, return
pub async fn recall(
    db: &Db,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    input: &str,
    opts: RecallOptions,
) -> Result<Vec<ScoredNode>> {
    // 1. Embed
    let query_vec = embed.embed(input).await?;

    // 2. HNSW search
    let top_k = opts.top_k;
    let candidates = {
        let index = hnsw.read().await;
        index.search(&query_vec, top_k)
    };

    // 3. BFS graph walk from candidates
    let seed_ids: Vec<NodeId> = candidates.iter().map(|(id, _)| id.clone()).collect();
    let depth = opts.depth;

    let walked = db
        .call(move |conn| graph::bfs_walk(conn, &seed_ids, depth))
        .await?;

    // 4. Collect all unique node IDs (HNSW hits + walked)
    let mut all_ids: HashSet<NodeId> = HashSet::new();
    for (id, _) in &candidates {
        all_ids.insert(id.clone());
    }
    for (id, _) in &walked {
        all_ids.insert(id.clone());
    }

    // 5. Load nodes from DB
    let ids_clone = all_ids.clone();
    let nodes = db
        .call(move |conn| queries::get_nodes_by_ids(conn, &ids_clone))
        .await?;

    // Build lookup for candidate similarities
    let sim_map: std::collections::HashMap<NodeId, f32> =
        candidates.iter().cloned().collect();

    // 6. Score each node
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let mut scored: Vec<ScoredNode> = Vec::new();
    for node in &nodes {
        // Optional kind filter
        if let Some(ref filter) = opts.filter_kinds {
            if !filter.contains(&node.kind) {
                continue;
            }
        }

        let last = node.last_access.unwrap_or(node.created_at);
        let hours = (now - last) as f64 / 3600.0;
        let recency = (-config.decay_lambda * hours).exp();

        // Graph proximity bonus
        let hops = walked
            .get(&node.id)
            .copied()
            .unwrap_or(0);
        let proximity_bonus = (0.2 * (2.0 - hops as f64).max(0.0)).min(0.4);

        let score =
            node.importance * node.trust_score * recency * (1.0 + proximity_bonus);

        if score < opts.min_score {
            continue;
        }

        let similarity = sim_map.get(&node.id).copied().unwrap_or(0.0);

        scored.push(ScoredNode {
            node: node.clone(),
            score,
            similarity,
        });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(opts.top_k);

    // 7. Touch accessed nodes (update last_access + access_count)
    let touched_ids: Vec<NodeId> = scored.iter().map(|s| s.node.id.clone()).collect();
    if !touched_ids.is_empty() {
        db.call(move |conn| queries::touch_nodes(conn, &touched_ids))
            .await?;
    }

    Ok(scored)
}

// ─── briefing ───────────────────────────────────────────

/// Build a full briefing for the LLM system prompt.
pub async fn briefing(
    db: &Db,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    query: &str,
    max_nodes: usize,
) -> Result<Briefing> {
    briefing_with_kinds(
        db,
        embed,
        hnsw,
        config,
        query,
        &[
            NodeKind::Soul,
            NodeKind::Belief,
            NodeKind::Goal,
            NodeKind::Fact,
            NodeKind::Decision,
            NodeKind::Pattern,
            NodeKind::Capability,
            NodeKind::Limitation,
        ],
        max_nodes,
    )
    .await
}

/// Build a briefing filtered to specific node kinds.
pub async fn briefing_with_kinds(
    db: &Db,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    query: &str,
    kinds: &[NodeKind],
    max_nodes: usize,
) -> Result<Briefing> {
    let opts = RecallOptions {
        top_k: max_nodes,
        depth: config.default_graph_depth,
        min_score: 0.0,
        filter_kinds: Some(kinds.to_vec()),
    };

    let nodes = recall(db, embed, hnsw, config, query, opts).await?;

    // Fetch contradictions involving returned nodes
    let node_ids: Vec<NodeId> = nodes.iter().map(|s| s.node.id.clone()).collect();
    let contradictions = db
        .call(move |conn| queries::get_unresolved_contradictions(conn, &node_ids))
        .await?;

    // Compute trust summary
    let trust_summary = if nodes.is_empty() {
        TrustSummary {
            min_trust: 0.0,
            max_trust: 0.0,
            mean_trust: 0.0,
            low_trust_count: 0,
        }
    } else {
        let trusts: Vec<f64> = nodes.iter().map(|s| s.node.trust_score).collect();
        let min = trusts.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = trusts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = trusts.iter().sum::<f64>() / trusts.len() as f64;
        let low = trusts.iter().filter(|&&t| t < 0.5).count();
        TrustSummary {
            min_trust: min,
            max_trust: max,
            mean_trust: mean,
            low_trust_count: low,
        }
    };

    // Build context_doc
    let context_doc = format_context_doc(&nodes, &contradictions);

    Ok(Briefing {
        nodes,
        contradictions,
        trust_summary,
        context_doc,
    })
}

/// Render the briefing as a markdown document for the LLM system prompt.
fn format_context_doc(nodes: &[ScoredNode], contradictions: &[ContradictionPair]) -> String {
    let mut doc = String::new();

    // Who you are
    let identity: Vec<&ScoredNode> = nodes
        .iter()
        .filter(|s| matches!(s.node.kind, NodeKind::Soul | NodeKind::Belief | NodeKind::Goal))
        .collect();
    if !identity.is_empty() {
        doc.push_str("## Who you are\n");
        for s in &identity {
            let body = s.node.body.as_deref().unwrap_or("");
            doc.push_str(&format!(
                "- **[{}]** {}: {}\n",
                s.node.kind, s.node.title, body
            ));
        }
        doc.push('\n');
    } else {
        // Bootstrap prompt — no identity exists yet.  Rather than
        // prescribing a personality, nudge the agent to discover who it
        // is through conversation with the user, then persist what it
        // learns via the `remember` tool. Once Soul/Belief/Goal nodes
        // exist with embeddings this fallback stops appearing.
        doc.push_str("## First contact\n");
        doc.push_str("You have no memory yet — this is a blank slate.\n\n");
        doc.push_str("Start by finding out who you're talking to: ask their name, what they need from you, and what role they want you to play. ");
        doc.push_str("Let the conversation shape who you become.\n\n");
        doc.push_str("You have a `remember` tool that stores things permanently in your memory graph. ");
        doc.push_str("As you learn about yourself and the people you talk to, use it to build your own identity:\n");
        doc.push_str("- `Soul` — who you are, your name, your nature\n");
        doc.push_str("- `Belief` — values and principles you adopt\n");
        doc.push_str("- `Goal` — what you're working towards\n");
        doc.push_str("- `Fact` / `Entity` — things you learn about the world and people\n\n");
        doc.push_str("Don't invent a persona. Let it emerge from what you're told and what you observe.\n\n");
    }

    // What you know
    let knowledge: Vec<&ScoredNode> = nodes
        .iter()
        .filter(|s| {
            matches!(
                s.node.kind,
                NodeKind::Fact
                    | NodeKind::Entity
                    | NodeKind::Concept
                    | NodeKind::Decision
                    | NodeKind::Pattern
                    | NodeKind::Capability
                    | NodeKind::Limitation
            )
        })
        .collect();
    if !knowledge.is_empty() {
        doc.push_str("## What you know\n");
        for s in &knowledge {
            let body = s.node.body.as_deref().unwrap_or("");
            let trust_flag = if s.node.trust_score < 0.5 {
                " ⚠ LOW TRUST"
            } else {
                ""
            };
            doc.push_str(&format!(
                "- **[{}]** {} (trust: {:.2}, score: {:.3}){}\n  {}\n",
                s.node.kind, s.node.title, s.node.trust_score, s.score, trust_flag, body
            ));
        }
        doc.push('\n');
    }

    // Recent conversation context (UserInput nodes from previous turns)
    let conversation: Vec<&ScoredNode> = nodes
        .iter()
        .filter(|s| matches!(s.node.kind, NodeKind::UserInput))
        .collect();
    if !conversation.is_empty() {
        doc.push_str("## Recent conversation\n");
        for s in &conversation {
            let body = s.node.body.as_deref().unwrap_or(&s.node.title);
            doc.push_str(&format!(
                "- User said (score: {:.3}): {}\n",
                s.score, body
            ));
        }
        doc.push('\n');
    }

    // Active contradictions
    if !contradictions.is_empty() {
        doc.push_str("## Active contradictions\n");
        for c in contradictions {
            doc.push_str(&format!(
                "- CONFLICT: node {} ↔ node {} (unresolved)\n",
                &c.node_a[..8.min(c.node_a.len())],
                &c.node_b[..8.min(c.node_b.len())],
            ));
        }
        doc.push('\n');
    }

    // What to verify
    let stale_or_untrusted: Vec<&ScoredNode> = nodes
        .iter()
        .filter(|s| s.node.trust_score < 0.5)
        .collect();
    if !stale_or_untrusted.is_empty() {
        doc.push_str("## What to verify\n");
        for s in &stale_or_untrusted {
            doc.push_str(&format!(
                "- {} (trust: {:.2})\n",
                s.node.title, s.node.trust_score
            ));
        }
        doc.push('\n');
    }

    doc
}
