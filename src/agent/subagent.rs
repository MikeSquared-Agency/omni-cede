use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::hnsw::VectorIndex;
use crate::llm::LlmClient;
use crate::tools::ToolRegistry;
use crate::types::*;

use super::orchestrator::Agent;

/// Spawn a sub-agent that shares the same graph. Its work is fully
/// visible, trusted, and linked to the parent session.
pub async fn spawn_subagent(
    spec: SubAgentSpec,
    task: &str,
    parent_session: NodeId,
    db: &Db,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    llm: Arc<dyn LlmClient>,
    tools: ToolRegistry,
    auto_link_tx: async_channel::Sender<NodeId>,
) -> Result<SubAgentResult> {
    // 1. Write SubAgent node
    let sub_node = Node::new(NodeKind::SubAgent, &spec.name)
        .with_body(format!(
            "Soul: {}\nCapabilities: {}",
            spec.soul,
            spec.capabilities.join(", ")
        ));
    let sub_id = sub_node.id.clone();
    db.call({
        let n = sub_node;
        move |conn| queries::insert_node(conn, &n)
    })
    .await?;

    // 2. Write Delegation node
    let deleg = Node::new(NodeKind::Delegation, format!("Delegate: {}", task))
        .with_body(task);
    let deleg_id = deleg.id.clone();
    db.call({
        let n = deleg;
        move |conn| queries::insert_node(conn, &n)
    })
    .await?;

    // Link: Delegation → SubAgent, Delegation → parent session
    let e1 = Edge::new(deleg_id.clone(), sub_id.clone(), EdgeKind::PartOf);
    let e2 = Edge::new(deleg_id.clone(), parent_session.clone(), EdgeKind::PartOf);
    db.call(move |conn| {
        queries::insert_edge(conn, &e1)?;
        queries::insert_edge(conn, &e2)
    })
    .await?;

    // 3. Run sub-agent with scoped config
    let sub_config = Config {
        max_iterations: spec.max_iterations,
        ..config.clone()
    };

    let agent = Agent {
        db: db.clone(),
        embed: embed.clone(),
        hnsw: hnsw.clone(),
        config: sub_config,
        llm,
        tools,
        auto_link_tx: auto_link_tx.clone(),
    };

    let answer = agent.run(task).await?;

    // 4. Write Synthesis node
    let synth = Node::new(NodeKind::Synthesis, format!("Synthesis: {}", spec.name))
        .with_body(&answer);
    let synth_id = synth.id.clone();
    db.call({
        let n = synth;
        move |conn| queries::insert_node(conn, &n)
    })
    .await?;

    // Link: Synthesis → Delegation, Synthesis → parent session
    let e3 = Edge::new(synth_id.clone(), deleg_id, EdgeKind::DerivesFrom);
    let e4 = Edge::new(synth_id, parent_session, EdgeKind::PartOf);
    db.call(move |conn| {
        queries::insert_edge(conn, &e3)?;
        queries::insert_edge(conn, &e4)
    })
    .await?;

    Ok(SubAgentResult {
        answer,
        facts_created: vec![], // TODO: collect fact IDs during sub-agent run
        tokens_used: 0,
    })
}
