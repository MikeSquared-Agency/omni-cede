//! Proactive cron scheduler.
//!
//! Loads `CronJob` nodes from the graph and fires them on schedule.
//! Each execution spawns a short-lived Agent loop and records a
//! `CronExecution` node linked to the originating `CronJob`.

use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::hnsw::VectorIndex;
use crate::llm::LlmClient;
use crate::memory::format_timestamp;
use crate::notification_delivery::{NotifEvent, NotifTx};
use crate::types::*;

/// Metadata stored in a CronJob node's body (JSON).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CronJobMeta {
    /// Standard cron expression (5 or 7 fields).
    pub cron: String,
    /// The task prompt to run when the schedule fires.
    pub task: String,
    /// Maximum agent loop iterations per execution (default 5).
    #[serde(default = "default_max_iter")]
    pub max_iterations: usize,
    /// Whether this job is active.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Unix timestamp of the last successful fire (0 = never).
    #[serde(default)]
    pub last_fired: i64,
    /// The user who created this job (for routing results back).
    #[serde(default)]
    pub user_id: Option<String>,
    /// The channel from which this job was created.
    #[serde(default)]
    pub channel: Option<String>,
}

fn default_max_iter() -> usize { 5 }
fn default_enabled() -> bool { true }

/// Run the scheduler loop. Call this from a `tokio::spawn`.
///
/// Every `tick_secs` seconds it loads all CronJob nodes, evaluates them
/// against the current time, and fires any that are due.
pub async fn run(
    db: Db,
    embed: EmbedHandle,
    hnsw: Arc<RwLock<VectorIndex>>,
    auto_link_tx: async_channel::Sender<NodeId>,
    llm: Arc<RwLock<Option<Arc<dyn LlmClient>>>>,
    config: Config,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    tick_secs: u64,
    notif_tx: Option<NotifTx>,
) {
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(tick_secs));
    ticker.tick().await; // skip the first immediate tick

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Err(e) = tick(&db, &embed, &hnsw, &auto_link_tx, &llm, &config, &notif_tx).await {
                    tracing::warn!("scheduler tick error: {e}");
                }
            }
            _ = shutdown_rx.changed() => {
                tracing::info!("scheduler shutting down");
                break;
            }
        }
    }
}

/// Single scheduler tick — evaluate all CronJob nodes and fire any due.
async fn tick(
    db: &Db,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    auto_link_tx: &async_channel::Sender<NodeId>,
    llm: &Arc<RwLock<Option<Arc<dyn LlmClient>>>>,
    config: &Config,
    notif_tx: &Option<NotifTx>,
) -> Result<()> {
    // 1. Load all CronJob nodes
    let cron_nodes = db
        .call(|conn| queries::get_nodes_by_kind(conn, NodeKind::CronJob))
        .await?;

    if cron_nodes.is_empty() {
        return Ok(());
    }

    let now = chrono::Utc::now();
    let now_ts = now.timestamp();

    for node in &cron_nodes {
        let meta: CronJobMeta = match &node.body {
            Some(body) => match serde_json::from_str(body) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("invalid CronJob meta for {}: {e}", &node.id[..8]);
                    continue;
                }
            },
            None => continue,
        };

        if !meta.enabled {
            continue;
        }

        // Parse the cron expression
        let schedule = match cron::Schedule::from_str(&meta.cron) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("bad cron expr '{}' for {}: {e}", meta.cron, &node.id[..8]);
                continue;
            }
        };

        // Determine if this job should fire:
        // Find the most recent scheduled time <= now, and check if it's after last_fired.
        let should_fire = if meta.last_fired == 0 {
            // Never fired — fire on the first tick
            true
        } else {
            let last_fired_dt = chrono::DateTime::from_timestamp(meta.last_fired, 0)
                .unwrap_or(chrono::DateTime::UNIX_EPOCH);
            // Check if any scheduled time exists between last_fired and now
            schedule
                .after(&last_fired_dt)
                .take(1)
                .any(|next| next <= now)
        };

        if !should_fire {
            continue;
        }

        tracing::info!("firing cron job '{}' ({})", node.title, &node.id[..8]);

        // Update last_fired in the node's body
        {
            let mut updated_meta = meta.clone();
            updated_meta.last_fired = now_ts;
            let new_body = serde_json::to_string(&updated_meta).unwrap_or_default();
            let nid = node.id.clone();
            db.call(move |conn| {
                conn.execute(
                    "UPDATE nodes SET body = ?1 WHERE id = ?2",
                    rusqlite::params![new_body, nid],
                )?;
                Ok(())
            })
            .await?;
        }

        // Get an LLM client, or skip if none set
        let llm_client = {
            let guard = llm.read().await;
            match &*guard {
                Some(c) => c.clone(),
                None => {
                    tracing::warn!("no LLM configured — skipping cron execution");
                    continue;
                }
            }
        };

        // Spawn the execution as a background task
        fire_cron_job(
            db.clone(),
            embed.clone(),
            hnsw.clone(),
            auto_link_tx.clone(),
            llm_client,
            config.clone(),
            node.id.clone(),
            node.title.clone(),
            meta.task.clone(),
            meta.max_iterations,
            meta.user_id.clone(),
            meta.channel.clone(),
            notif_tx.clone(),
        );
    }

    Ok(())
}

/// Spawn a background agent loop for a cron execution.
fn fire_cron_job(
    db: Db,
    embed: EmbedHandle,
    hnsw: Arc<RwLock<VectorIndex>>,
    auto_link_tx: async_channel::Sender<NodeId>,
    llm: Arc<dyn LlmClient>,
    config: Config,
    cron_job_id: NodeId,
    job_title: String,
    task: String,
    max_iterations: usize,
    user_id: Option<String>,
    channel: Option<String>,
    notif_tx: Option<NotifTx>,
) {
    tokio::spawn(async move {
        // 1. Create a CronExecution node
        let exec_node = Node::new(NodeKind::CronExecution, format!("[{}] Ran scheduled task: {job_title}", format_timestamp(crate::types::now_unix())))
            .with_body(&format!("Status: running\nTask: {task}"));
        let exec_id = exec_node.id.clone();
        if let Err(e) = db
            .call({
                let n = exec_node;
                move |conn| queries::insert_node(conn, &n)
            })
            .await
        {
            tracing::error!("failed to create CronExecution node: {e}");
            return;
        }

        // 2. Link CronExecution → CronJob via DerivesFrom
        let edge = Edge::new(exec_id.clone(), cron_job_id.clone(), EdgeKind::DerivesFrom);
        let _ = db
            .call(move |conn| queries::insert_edge(conn, &edge))
            .await;

        // 3. Build a tools registry and agent
        let tools = crate::tools::builtin_registry_core(
            db.clone(),
            embed.clone(),
            hnsw.clone(),
            auto_link_tx.clone(),
            None, // no recursive spawn_task from cron
            config.clone(),
        );

        let mut agent_config = config;
        agent_config.max_iterations = max_iterations;

        let agent = crate::agent::orchestrator::Agent {
            db: db.clone(),
            embed,
            hnsw,
            config: agent_config,
            llm,
            tools,
            auto_link_tx: auto_link_tx.clone(),
            notif_tx: None, // cron agent doesn't trigger event-driven delivery itself
        };

        // 4. Run the agent loop
        let result = agent.run(&task).await;

        // 5. Update the CronExecution node with results
        let (status, result_body) = match &result {
            Ok(answer) => ("completed", format!("Status: completed\n\n{answer}")),
            Err(e) => ("failed", format!("Status: failed\n\nError: {e}")),
        };

        // Store result as a Fact linked to the execution
        let fact = Node::new(NodeKind::Fact, format!("[{}] Result of scheduled task: {job_title}", format_timestamp(crate::types::now_unix())))
            .with_body(&result_body)
            .with_importance(0.5);
        let fact_id = fact.id.clone();
        let _ = db
            .call({
                let f = fact;
                move |conn| queries::insert_node(conn, &f)
            })
            .await;

        let derives = Edge::new(fact_id.clone(), exec_id.clone(), EdgeKind::DerivesFrom);
        let _ = db
            .call(move |conn| queries::insert_edge(conn, &derives))
            .await;

        let _ = auto_link_tx.try_send(fact_id);

        // Create a Notification in the user's session so it gets delivered
        // to the right channel by the notification delivery loop.
        if let (Some(ref uid), Some(ref ch)) = (&user_id, &channel) {
            let uid2 = uid.clone();
            let ch2 = ch.clone();
            let session_id: Option<String> = db
                .call(move |conn| {
                    crate::session::create_tables(conn)?;
                    let mut stmt = conn.prepare(
                        "SELECT node_id FROM managed_sessions WHERE user_id = ?1 AND channel = ?2",
                    )?;
                    let rows: Vec<String> = stmt
                        .query_map(rusqlite::params![uid2, ch2], |row| row.get(0))?
                        .filter_map(|r| r.ok())
                        .collect();
                    Ok(rows.into_iter().next())
                })
                .await
                .ok()
                .flatten();

            if let Some(sid) = session_id {
                let notif = Node::new(
                    NodeKind::Notification,
                    format!(
                        "[{}] Scheduled task completed: {job_title}",
                        format_timestamp(crate::types::now_unix())
                    ),
                )
                .with_body(&result_body);
                let notif_id = notif.id.clone();
                let _ = db
                    .call({
                        let n = notif;
                        move |conn| queries::insert_node(conn, &n)
                    })
                    .await;
                let notif_edge = Edge::new(notif_id, sid.clone(), EdgeKind::PartOf);
                let _ = db
                    .call(move |conn| queries::insert_edge(conn, &notif_edge))
                    .await;
                // Fire event for immediate delivery
                if let Some(ref tx) = notif_tx {
                    let _ = tx.send(NotifEvent { session_id: sid });
                }
                tracing::info!(
                    user_id = %uid.as_str(),
                    channel = %ch.as_str(),
                    "created notification for cron result in user session"
                );
            }
        }

        // Update execution node body
        let eid = exec_id;
        let _ = db
            .call(move |conn| {
                conn.execute(
                    "UPDATE nodes SET body = ?1 WHERE id = ?2",
                    rusqlite::params![result_body, eid],
                )?;
                Ok(())
            })
            .await;

        tracing::info!("cron execution [{status}]: {job_title}");
    });
}
