//! Event-driven notification delivery.
//!
//! When background tool execution or a cron job completes, the producer
//! creates a `Notification` graph node and fires a [`NotifEvent`] on a
//! tokio mpsc channel. This module awaits that channel and delivers the
//! notification immediately — no polling.
//!
//! The delivery LLM receives the same full briefing (identity, knowledge,
//! recent conversation) as the main agent — no opinionated rules. It reads
//! the context, sees the completed task result, and decides naturally
//! whether and how to tell the user.
//!
//! # Flow
//!
//! ```text
//!   NotifEvent received on mpsc channel
//!     → query pending notifications for that session
//!     → resolve outbound routing (channel + external_id)
//!     → build full briefing (same as main agent)
//!     → LLM call to formulate a natural follow-up message
//!     → Pipeline::send_outbound() to push it to the user
//!     → touch_nodes() to mark notifications as delivered
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::channels::Pipeline;
use crate::channels::types::*;
use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::hnsw::VectorIndex;
use crate::identity;
use crate::llm::LlmClient;
use crate::memory;
use crate::types::*;

/// Lightweight signal fired when a Notification node is created.
#[derive(Debug, Clone)]
pub struct NotifEvent {
    /// The session node_id the notification is linked to.
    pub session_id: String,
}

/// Sender half — cloned into every producer (Agent, scheduler).
pub type NotifTx = tokio::sync::mpsc::UnboundedSender<NotifEvent>;
/// Receiver half — owned by the delivery loop.
pub type NotifRx = tokio::sync::mpsc::UnboundedReceiver<NotifEvent>;

/// Run the event-driven notification delivery loop until shutdown.
///
/// Awaits [`NotifEvent`]s on the mpsc channel. When one arrives, queries
/// the session for pending notification nodes and delivers them immediately.
pub async fn run(
    db: Db,
    pipeline: Arc<Pipeline>,
    llm: Arc<dyn LlmClient>,
    embed: EmbedHandle,
    hnsw: Arc<RwLock<VectorIndex>>,
    config: Config,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    mut rx: NotifRx,
) {
    tracing::info!("event-driven notification delivery loop started");

    loop {
        tokio::select! {
            event = rx.recv() => {
                let Some(event) = event else {
                    tracing::info!("notification channel closed — shutting down");
                    break;
                };
                if let Err(e) = deliver_for_event(&db, &pipeline, &llm, &embed, &hnsw, &config, &event.session_id).await {
                    tracing::warn!(session_id = %event.session_id, error = %e, "notification delivery failed");
                }
            }
            _ = shutdown_rx.changed() => {
                tracing::info!("notification delivery loop shutting down");
                break;
            }
        }
    }
}

/// Handle a single event: look up routing info, fetch pending nodes, deliver.
async fn deliver_for_event(
    db: &Db,
    pipeline: &Arc<Pipeline>,
    llm: &Arc<dyn LlmClient>,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    session_id: &str,
) -> crate::error::Result<()> {
    // 1. Fetch pending notification nodes for this session
    let sid = session_id.to_string();
    let notifications = db
        .call(move |conn| queries::get_pending_notification_nodes(conn, &sid))
        .await?;
    if notifications.is_empty() {
        // Already claimed by the inline "Updates while you were away" path
        return Ok(());
    }

    // 2. Look up (user_id, channel) from managed_sessions
    let sid2 = session_id.to_string();
    let routing = db
        .call(move |conn| queries::get_session_routing(conn, &sid2))
        .await?;
    let (user_id, channel) = match routing {
        Some((uid, ch)) => (uid, ch),
        None => {
            tracing::warn!(session_id = %session_id, "no managed_session routing — cannot deliver");
            return Ok(());
        }
    };

    deliver_for_session(
        db, pipeline, llm, embed, hnsw, config,
        &user_id, &channel, session_id, &notifications,
    ).await
}

/// Deliver all pending notifications for a single session.
async fn deliver_for_session(
    db: &Db,
    pipeline: &Arc<Pipeline>,
    llm: &Arc<dyn LlmClient>,
    embed: &EmbedHandle,
    hnsw: &Arc<RwLock<VectorIndex>>,
    config: &Config,
    user_id: &str,
    channel: &str,
    session_id: &str,
    notifications: &[Node],
) -> crate::error::Result<()> {
    // Claim notifications immediately — whoever reads first, wins.
    // This closes the race window between the proactive delivery loop
    // and the main agent's "Updates while you were away" path.
    let delivered_ids: Vec<String> = notifications.iter().map(|n| n.id.clone()).collect();
    {
        let ids = delivered_ids.clone();
        db.call(move |conn| queries::touch_nodes(conn, &ids)).await?;
    }

    // 1. Resolve outbound routing: channel + external_id
    let uid = user_id.to_string();
    let ch = channel.to_string();
    let external_id = db
        .call(move |conn| {
            identity::create_tables(conn)?;
            Ok(identity::get_external_id(conn, &uid, &ch)?)
        })
        .await?;

    let external_id = match external_id {
        Some(eid) => eid,
        None => {
            tracing::warn!(
                user_id = %user_id,
                channel = %channel,
                "no external_id found — cannot deliver proactive notification"
            );
            return Ok(());
        }
    };

    let target = OutboundTarget {
        channel: channel.to_string(),
        external_id,
        group_id: None, // proactive notifications go to DMs
        reply_to_message_id: None,
        callback_url: None,
    };

    // 2. Build a full briefing — same as the main agent gets.
    //    Use the first notification body as the semantic query so recall
    //    surfaces the most relevant identity/knowledge nodes.
    let semantic_query = notifications
        .first()
        .and_then(|n| n.body.as_deref())
        .unwrap_or(&notifications[0].title);

    let brief = memory::briefing_with_kinds(
        db, embed, hnsw, config,
        semantic_query,
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
        config.default_recall_top_k,
    )
    .await?;

    let mut context_doc = brief.context_doc;

    // 3. Append recent session nodes (recency window) — mirrors orchestrator
    let sid = session_id.to_string();
    let recency_window = config.session_recency_window;
    let recent_nodes = db
        .call(move |conn| queries::get_recent_session_nodes(conn, &sid, recency_window))
        .await
        .unwrap_or_default();

    if !recent_nodes.is_empty() {
        let mut recency_section = String::new();
        for node in recent_nodes.iter().rev() {
            let body = node.body.as_deref().unwrap_or(&node.title);
            let label = match node.kind {
                NodeKind::UserInput => "User",
                _ => "Assistant",
            };
            let ts = memory::format_timestamp(node.created_at);
            let meta = memory::node_metadata_label(node);
            recency_section.push_str(&format!("- [{ts}] ({meta}) {label}: {body}\n"));
        }
        context_doc.push_str("## Session context (recent)\n");
        context_doc.push_str(&recency_section);
        context_doc.push('\n');
    }

    // 4. Append notification / background task results
    let mut notification_block = String::new();
    for node in notifications {
        let rel = memory::relative_time(node.created_at);
        let body = node.body.as_deref().unwrap_or(&node.title);
        notification_block.push_str(&format!("- ({}) {}\n", rel, body));
    }

    // Pull the full body of linked BackgroundTask nodes for richer context.
    let bg_bodies = fetch_background_task_bodies(db, notifications).await;
    let bg_context = if bg_bodies.is_empty() {
        String::new()
    } else {
        format!("\n## Full background task results\n{}\n", bg_bodies.join("\n---\n"))
    };

    context_doc.push_str(&format!(
        "## Completed background tasks\n\
         The following tasks have finished:\n\n{notification_block}\n\
         {bg_context}\n"
    ));

    // 5. Simple, non-opinionated instruction — let the briefing do the work
    context_doc.push_str(
        "## Your task\n\
         You have background work that just completed. You are following up \
         proactively — the user has not sent a new message.\n\n\
         Read everything above: your identity, what you know, the recent \
         conversation, and the completed task results. Then decide:\n\n\
         - If the results contain information the user should hear, write a \
           brief, natural follow-up message. Be conversational — this is a \
           proactive update, not a formal report.\n\
         - If there is genuinely nothing worth sending (e.g. the result is \
           empty, meaningless, or the user truly already has this exact \
           information), respond with [SKIP].\n\n\
         Do NOT say \"notification\" or refer to yourself as a system. \
         Stay in character.\n"
    );

    let messages = vec![
        Message::system(context_doc),
        Message::user("What's the update?"),
    ];

    // 6. LLM call to formulate the message
    let response = llm.complete(&messages).await?;
    let reply_text = response.text.trim().to_string();

    // Guard: [SKIP] anywhere in the response means don't send
    if reply_text.is_empty() || reply_text.contains("[SKIP]") {
        tracing::info!(
            session_id = %session_id,
            count = notifications.len(),
            "notification delivery skipped (LLM decided nothing to send)"
        );
        return Ok(());
    }

    // 7. Send via the pipeline's outbound path
    let message = OutboundMessage::text(&reply_text);
    if let Err(e) = pipeline.send_outbound(&target, message).await {
        tracing::error!(
            channel = %channel,
            session_id = %session_id,
            error = %e,
            "proactive notification delivery failed"
        );
        return Err(e);
    }

    tracing::info!(
        session_id = %session_id,
        channel = %channel,
        count = notifications.len(),
        "proactive notifications delivered"
    );

    Ok(())
}

/// Follow DerivesFrom edges from notification nodes to their BackgroundTask
/// nodes and collect the full bodies. This gives the delivery LLM richer
/// context than the truncated notification summary alone.
async fn fetch_background_task_bodies(db: &Db, notifications: &[Node]) -> Vec<String> {
    let mut bodies = Vec::new();
    for notif in notifications {
        let nid = notif.id.clone();
        if let Ok(edges) = db
            .call(move |conn| queries::get_edges_from(conn, &nid))
            .await
        {
            for edge in edges {
                if edge.kind == EdgeKind::DerivesFrom {
                    let target_id = edge.dst.clone();
                    if let Ok(Some(node)) = db
                        .call(move |conn| queries::get_node(conn, &target_id))
                        .await
                    {
                        if node.kind == NodeKind::BackgroundTask {
                            if let Some(body) = &node.body {
                                bodies.push(body.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    bodies
}
