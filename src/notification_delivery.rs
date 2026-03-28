//! Proactive notification delivery — timer-based background loop.
//!
//! When background tool execution completes, the result is stored as a
//! `Notification` graph node linked to the user's session. This module runs
//! a periodic loop that detects undelivered notifications, formulates a
//! natural message via a brief LLM call, and pushes it proactively to the
//! user's channel — so the user doesn't have to send another message to
//! see the results.
//!
//! # Flow
//!
//! ```text
//!   tick (every N seconds)
//!     → query sessions with pending notifications
//!     → for each: resolve outbound routing (channel + external_id)
//!     → brief LLM call to formulate a natural update message
//!     → Pipeline::send_outbound() to push it to the user
//!     → touch_nodes() to mark notifications as delivered
//! ```

use std::sync::Arc;

use crate::channels::Pipeline;
use crate::channels::types::*;
use crate::db::Db;
use crate::db::queries;
use crate::identity;
use crate::llm::LlmClient;
use crate::memory;
use crate::types::*;

/// Run the notification delivery loop until shutdown is signalled.
///
/// Checks for pending notification nodes every `interval_secs` seconds.
/// When found, resolves outbound routing, runs a brief LLM call to
/// produce a natural message, and sends it via the pipeline.
pub async fn run(
    db: Db,
    pipeline: Arc<Pipeline>,
    llm: Arc<dyn LlmClient>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    interval_secs: u64,
) {
    let interval = std::time::Duration::from_secs(interval_secs);
    let mut ticker = tokio::time::interval(interval);
    ticker.tick().await; // skip the first immediate tick

    tracing::info!(interval_secs, "notification delivery loop started");

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Err(e) = deliver_pending(&db, &pipeline, &llm).await {
                    tracing::warn!(error = %e, "notification delivery tick failed");
                }
            }
            _ = shutdown_rx.changed() => {
                tracing::info!("notification delivery loop shutting down");
                break;
            }
        }
    }
}

/// One tick: find all sessions with pending notifications and deliver them.
async fn deliver_pending(
    db: &Db,
    pipeline: &Arc<Pipeline>,
    llm: &Arc<dyn LlmClient>,
) -> crate::error::Result<()> {
    // Query all sessions that have undelivered notification nodes.
    let sessions_with_notifs = db
        .call(|conn| queries::get_sessions_with_pending_notifications(conn))
        .await?;

    if sessions_with_notifs.is_empty() {
        return Ok(());
    }

    for (user_id, channel, session_id, notifications) in sessions_with_notifs {
        if let Err(e) = deliver_for_session(
            db, pipeline, llm,
            &user_id, &channel, &session_id, &notifications,
        ).await {
            tracing::warn!(
                user_id = %user_id,
                channel = %channel,
                session_id = %session_id,
                error = %e,
                "failed to deliver notifications for session"
            );
        }
    }

    Ok(())
}

/// Deliver all pending notifications for a single session.
async fn deliver_for_session(
    db: &Db,
    pipeline: &Arc<Pipeline>,
    llm: &Arc<dyn LlmClient>,
    user_id: &str,
    channel: &str,
    session_id: &str,
    notifications: &[Node],
) -> crate::error::Result<()> {
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

    // 2. Build a brief prompt with the notification summaries + persona
    //    Pull Soul + Belief nodes so the LLM reply stays in character.
    let persona = db
        .call(|conn| {
            let mut parts = Vec::new();
            let souls = queries::get_nodes_by_kind(conn, NodeKind::Soul)?;
            for n in &souls {
                if let Some(ref b) = n.body {
                    parts.push(b.clone());
                }
            }
            let beliefs = queries::get_nodes_by_kind(conn, NodeKind::Belief)?;
            for n in &beliefs {
                if let Some(ref b) = n.body {
                    parts.push(format!("Belief: {}", b));
                }
            }
            Ok(parts.join("\n"))
        })
        .await
        .unwrap_or_default();

    let mut notification_block = String::new();
    let mut delivered_ids: Vec<String> = Vec::new();
    for node in notifications {
        let rel = memory::relative_time(node.created_at);
        let body = node.body.as_deref().unwrap_or(&node.title);
        notification_block.push_str(&format!("- ({}) {}\n", rel, body));
        delivered_ids.push(node.id.clone());
    }

    let persona_section = if persona.is_empty() {
        String::new()
    } else {
        format!("## Your identity\n{}\n\n", persona)
    };

    let system_prompt = format!(
        "{persona_section}\
         You are following up on background work you kicked off earlier. \
         The following tasks have completed:\n\n{notification_block}\n\
         Write a brief, natural message to let the user know what happened. \
         Be conversational and concise — this is a proactive update, not a \
         formal report. If something failed, mention it clearly but calmly. \
         Do NOT say \"notification\" or refer to yourself as a system. \
         Stay in character.",
    );

    let messages = vec![
        Message::system(system_prompt),
        Message::user("What's the update?"),
    ];

    // 3. Brief LLM call to formulate the message
    let response = llm.complete(&messages).await?;
    let reply_text = response.text;

    if reply_text.is_empty() {
        tracing::warn!(session_id = %session_id, "LLM returned empty notification delivery text");
        return Ok(());
    }

    // 4. Send via the pipeline's outbound path
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

    // 5. Mark notifications as delivered (bump access_count from 0)
    db.call(move |conn| queries::touch_nodes(conn, &delivered_ids))
        .await?;

    Ok(())
}
