//! Event-driven notification delivery via pipeline injection.
//!
//! When background tool execution or a cron job completes, the producer
//! creates a `Notification` graph node and fires a [`NotifEvent`] on a
//! tokio mpsc channel. This module awaits that channel and injects a
//! synthetic inbound message into the main pipeline — the same pipeline
//! that handles real user messages.
//!
//! The agent's `run_turn` already has an "Updates while you were away"
//! section that picks up pending notification nodes and weaves them into
//! its response naturally. By routing notifications through the pipeline
//! we get **one agent, one voice, one context** — no separate LLM call.
//!
//! # Flow
//!
//! ```text
//!   NotifEvent received on mpsc channel
//!     → query session routing (user_id, channel)
//!     → resolve external_id for that (user, channel)
//!     → build a synthetic InboundEnvelope
//!     → send it on the pipeline's inbound_tx
//!     → pipeline.process() → agent.run_turn() picks up pending
//!       notification nodes via "Updates while you were away"
//!     → outbound delivery happens through the normal pipeline path
//! ```

use tokio::sync::mpsc;

use crate::channels::types::InboundEnvelope;
use crate::db::Db;
use crate::db::queries;
use crate::identity;

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
/// Awaits [`NotifEvent`]s on the mpsc channel. When one arrives, resolves
/// the session's routing and injects a synthetic message into the pipeline
/// so the main agent handles it.
pub async fn run(
    db: Db,
    inbound_tx: mpsc::Sender<InboundEnvelope>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    mut rx: NotifRx,
) {
    tracing::info!("notification delivery loop started (pipeline injection)");

    loop {
        tokio::select! {
            event = rx.recv() => {
                let Some(event) = event else {
                    tracing::info!("notification channel closed — shutting down");
                    break;
                };
                if let Err(e) = inject_for_event(&db, &inbound_tx, &event.session_id).await {
                    tracing::warn!(session_id = %event.session_id, error = %e, "notification injection failed");
                }
            }
            _ = shutdown_rx.changed() => {
                tracing::info!("notification delivery loop shutting down");
                break;
            }
        }
    }
}

/// Handle a single event: look up routing, build a synthetic envelope, inject.
async fn inject_for_event(
    db: &Db,
    inbound_tx: &mpsc::Sender<InboundEnvelope>,
    session_id: &str,
) -> crate::error::Result<()> {
    // 1. Check there are actually pending notifications (may already be claimed)
    let sid = session_id.to_string();
    let pending = db
        .call(move |conn| queries::get_pending_notification_nodes(conn, &sid))
        .await?;
    if pending.is_empty() {
        tracing::debug!(session_id = %session_id, "no pending notifications — already claimed");
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

    // 3. Resolve the external_id for this (user, channel) pair
    let uid = user_id.clone();
    let ch = channel.clone();
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
                "no external_id found — cannot inject notification"
            );
            return Ok(());
        }
    };

    // 4. Build a synthetic InboundEnvelope
    //    The text is a minimal trigger — the agent's "Updates while you were
    //    away" section will pick up the actual notification content.
    let envelope = InboundEnvelope::new(
        &channel,
        &external_id,
        "[background tasks completed]",
    );

    // 5. Inject into the pipeline's inbound channel
    if let Err(e) = inbound_tx.send(envelope).await {
        tracing::error!(
            session_id = %session_id,
            error = %e,
            "failed to inject notification into pipeline"
        );
        return Err(crate::error::CortexError::Pipeline(
            format!("inbound_tx send failed: {e}"),
        ));
    }

    tracing::info!(
        session_id = %session_id,
        channel = %channel,
        count = pending.len(),
        "notification injected into pipeline"
    );

    Ok(())
}
