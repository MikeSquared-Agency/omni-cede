//! Pipeline — inbound and outbound message processing.
//!
//! The pipeline is the heart of the omnichannel system. Every inbound message
//! — regardless of which channel it came from — flows through the same stages:
//!
//! ```text
//! Inbound:
//!   1. Normalise (trim, detect /commands)
//!   2. Identity resolution (channel + external_id → internal user_id)
//!   3. Session resolution (user_id + channel → session graph node)
//!   4. Hooks: before_agent (allowlist, rate-limit, commands)
//!   5. Agent: run_turn(session_id, text) → reply
//!   6. Hooks: after_agent (content filtering, augmentation)
//!   7. Record turn
//!   8. Outbound delivery
//!
//! Outbound:
//!   1. Hooks: before_send
//!   2. Chunk (split long replies per channel limits)
//!   3. Channel.send(target, chunk)
//!   4. Hooks: after_send
//! ```

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::db::Db;
use crate::error::{CortexError, Result};
use crate::identity::{self, ChannelId};
use crate::session;

use super::hooks::ChannelHook;
use super::registry::ChannelRegistry;
use super::types::*;

/// The unified message pipeline.
pub struct Pipeline {
    /// Channel registry — used for outbound routing.
    registry: Arc<ChannelRegistry>,
    /// Registered hooks, executed in order.
    hooks: Vec<Arc<dyn ChannelHook>>,
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(registry: Arc<ChannelRegistry>) -> Self {
        Self {
            registry,
            hooks: Vec::new(),
        }
    }

    /// Register a lifecycle hook. Hooks execute in registration order.
    pub fn add_hook(&mut self, hook: Arc<dyn ChannelHook>) {
        tracing::info!(hook = hook.name(), "pipeline hook registered");
        self.hooks.push(hook);
    }

    /// Process a single inbound message through the full pipeline.
    ///
    /// This is the core method. It performs identity resolution, session
    /// management, agent execution, and outbound delivery.
    pub async fn process(
        &self,
        mut envelope: InboundEnvelope,
        db: &Db,
        agent: &Agent,
    ) -> Result<PipelineResult> {
        // ── 1. Normalise ────────────────────────────────
        normalise(&mut envelope);

        // ── 2. Identity resolution ──────────────────────
        let channel_id = ChannelId::new(&envelope.channel, &envelope.external_id);
        let user = identity::resolve_user(db, channel_id).await.map_err(|e| {
            CortexError::Pipeline(format!("Identity resolution failed: {e}"))
        })?;

        // ── 3. Session resolution ───────────────────────
        let managed = session::get_or_create(db, &user.id, &envelope.channel)
            .await
            .map_err(|e| {
                CortexError::Pipeline(format!("Session resolution failed: {e}"))
            })?;

        // ── 4. Hooks: before_agent ──────────────────────
        for hook in &self.hooks {
            if let Err(e) = hook.before_agent(&mut envelope).await {
                tracing::warn!(
                    hook = hook.name(),
                    error = %e,
                    "before_agent hook rejected message"
                );
                return Err(e);
            }
        }

        // ── 5. Agent ────────────────────────────────────
        let mut reply = agent
            .run_turn(&managed.node_id, &envelope.text)
            .await
            .map_err(|e| CortexError::Pipeline(format!("Agent error: {e}")))?;

        // ── 6. Hooks: after_agent ───────────────────────
        for hook in &self.hooks {
            if let Err(e) = hook.after_agent(&envelope, &mut reply).await {
                tracing::warn!(
                    hook = hook.name(),
                    error = %e,
                    "after_agent hook error (continuing)"
                );
                // after_agent errors are non-fatal — we still have a reply
            }
        }

        // ── 7. Record turn ─────────────────────────────
        let _ = session::record_turn(db, &managed.node_id).await;

        // ── 8. Outbound delivery ────────────────────────
        let target = OutboundTarget::from_envelope(&envelope);
        let message = OutboundMessage::text(&reply);

        if let Err(e) = self.send_outbound(&target, message).await {
            tracing::error!(
                channel = %target.channel,
                error = %e,
                "outbound delivery failed"
            );
            // Don't fail the whole pipeline — the reply was produced, just delivery failed
        }

        Ok(PipelineResult {
            reply,
            user_id: user.id,
            session_id: managed.node_id,
        })
    }

    /// Process a message synchronously (no outbound delivery).
    ///
    /// Used by the HTTP API where the caller handles the response directly.
    pub async fn process_sync(
        &self,
        mut envelope: InboundEnvelope,
        db: &Db,
        agent: &Agent,
    ) -> Result<PipelineResult> {
        normalise(&mut envelope);

        let channel_id = ChannelId::new(&envelope.channel, &envelope.external_id);
        let user = identity::resolve_user(db, channel_id).await.map_err(|e| {
            CortexError::Pipeline(format!("Identity resolution failed: {e}"))
        })?;

        let managed = session::get_or_create(db, &user.id, &envelope.channel)
            .await
            .map_err(|e| {
                CortexError::Pipeline(format!("Session resolution failed: {e}"))
            })?;

        for hook in &self.hooks {
            hook.before_agent(&mut envelope).await?;
        }

        let mut reply = agent
            .run_turn(&managed.node_id, &envelope.text)
            .await
            .map_err(|e| CortexError::Pipeline(format!("Agent error: {e}")))?;

        for hook in &self.hooks {
            let _ = hook.after_agent(&envelope, &mut reply).await;
        }

        let _ = session::record_turn(db, &managed.node_id).await;

        Ok(PipelineResult {
            reply,
            user_id: user.id,
            session_id: managed.node_id,
        })
    }

    /// Send an outbound message, running hooks and applying chunking.
    pub async fn send_outbound(
        &self,
        target: &OutboundTarget,
        mut message: OutboundMessage,
    ) -> Result<()> {
        // ── before_send hooks ───────────────────────────
        for hook in &self.hooks {
            if let Err(e) = hook.before_send(target, &mut message).await {
                tracing::warn!(
                    hook = hook.name(),
                    error = %e,
                    "before_send hook rejected"
                );
                return Err(e);
            }
        }

        // ── Chunking ───────────────────────────────────
        let channel = self.registry.get(&target.channel).await;
        let max_len = channel
            .as_ref()
            .map(|ch| ch.max_message_length())
            .unwrap_or(4096);

        let chunks = chunk_text(&message.text, max_len);

        // ── Send each chunk ─────────────────────────────
        if let Some(ch) = channel {
            for (i, chunk) in chunks.iter().enumerate() {
                let chunk_msg = OutboundMessage {
                    text: chunk.clone(),
                    media: if i == 0 { message.media.clone() } else { None },
                    metadata: message.metadata.clone(),
                };

                ch.send(target, chunk_msg).await?;

                // Send typing between chunks (except the last)
                if i < chunks.len() - 1 {
                    let _ = ch.send_typing(target).await;
                }
            }
        } else {
            tracing::warn!(
                channel = %target.channel,
                "no channel adapter found for outbound — skipping delivery"
            );
        }

        // ── after_send hooks ────────────────────────────
        for hook in &self.hooks {
            if let Err(e) = hook.after_send(target, &message).await {
                tracing::warn!(
                    hook = hook.name(),
                    error = %e,
                    "after_send hook error (ignored)"
                );
            }
        }

        Ok(())
    }

    /// Start the background inbound processing loop.
    ///
    /// Reads envelopes from the channel registry's inbound receiver and
    /// processes each one through the pipeline. Runs until the receiver is
    /// closed (i.e. all channels have stopped).
    pub async fn run_inbound_loop(
        self: Arc<Self>,
        mut rx: mpsc::Receiver<InboundEnvelope>,
        db: Db,
        agent: Arc<Agent>,
    ) {
        tracing::info!("pipeline inbound loop started");
        while let Some(envelope) = rx.recv().await {
            let pipeline = Arc::clone(&self);
            let db = db.clone();
            let agent = Arc::clone(&agent);

            // Process each message in its own task so one slow turn
            // doesn't block the rest.
            tokio::spawn(async move {
                match pipeline.process(envelope, &db, &agent).await {
                    Ok(result) => {
                        tracing::debug!(
                            user_id = %result.user_id,
                            session_id = %result.session_id,
                            reply_len = result.reply.len(),
                            "pipeline turn complete"
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "pipeline processing error");
                    }
                }
            });
        }
        tracing::info!("pipeline inbound loop ended (all channels closed)");
    }
}

// ─── Helpers ────────────────────────────────────────────

/// Normalise an inbound envelope: trim whitespace, collapse newlines.
fn normalise(envelope: &mut InboundEnvelope) {
    envelope.text = envelope.text.trim().to_string();
}

/// Split text into chunks respecting the given max length.
///
/// Tries to split on double-newlines first, then single newlines, then spaces.
/// Falls back to hard character splits as a last resort.
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    if text.len() <= max_len {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        // Try to find a good split point
        let slice = &remaining[..max_len];
        let split_at = slice
            .rfind("\n\n")
            .or_else(|| slice.rfind('\n'))
            .or_else(|| slice.rfind(' '))
            .unwrap_or(max_len);

        let (chunk, rest) = remaining.split_at(split_at);
        chunks.push(chunk.trim_end().to_string());
        remaining = rest.trim_start();
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_message() {
        let chunks = chunk_text("hello", 100);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn test_chunk_on_newline() {
        let text = "line one\n\nline two\n\nline three";
        let chunks = chunk_text(text, 15);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "line one");
        assert_eq!(chunks[1], "line two");
        assert_eq!(chunks[2], "line three");
    }

    #[test]
    fn test_chunk_on_space() {
        let text = "word1 word2 word3 word4";
        let chunks = chunk_text(text, 12);
        assert!(chunks.len() >= 2);
    }
}
