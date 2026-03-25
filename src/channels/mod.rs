//! Channel system — omnichannel messaging adapters.
//!
//! Every messaging platform (Telegram, Discord, Slack, WhatsApp, WebChat,
//! generic webhook) implements the [`Channel`] trait. The [`ChannelRegistry`]
//! manages their lifecycle, and the [`Pipeline`] routes messages through a
//! normalised inbound → agent → outbound flow with [`ChannelHook`] interception.
//!
//! # Architecture
//!
//! ```text
//!   Platform → Adapter → InboundEnvelope → Pipeline → Agent → OutboundMessage → Adapter → Platform
//! ```
//!
//! All channels share the same pipeline. Business logic stays in the pipeline
//! and agent; the Channel trait only handles platform wire protocol.

pub mod types;
pub mod hooks;
pub mod registry;
pub mod pipeline;
pub mod webhook;
pub mod telegram;
pub mod discord;
pub mod webchat;

// Re-export the public surface.
pub use types::*;
pub use hooks::{ChannelHook, TracingHook};
pub use registry::ChannelRegistry;
pub use pipeline::Pipeline;

use async_trait::async_trait;

use crate::error::Result;

// ─── Channel trait ──────────────────────────────────────

/// The core abstraction: every messaging platform adapter implements this.
///
/// A channel knows how to:
/// - Start listening for inbound messages (push them into the pipeline).
/// - Send outbound messages back to users on its platform.
/// - Report its health status.
///
/// Optional capabilities (typing indicators, message editing, media) have
/// default no-op implementations so simple channels needn't bother.
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    /// Unique, lowercase channel identifier: `"telegram"`, `"discord"`, etc.
    fn id(&self) -> &str;

    /// Human-readable display name.
    fn display_name(&self) -> &str;

    /// Start the adapter.
    ///
    /// The implementation should spawn any long-running tasks (polling loops,
    /// WebSocket connections) and push inbound messages into
    /// `ctx.inbound_tx`. It must respect `ctx.shutdown` to exit cleanly.
    async fn start(&self, ctx: ChannelContext) -> Result<()>;

    /// Stop the adapter gracefully. Called before process exit.
    async fn stop(&self) -> Result<()>;

    /// Send a message to a user on this platform.
    async fn send(&self, target: &OutboundTarget, message: OutboundMessage) -> Result<()>;

    /// Report current health.
    async fn health(&self) -> ChannelHealth;

    // ── Optional capabilities ───────────────────────────

    /// Maximum text length this channel supports per message.
    /// The outbound pipeline uses this for chunking.
    fn max_message_length(&self) -> usize {
        types::max_message_length(self.id())
    }

    /// Send a "typing…" indicator.
    async fn send_typing(&self, _target: &OutboundTarget) -> Result<()> {
        Ok(()) // no-op by default
    }

    /// Edit an already-sent message (for streaming responses).
    async fn edit_message(&self, _message_id: &str, _new_text: &str) -> Result<()> {
        Err(crate::error::CortexError::Unsupported(
            "edit_message not supported on this channel".into(),
        ))
    }

    /// Whether this channel supports media attachments.
    fn supports_media(&self) -> bool {
        false
    }

    /// Send a media attachment.
    async fn send_media(
        &self,
        _target: &OutboundTarget,
        _media: MediaPayload,
    ) -> Result<()> {
        Err(crate::error::CortexError::Unsupported(
            "send_media not supported on this channel".into(),
        ))
    }
}
