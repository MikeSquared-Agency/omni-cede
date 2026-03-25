//! Hooks — lifecycle interception points for the channel pipeline.
//!
//! Inspired by OpenClaw's `before_dispatch` / `after_tool_call` / `session:patch`
//! hooks. Any struct implementing [`ChannelHook`] can be registered with the
//! pipeline to intercept and transform messages at well-defined points:
//!
//! 1. **before_agent** — after identity/session resolution, before the agent runs.
//!    The hook receives the mutable envelope and can modify or reject it.
//! 2. **after_agent** — after the agent produces a reply. Can modify the reply.
//! 3. **before_send** — just before a message is sent on a channel.
//! 4. **after_send** — after successful delivery (for logging, metrics, etc.).
//!
//! Hooks are executed in registration order. A hook returning `Err` aborts
//! the pipeline at that stage (except `after_send`, which is best-effort).

use async_trait::async_trait;

use crate::error::Result;
use super::types::{InboundEnvelope, OutboundMessage, OutboundTarget};

/// A lifecycle hook that intercepts messages at defined pipeline stages.
///
/// All methods have default no-op implementations so you only need to
/// override the stages you care about.
#[async_trait]
pub trait ChannelHook: Send + Sync {
    /// Human-readable name for logging.
    fn name(&self) -> &str {
        "unnamed-hook"
    }

    /// Called after identity/session resolution and before the agent processes
    /// the message. Return `Err(CortexError::Pipeline(...))` to reject the message.
    ///
    /// Use cases: allowlist checks, rate limiting, command parsing.
    async fn before_agent(&self, _envelope: &mut InboundEnvelope) -> Result<()> {
        Ok(())
    }

    /// Called after the agent produces a reply. The hook receives the original
    /// envelope (immutable) and the reply text (mutable).
    ///
    /// Use cases: content filtering, response augmentation, analytics.
    async fn after_agent(
        &self,
        _envelope: &InboundEnvelope,
        _reply: &mut String,
    ) -> Result<()> {
        Ok(())
    }

    /// Called just before a message chunk is sent on a channel.
    ///
    /// Use cases: rate limiting, audit logging, message transformation.
    async fn before_send(
        &self,
        _target: &OutboundTarget,
        _message: &mut OutboundMessage,
    ) -> Result<()> {
        Ok(())
    }

    /// Called after a message chunk is successfully sent.
    ///
    /// Use cases: delivery tracking, metrics, follow-up scheduling.
    /// Errors from this hook are logged but do not fail the pipeline.
    async fn after_send(
        &self,
        _target: &OutboundTarget,
        _message: &OutboundMessage,
    ) -> Result<()> {
        Ok(())
    }
}

// ─── Built-in hooks ─────────────────────────────────────

/// A logging hook that traces every pipeline stage.
pub struct TracingHook;

#[async_trait]
impl ChannelHook for TracingHook {
    fn name(&self) -> &str {
        "tracing"
    }

    async fn before_agent(&self, envelope: &mut InboundEnvelope) -> Result<()> {
        tracing::info!(
            channel = %envelope.channel,
            sender = %envelope.external_id,
            text_len = envelope.text.len(),
            "before_agent"
        );
        Ok(())
    }

    async fn after_agent(
        &self,
        envelope: &InboundEnvelope,
        reply: &mut String,
    ) -> Result<()> {
        tracing::info!(
            channel = %envelope.channel,
            sender = %envelope.external_id,
            reply_len = reply.len(),
            "after_agent"
        );
        Ok(())
    }

    async fn before_send(
        &self,
        target: &OutboundTarget,
        message: &mut OutboundMessage,
    ) -> Result<()> {
        tracing::info!(
            channel = %target.channel,
            recipient = %target.external_id,
            text_len = message.text.len(),
            "before_send"
        );
        Ok(())
    }

    async fn after_send(
        &self,
        target: &OutboundTarget,
        _message: &OutboundMessage,
    ) -> Result<()> {
        tracing::info!(
            channel = %target.channel,
            recipient = %target.external_id,
            "after_send — delivered"
        );
        Ok(())
    }
}
