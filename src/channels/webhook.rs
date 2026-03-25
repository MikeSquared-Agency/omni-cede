//! Webhook channel — generic inbound/outbound for any platform that POSTs JSON.
//!
//! This is the simplest channel adapter. It doesn't poll or hold connections —
//! it receives messages via the HTTP API (`POST /v1/channels/webhook/inbound`)
//! and optionally delivers replies by POSTing to a callback URL.
//!
//! Any system can integrate with omni-cede without a dedicated adapter by
//! using the webhook channel.

use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;

use crate::error::{CortexError, Result};

use super::types::*;
use super::Channel;

/// A generic webhook channel adapter.
///
/// Inbound: messages arrive via the HTTP API (the API handler creates
/// `InboundEnvelope` and pushes it into the pipeline).
///
/// Outbound: if the inbound message included a `callback_url`, the reply
/// is POSTed there. Otherwise the reply is returned synchronously via the
/// HTTP response.
pub struct WebhookChannel {
    /// HTTP client for callback delivery.
    client: reqwest::Client,
    /// Whether the channel is "started" (always true once start() is called).
    started: AtomicBool,
}

impl WebhookChannel {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            started: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Channel for WebhookChannel {
    fn id(&self) -> &str {
        "webhook"
    }

    fn display_name(&self) -> &str {
        "Generic Webhook"
    }

    async fn start(&self, _ctx: ChannelContext) -> Result<()> {
        // Webhook channel is passive — it doesn't poll. Messages come in via
        // the HTTP API. We just mark ourselves as started.
        self.started.store(true, Ordering::SeqCst);
        tracing::info!("webhook channel started (passive — receives via HTTP API)");
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.started.store(false, Ordering::SeqCst);
        tracing::info!("webhook channel stopped");
        Ok(())
    }

    async fn send(&self, target: &OutboundTarget, message: OutboundMessage) -> Result<()> {
        // If there's a callback URL, POST the reply there
        if let Some(ref url) = target.callback_url {
            let payload = serde_json::json!({
                "channel": target.channel,
                "external_id": target.external_id,
                "text": message.text,
                "metadata": message.metadata,
            });

            let resp = self
                .client
                .post(url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| {
                    CortexError::Channel(format!("Webhook callback failed: {e}"))
                })?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(CortexError::Channel(format!(
                    "Webhook callback returned {status}: {body}"
                )));
            }

            tracing::debug!(url = %url, "webhook callback delivered");
        } else {
            // No callback URL — reply was returned synchronously via the HTTP response.
            // Nothing to do here.
            tracing::trace!("webhook outbound: no callback_url (reply returned synchronously)");
        }

        Ok(())
    }

    async fn health(&self) -> ChannelHealth {
        if self.started.load(Ordering::SeqCst) {
            ChannelHealth::Connected
        } else {
            ChannelHealth::Disconnected {
                reason: "not started".into(),
            }
        }
    }

    fn max_message_length(&self) -> usize {
        // Webhooks have no inherent limit — use a generous default
        100_000
    }
}
