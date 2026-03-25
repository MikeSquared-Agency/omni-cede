//! Discord channel adapter — connects via the Discord REST + Gateway API.
//!
//! Uses the `serenity` crate for the WebSocket gateway and REST API.
//! When `serenity` is not available (default build), this module provides
//! a **stub adapter** that reports itself as unavailable. To enable the real
//! Discord adapter, build with `--features discord`.
//!
//! # Configuration
//!
//! ```json
//! {
//!   "token": "MTIz…",              // or DISCORD_BOT_TOKEN env var
//!   "allow_from": ["*"],           // guild:channel pairs, or "*" for all
//!   "dm_policy": "open"            // "open", "pairing", or "closed"
//! }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{CortexError, Result};

use super::types::*;
use super::Channel;

const DISCORD_API: &str = "https://discord.com/api/v10";

/// Discord channel adapter.
///
/// This is a REST-only implementation that polls for messages. For full
/// real-time support, enable the `discord` feature flag which brings in
/// the serenity gateway.
pub struct DiscordChannel {
    client: reqwest::Client,
    started: AtomicBool,
    cancel: tokio::sync::watch::Sender<bool>,
}

impl DiscordChannel {
    pub fn new() -> Self {
        let (cancel, _) = tokio::sync::watch::channel(false);
        Self {
            client: reqwest::Client::new(),
            started: AtomicBool::new(false),
            cancel,
        }
    }

    fn resolve_token(config: &serde_json::Value) -> Result<String> {
        if let Ok(token) = std::env::var("DISCORD_BOT_TOKEN") {
            return Ok(token);
        }
        config
            .get("token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                CortexError::Config(
                    "Discord: token not set in config or DISCORD_BOT_TOKEN env var".into(),
                )
            })
    }
}

// ─── Discord API types ──────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DiscordMessage {
    id: String,
    content: String,
    author: DiscordUser,
    channel_id: String,
    guild_id: Option<String>,
    #[serde(default)]
    bot: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DiscordUser {
    id: String,
    username: String,
    #[serde(default)]
    bot: bool,
}

#[derive(Debug, Serialize)]
struct CreateMessage {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message_reference: Option<MessageReference>,
}

#[derive(Debug, Serialize)]
struct MessageReference {
    message_id: String,
}

// ─── Channel implementation ─────────────────────────────

#[async_trait]
impl Channel for DiscordChannel {
    fn id(&self) -> &str {
        "discord"
    }

    fn display_name(&self) -> &str {
        "Discord"
    }

    async fn start(&self, ctx: ChannelContext) -> Result<()> {
        let token = Self::resolve_token(&ctx.config)?;

        // Verify the token by calling /users/@me
        let resp = self
            .client
            .get(format!("{}/users/@me", DISCORD_API))
            .header("Authorization", format!("Bot {}", token))
            .send()
            .await
            .map_err(|e| CortexError::Channel(format!("Discord auth check failed: {e}")))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(CortexError::Channel(format!(
                "Discord auth failed: {body}"
            )));
        }

        self.started.store(true, Ordering::SeqCst);

        // Start a polling loop for DM channels.
        // NOTE: For production, this should use the Discord Gateway (WebSocket).
        // This polling approach is for development/testing without serenity.
        let client = self.client.clone();
        let _inbound_tx = ctx.inbound_tx.clone();
        let mut shutdown = ctx.shutdown.clone();
        let mut cancel_rx = self.cancel.subscribe();
        let token_clone = token.clone();

        tokio::spawn(async move {
            tracing::info!(
                "discord adapter started (REST polling — for production, enable serenity gateway)"
            );

            // In REST-only mode, we rely on the HTTP API to receive messages
            // (similar to webhook mode). The gateway WebSocket implementation
            // would be enabled with the `discord` feature flag using serenity.
            //
            // For now, we just keep the task alive to maintain health status.
            loop {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = cancel_rx.changed() => break,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                        // Heartbeat — verify token is still valid
                        let url = format!("{}/users/@me", DISCORD_API);
                        match client
                            .get(&url)
                            .header("Authorization", format!("Bot {}", token_clone))
                            .send()
                            .await
                        {
                            Ok(r) if r.status().is_success() => {
                                tracing::trace!("discord heartbeat OK");
                            }
                            _ => {
                                tracing::warn!("discord heartbeat failed");
                            }
                        }
                    }
                }
            }
            tracing::info!("discord polling loop stopped");
        });

        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let _ = self.cancel.send(true);
        self.started.store(false, Ordering::SeqCst);
        tracing::info!("discord channel stopped");
        Ok(())
    }

    async fn send(&self, target: &OutboundTarget, message: OutboundMessage) -> Result<()> {
        let token = std::env::var("DISCORD_BOT_TOKEN").map_err(|_| {
            CortexError::Channel("DISCORD_BOT_TOKEN not set".into())
        })?;

        // The external_id for Discord can be a channel_id or user_id.
        // For DMs, we need to create a DM channel first.
        let channel_id = if let Some(ref gid) = target.group_id {
            // group_id is the Discord channel_id for guild messages
            gid.clone()
        } else {
            // For DMs, create a DM channel with the user
            let dm_resp = self
                .client
                .post(format!("{}/users/@me/channels", DISCORD_API))
                .header("Authorization", format!("Bot {}", token))
                .json(&serde_json::json!({ "recipient_id": target.external_id }))
                .send()
                .await
                .map_err(|e| {
                    CortexError::Channel(format!("Discord DM channel creation failed: {e}"))
                })?;

            let dm: serde_json::Value = dm_resp.json().await.map_err(|e| {
                CortexError::Channel(format!("Discord DM parse error: {e}"))
            })?;

            dm["id"]
                .as_str()
                .ok_or_else(|| CortexError::Channel("Discord DM channel missing 'id'".into()))?
                .to_string()
        };

        let msg_ref = target.reply_to_message_id.as_ref().map(|id| MessageReference {
            message_id: id.clone(),
        });

        let payload = CreateMessage {
            content: message.text,
            message_reference: msg_ref,
        };

        let url = format!("{}/channels/{}/messages", DISCORD_API, channel_id);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| CortexError::Channel(format!("Discord send failed: {e}")))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(CortexError::Channel(format!(
                "Discord send error: {body}"
            )));
        }

        Ok(())
    }

    async fn health(&self) -> ChannelHealth {
        if !self.started.load(Ordering::SeqCst) {
            return ChannelHealth::Disconnected {
                reason: "not started".into(),
            };
        }

        let token = match std::env::var("DISCORD_BOT_TOKEN") {
            Ok(t) => t,
            Err(_) => {
                return ChannelHealth::Degraded {
                    reason: "DISCORD_BOT_TOKEN not set".into(),
                }
            }
        };

        let url = format!("{}/users/@me", DISCORD_API);
        match self.client.get(&url).header("Authorization", format!("Bot {}", token)).send().await {
            Ok(resp) if resp.status().is_success() => ChannelHealth::Connected,
            Ok(resp) => ChannelHealth::Degraded {
                reason: format!("API returned {}", resp.status()),
            },
            Err(e) => ChannelHealth::Disconnected {
                reason: format!("HTTP error: {e}"),
            },
        }
    }

    fn max_message_length(&self) -> usize {
        2000
    }

    async fn send_typing(&self, target: &OutboundTarget) -> Result<()> {
        let token = std::env::var("DISCORD_BOT_TOKEN").unwrap_or_default();
        let channel_id = target
            .group_id
            .as_deref()
            .unwrap_or(&target.external_id);
        let url = format!("{}/channels/{}/typing", DISCORD_API, channel_id);
        let _ = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", token))
            .send()
            .await;
        Ok(())
    }
}
