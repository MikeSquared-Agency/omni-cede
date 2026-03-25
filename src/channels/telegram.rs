//! Telegram channel adapter — connects to the Telegram Bot API.
//!
//! Supports two modes:
//! - **Polling** (default): calls `getUpdates` in a loop. Simple, no public URL needed.
//! - **Webhook**: Telegram POSTs updates to our `/v1/channels/telegram/webhook`.
//!   Requires a public HTTPS URL.
//!
//! # Configuration
//!
//! ```json
//! {
//!   "bot_token": "123456:ABCDEF…",
//!   "mode": "polling",               // "polling" or "webhook"
//!   "webhook_url": "https://…",      // required if mode = "webhook"
//!   "allow_from": ["*"],             // Telegram user IDs, or "*" for all
//!   "polling_timeout": 30            // long-poll timeout in seconds
//! }
//! ```
//!
//! The `bot_token` can also be set via the `TELEGRAM_BOT_TOKEN` env var
//! (env var takes precedence).

use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{CortexError, Result};

use super::types::*;
use super::Channel;

const BASE_URL: &str = "https://api.telegram.org/bot";

/// Telegram channel adapter.
pub struct TelegramChannel {
    client: reqwest::Client,
    started: AtomicBool,
    /// Stored after start() to allow stop() to signal shutdown.
    cancel: tokio::sync::watch::Sender<bool>,
}

impl TelegramChannel {
    pub fn new() -> Self {
        let (cancel, _) = tokio::sync::watch::channel(false);
        Self {
            client: reqwest::Client::new(),
            started: AtomicBool::new(false),
            cancel,
        }
    }

    fn resolve_token(config: &serde_json::Value) -> Result<String> {
        // Env var takes precedence
        if let Ok(token) = std::env::var("TELEGRAM_BOT_TOKEN") {
            return Ok(token);
        }
        config
            .get("bot_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                CortexError::Config(
                    "Telegram: bot_token not set in config or TELEGRAM_BOT_TOKEN env var"
                        .into(),
                )
            })
    }
}

// ─── Telegram API types ─────────────────────────────────

#[derive(Debug, Deserialize)]
struct TgResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TgUpdate {
    update_id: i64,
    message: Option<TgMessage>,
}

#[derive(Debug, Deserialize)]
struct TgMessage {
    message_id: i64,
    from: Option<TgUser>,
    chat: TgChat,
    text: Option<String>,
    // TODO: photo, document, voice, etc.
}

#[derive(Debug, Deserialize)]
struct TgUser {
    id: i64,
    first_name: String,
    last_name: Option<String>,
    #[allow(dead_code)]
    username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TgChat {
    id: i64,
    #[serde(rename = "type")]
    chat_type: String,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest {
    chat_id: i64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply_to_message_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TgSentMessage {
    message_id: i64,
}

// ─── Channel implementation ─────────────────────────────

#[async_trait]
impl Channel for TelegramChannel {
    fn id(&self) -> &str {
        "telegram"
    }

    fn display_name(&self) -> &str {
        "Telegram"
    }

    async fn start(&self, ctx: ChannelContext) -> Result<()> {
        let token = Self::resolve_token(&ctx.config)?;
        let mode = ctx
            .config
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("polling");

        let polling_timeout = ctx
            .config
            .get("polling_timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(30);

        match mode {
            "polling" => {
                self.started.store(true, Ordering::SeqCst);
                let client = self.client.clone();
                let inbound_tx = ctx.inbound_tx.clone();
                let mut shutdown = ctx.shutdown.clone();
                let mut cancel_rx = self.cancel.subscribe();

                tokio::spawn(async move {
                    let mut offset: i64 = 0;
                    tracing::info!("telegram polling loop started");

                    loop {
                        // Check shutdown signals
                        if *shutdown.borrow() || *cancel_rx.borrow() {
                            tracing::info!("telegram polling loop shutting down");
                            break;
                        }

                        let url = format!(
                            "{}{}/getUpdates?offset={}&timeout={}&allowed_updates=[\"message\"]",
                            BASE_URL, token, offset, polling_timeout
                        );

                        let result = tokio::select! {
                            r = client.get(&url).send() => r,
                            _ = shutdown.changed() => break,
                            _ = cancel_rx.changed() => break,
                        };

                        match result {
                            Ok(resp) => {
                                match resp.json::<TgResponse<Vec<TgUpdate>>>().await {
                                    Ok(tg_resp) if tg_resp.ok => {
                                        if let Some(updates) = tg_resp.result {
                                            for update in updates {
                                                offset = update.update_id + 1;
                                                if let Some(msg) = update.message {
                                                    if let Some(text) = msg.text {
                                                        let sender_id = msg
                                                            .from
                                                            .as_ref()
                                                            .map(|u| u.id.to_string())
                                                            .unwrap_or_else(|| {
                                                                msg.chat.id.to_string()
                                                            });
                                                        let sender_name = msg.from.as_ref().map(
                                                            |u| {
                                                                let mut name = u.first_name.clone();
                                                                if let Some(ref last) = u.last_name
                                                                {
                                                                    name.push(' ');
                                                                    name.push_str(last);
                                                                }
                                                                name
                                                            },
                                                        );

                                                        let group_id =
                                                            if msg.chat.chat_type != "private" {
                                                                Some(msg.chat.id.to_string())
                                                            } else {
                                                                None
                                                            };

                                                        let envelope = InboundEnvelope {
                                                            channel: "telegram".into(),
                                                            external_id: sender_id,
                                                            sender_name,
                                                            text,
                                                            media: None,
                                                            reply_to: None,
                                                            group_id,
                                                            callback_url: None,
                                                            raw: serde_json::json!({
                                                                "chat_id": msg.chat.id,
                                                                "message_id": msg.message_id,
                                                            }),
                                                            timestamp: now_unix(),
                                                        };

                                                        if inbound_tx.send(envelope).await.is_err()
                                                        {
                                                            tracing::error!(
                                                                "telegram: inbound channel closed"
                                                            );
                                                            return;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Ok(tg_resp) => {
                                        tracing::warn!(
                                            desc = ?tg_resp.description,
                                            "telegram API error"
                                        );
                                        tokio::time::sleep(
                                            std::time::Duration::from_secs(5),
                                        )
                                        .await;
                                    }
                                    Err(e) => {
                                        tracing::warn!(error = %e, "telegram parse error");
                                        tokio::time::sleep(
                                            std::time::Duration::from_secs(5),
                                        )
                                        .await;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "telegram HTTP error");
                                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            }
                        }
                    }
                });
            }
            "webhook" => {
                // Webhook mode: Telegram will POST to our endpoint.
                // We need to register the webhook URL with Telegram.
                let webhook_url = ctx
                    .config
                    .get("webhook_url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        CortexError::Config(
                            "Telegram webhook mode requires 'webhook_url' in config".into(),
                        )
                    })?;

                let url = format!(
                    "{}{}/setWebhook?url={}/v1/channels/telegram/webhook",
                    BASE_URL, token, webhook_url
                );
                let resp = self.client.get(&url).send().await.map_err(|e| {
                    CortexError::Channel(format!("Failed to set Telegram webhook: {e}"))
                })?;

                let body: TgResponse<bool> = resp.json().await.map_err(|e| {
                    CortexError::Channel(format!("Failed to parse webhook response: {e}"))
                })?;

                if !body.ok {
                    return Err(CortexError::Channel(format!(
                        "Telegram setWebhook failed: {}",
                        body.description.unwrap_or_default()
                    )));
                }

                self.started.store(true, Ordering::SeqCst);
                tracing::info!(url = %webhook_url, "telegram webhook registered");
            }
            other => {
                return Err(CortexError::Config(format!(
                    "Unknown Telegram mode: '{other}'. Use 'polling' or 'webhook'."
                )));
            }
        }

        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let _ = self.cancel.send(true);
        self.started.store(false, Ordering::SeqCst);
        tracing::info!("telegram channel stopped");
        Ok(())
    }

    async fn send(&self, target: &OutboundTarget, message: OutboundMessage) -> Result<()> {
        // Determine the chat_id: use group_id if present, otherwise external_id
        let chat_id: i64 = target
            .group_id
            .as_deref()
            .or(Some(&target.external_id))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                CortexError::Channel(format!(
                    "Invalid Telegram chat_id: {}",
                    target.external_id
                ))
            })?;

        // We need the token — try env var first, then check raw metadata
        let token = std::env::var("TELEGRAM_BOT_TOKEN").map_err(|_| {
            CortexError::Channel(
                "TELEGRAM_BOT_TOKEN not set — cannot send outbound message".into(),
            )
        })?;

        let reply_to = target
            .reply_to_message_id
            .as_ref()
            .and_then(|s| s.parse::<i64>().ok());

        let req = SendMessageRequest {
            chat_id,
            text: message.text,
            reply_to_message_id: reply_to,
            parse_mode: Some("Markdown".into()),
        };

        let url = format!("{}{}/sendMessage", BASE_URL, token);
        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| CortexError::Channel(format!("Telegram sendMessage failed: {e}")))?;

        let body: TgResponse<TgSentMessage> = resp.json().await.map_err(|e| {
            CortexError::Channel(format!("Telegram sendMessage parse error: {e}"))
        })?;

        if !body.ok {
            return Err(CortexError::Channel(format!(
                "Telegram sendMessage error: {}",
                body.description.unwrap_or_default()
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

        // Quick liveness check: call getMe
        let token = match std::env::var("TELEGRAM_BOT_TOKEN") {
            Ok(t) => t,
            Err(_) => {
                return ChannelHealth::Degraded {
                    reason: "TELEGRAM_BOT_TOKEN not set".into(),
                }
            }
        };

        let url = format!("{}{}/getMe", BASE_URL, token);
        match self.client.get(&url).send().await {
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
        4096
    }

    async fn send_typing(&self, target: &OutboundTarget) -> Result<()> {
        let chat_id: i64 = target
            .group_id
            .as_deref()
            .or(Some(&target.external_id))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if chat_id == 0 {
            return Ok(());
        }

        let token = std::env::var("TELEGRAM_BOT_TOKEN").unwrap_or_default();
        let url = format!(
            "{}{}/sendChatAction?chat_id={}&action=typing",
            BASE_URL, token, chat_id
        );
        let _ = self.client.get(&url).send().await;
        Ok(())
    }

    async fn edit_message(&self, message_id: &str, new_text: &str) -> Result<()> {
        // Telegram supports editMessageText but we'd need the chat_id too.
        // For now, the basic implementation assumes the message_id is "chat_id:message_id".
        let parts: Vec<&str> = message_id.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(CortexError::Channel(
                "Telegram edit_message requires 'chat_id:message_id' format".into(),
            ));
        }

        let token = std::env::var("TELEGRAM_BOT_TOKEN").map_err(|_| {
            CortexError::Channel("TELEGRAM_BOT_TOKEN not set".into())
        })?;

        let payload = serde_json::json!({
            "chat_id": parts[0].parse::<i64>().unwrap_or(0),
            "message_id": parts[1].parse::<i64>().unwrap_or(0),
            "text": new_text,
            "parse_mode": "Markdown",
        });

        let url = format!("{}{}/editMessageText", BASE_URL, token);
        let _ = self.client.post(&url).json(&payload).send().await;
        Ok(())
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
