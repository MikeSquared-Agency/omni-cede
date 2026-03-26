//! Channel types — the shared vocabulary for the omnichannel pipeline.
//!
//! These types are channel-agnostic: every adapter speaks in terms of
//! [`InboundEnvelope`] (messages coming in) and [`OutboundMessage`] /
//! [`OutboundTarget`] (messages going out). The pipeline never sees
//! platform-specific payloads.

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::db::Db;

// ─── Inbound ────────────────────────────────────────────

/// A normalised inbound message from any channel.
///
/// Channel adapters construct this from raw platform payloads and push it
/// into the pipeline via `ChannelContext::inbound_tx`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundEnvelope {
    /// Which channel this came from ("telegram", "discord", "webhook", …).
    pub channel: String,
    /// The sender's external identifier on the channel.
    pub external_id: String,
    /// Display name of the sender (if the channel provides one).
    pub sender_name: Option<String>,
    /// The user's message text.
    pub text: String,
    /// Optional media attachment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<MediaPayload>,
    /// If replying to a specific message, its platform message ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
    /// Group / guild / workspace ID (if this is a group message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    /// A URL the channel can POST the reply to (webhook callback).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    /// The raw, channel-specific payload for hooks that need it.
    #[serde(default)]
    pub raw: serde_json::Value,
    /// Unix timestamp (seconds) when the message was received.
    pub timestamp: i64,
}

impl InboundEnvelope {
    /// Build a minimal envelope (used by the webhook adapter and tests).
    pub fn new(channel: &str, external_id: &str, text: &str) -> Self {
        Self {
            channel: channel.to_string(),
            external_id: external_id.to_string(),
            sender_name: None,
            text: text.to_string(),
            media: None,
            reply_to: None,
            group_id: None,
            callback_url: None,
            raw: serde_json::Value::Null,
            timestamp: now_unix(),
        }
    }
}

// ─── Outbound ───────────────────────────────────────────

/// Who to send a reply to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundTarget {
    /// Channel identifier to route through.
    pub channel: String,
    /// External user/chat ID on that channel.
    pub external_id: String,
    /// Group/guild context (if replying in a group).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    /// Platform message ID to reply to (threaded replies).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to_message_id: Option<String>,
    /// Optional callback URL (for webhook channels that POST replies).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
}

impl OutboundTarget {
    /// Derive an outbound target from an inbound envelope.
    pub fn from_envelope(env: &InboundEnvelope) -> Self {
        Self {
            channel: env.channel.clone(),
            external_id: env.external_id.clone(),
            group_id: env.group_id.clone(),
            reply_to_message_id: env.reply_to.clone(),
            callback_url: env.callback_url.clone(),
        }
    }
}

/// An outbound message — text, optional media, arbitrary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundMessage {
    /// The reply text.
    pub text: String,
    /// Optional media attachment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<MediaPayload>,
    /// Arbitrary metadata (per-channel or per-hook).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl OutboundMessage {
    /// Plain text reply.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            media: None,
            metadata: serde_json::Value::Null,
        }
    }
}

// ─── Media ──────────────────────────────────────────────

/// A media attachment (image, audio, video, document).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaPayload {
    pub kind: MediaKind,
    /// Raw bytes — `#[serde(skip)]` because we don't serialise blobs over JSON.
    #[serde(skip)]
    pub data: Vec<u8>,
    /// MIME type, e.g. "image/jpeg".
    pub mime_type: String,
    /// Original filename, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// URL where the media can be fetched (for channels that use URLs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Audio,
    Video,
    Document,
}

// ─── Channel health ─────────────────────────────────────

/// Reported by each channel adapter via the `health()` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ChannelHealth {
    Connected,
    Degraded { reason: String },
    Disconnected { reason: String },
}

// ─── Channel context ────────────────────────────────────

/// Passed to a channel adapter when it starts.
///
/// Gives the adapter everything it needs to push inbound messages into the
/// pipeline and access channel-specific configuration.
pub struct ChannelContext {
    /// Push inbound messages here — the pipeline picks them up.
    pub inbound_tx: mpsc::Sender<InboundEnvelope>,
    /// Database handle (for low-level needs; most channels don't need this).
    pub db: Db,
    /// Channel-specific configuration section (parsed from master config).
    pub config: serde_json::Value,
    /// Shutdown signal — channels should select on this and exit gracefully.
    pub shutdown: tokio::sync::watch::Receiver<bool>,
}

// ─── Message length limits (for outbound chunking) ──────

/// Maximum message length per channel. If a reply exceeds this, the outbound
/// pipeline will split it into multiple sends.
pub fn max_message_length(channel: &str) -> usize {
    match channel {
        "telegram" => 4096,
        "discord" => 2000,
        "slack" => 40_000,
        "whatsapp" => 65_536,
        "webchat" => 100_000, // practically unlimited
        _ => 4096,            // conservative default
    }
}

// ─── Pipeline result (returned to the HTTP API) ─────────

/// The result of processing a single inbound message through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// The agent's reply text.
    pub reply: String,
    /// Internal user ID resolved by the identity layer.
    pub user_id: String,
    /// Graph-node session ID.
    pub session_id: String,
}

// ─── Helpers ────────────────────────────────────────────

pub(crate) fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
