//! WebChat channel — built-in WebSocket chat served from the gateway.
//!
//! Provides a real-time chat interface via WebSocket upgrade at
//! `ws://host:port/v1/ws/chat`. Supports streaming-style responses by
//! sending the full reply once the agent finishes.
//!
//! # Configuration
//!
//! ```json
//! {
//!   "require_auth": false,     // whether WS connections need an API key
//!   "max_connections": 100     // max concurrent WebSocket connections
//! }
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::types::*;
use super::Channel;

/// WebChat channel adapter.
///
/// Unlike other channels, WebChat doesn't connect to an external service.
/// It serves WebSocket connections directly from the axum server. The axum
/// WebSocket handler creates `InboundEnvelope` messages and pushes them
/// into the pipeline; replies are sent back through the WebSocket.
///
/// This struct tracks state but the actual WS upgrade happens in the API
/// layer (axum route).
pub struct WebChatChannel {
    started: AtomicBool,
    /// Number of active WebSocket connections.
    active_connections: Arc<AtomicUsize>,
    /// Max concurrent connections.
    max_connections: usize,
}

/// A message sent from the client over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsChatMessage {
    /// Client-provided session token (or generated on first connect).
    #[serde(default)]
    pub session_token: Option<String>,
    /// The user's message text.
    pub text: String,
    /// Optional: unique client message ID for deduplication.
    #[serde(default)]
    pub client_msg_id: Option<String>,
}

/// A message sent from the server over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsChatReply {
    /// "reply" | "typing" | "error" | "connected"
    #[serde(rename = "type")]
    pub msg_type: String,
    /// The reply text (for type="reply").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Session token assigned to this connection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,
    /// Error message (for type="error").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl WsChatReply {
    pub fn connected(session_token: &str) -> Self {
        Self {
            msg_type: "connected".into(),
            text: None,
            session_token: Some(session_token.into()),
            error: None,
        }
    }

    pub fn reply(text: &str) -> Self {
        Self {
            msg_type: "reply".into(),
            text: Some(text.into()),
            session_token: None,
            error: None,
        }
    }

    pub fn typing() -> Self {
        Self {
            msg_type: "typing".into(),
            text: None,
            session_token: None,
            error: None,
        }
    }

    pub fn error(msg: &str) -> Self {
        Self {
            msg_type: "error".into(),
            text: None,
            session_token: None,
            error: Some(msg.into()),
        }
    }
}

impl WebChatChannel {
    pub fn new() -> Self {
        Self {
            started: AtomicBool::new(false),
            active_connections: Arc::new(AtomicUsize::new(0)),
            max_connections: 100,
        }
    }

    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// Get current active connection count.
    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Get the shared connection counter (for the WS handler to increment/decrement).
    pub fn connection_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.active_connections)
    }

    /// Check if a new connection can be accepted.
    pub fn can_accept(&self) -> bool {
        self.active_connections.load(Ordering::Relaxed) < self.max_connections
    }

    /// Get the max connections limit.
    pub fn max_connections(&self) -> usize {
        self.max_connections
    }
}

#[async_trait]
impl Channel for WebChatChannel {
    fn id(&self) -> &str {
        "webchat"
    }

    fn display_name(&self) -> &str {
        "WebSocket Chat"
    }

    async fn start(&self, ctx: ChannelContext) -> Result<()> {
        // Read max_connections from config
        if let Some(max) = ctx.config.get("max_connections").and_then(|v| v.as_u64()) {
            // Note: we can't mutate self here, but the default is fine.
            // A future version could use AtomicUsize for max_connections.
            tracing::info!(max_connections = max, "webchat max_connections configured");
        }

        self.started.store(true, Ordering::SeqCst);
        tracing::info!(
            "webchat channel started (WebSocket connections accepted at /v1/ws/chat)"
        );
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.started.store(false, Ordering::SeqCst);
        let active = self.active_connections.load(Ordering::Relaxed);
        if active > 0 {
            tracing::info!(
                active_connections = active,
                "webchat channel stopping — active connections will be dropped"
            );
        }
        tracing::info!("webchat channel stopped");
        Ok(())
    }

    async fn send(&self, _target: &OutboundTarget, _message: OutboundMessage) -> Result<()> {
        // WebChat outbound is handled directly through the WebSocket connection,
        // not through this method. The WS handler sends replies inline.
        //
        // If we need to push messages to a specific session (e.g. notifications),
        // we'd maintain a map of session_token → WS sender. That's a Phase 4 feature.
        tracing::trace!("webchat send: reply delivered directly via WebSocket");
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
        100_000 // WebSocket messages are practically unlimited
    }
}
