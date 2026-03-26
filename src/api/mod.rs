//! HTTP API — the omnichannel gateway.
//!
//! Provides a REST API that any messaging platform adapter can call:
//!
//! - `POST /v1/message`         — send a message (resolves identity, gets/creates session, runs turn)
//! - `POST /v1/channels/webhook/inbound` — generic webhook inbound (pipeline-routed)
//! - `GET  /v1/channels`        — list channels and their health
//! - `GET  /v1/health`          — liveness check
//! - `GET  /v1/sessions/:user_id` — list sessions for a user
//! - `GET  /v1/stats`           — graph + session statistics
//! - `GET  /v1/ws/chat`         — WebSocket chat endpoint
//!
//! Authentication is via an `x-api-key` header checked against the `API_KEY` env var.
//! If `API_KEY` is not set, authentication is disabled (development mode).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::{
    Json, Router,
    extract::{Path, Query, State, ws::{WebSocket, WebSocketUpgrade, Message as WsMessage}},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::agent::orchestrator::Agent;
use crate::channels::pipeline::Pipeline;
use crate::channels::types::InboundEnvelope;
use crate::channels::webchat::{WsChatMessage, WsChatReply};
use crate::channels::registry::ChannelRegistry;
use crate::session;
use crate::CortexEmbedded;

// ─── Shared state ───────────────────────────────────────

/// Application state shared across all request handlers.
pub struct AppState {
    pub cx: CortexEmbedded,
    pub agent: Agent,
    pub api_key: Option<String>,
    /// The omnichannel pipeline (identity → session → hooks → agent → outbound).
    pub pipeline: Arc<Pipeline>,
    /// Channel registry for health/status queries.
    pub registry: Arc<ChannelRegistry>,
    /// WebChat active connection counter (shared with WebChatChannel).
    pub webchat_counter: Arc<AtomicUsize>,
    /// WebChat maximum concurrent connections.
    pub webchat_max: usize,
}

// ─── Request / Response types ───────────────────────────

#[derive(Debug, Deserialize)]
pub struct MessageRequest {
    /// Channel identifier, e.g. "whatsapp", "telegram", "api", "cli".
    pub channel: String,
    /// The external user ID on that channel (phone number, chat id, etc.).
    pub external_id: String,
    /// The user's message text.
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    /// The agent's reply.
    pub reply: String,
    /// Internal user ID (for follow-up requests).
    pub user_id: String,
    /// Session ID (graph node id used for this conversation).
    pub session_id: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub nodes: i64,
    pub edges: i64,
    pub by_kind: std::collections::HashMap<String, i64>,
    pub managed_sessions: i64,
    pub total_turns: i64,
}

#[derive(Debug, Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub channel: String,
    pub created_at: i64,
    pub turn_count: i64,
    pub last_active: i64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Webhook inbound request — superset of MessageRequest with optional fields.
#[derive(Debug, Deserialize)]
pub struct WebhookInboundRequest {
    pub channel: Option<String>,
    pub external_id: String,
    pub text: String,
    #[serde(default)]
    pub sender_name: Option<String>,
    #[serde(default)]
    pub callback_url: Option<String>,
    #[serde(default)]
    pub group_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChannelStatusResponse {
    pub id: String,
    pub health: crate::channels::types::ChannelHealth,
}

// ─── Router ─────────────────────────────────────────────

/// Build the axum `Router` with all routes and middleware.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // Core messaging endpoints
        .route("/v1/message", post(handle_message))
        .route("/v1/channels/webhook/inbound", post(handle_webhook_inbound))
        // Session / stats endpoints
        .route("/v1/sessions/{user_id}", get(handle_sessions))
        .route("/v1/stats", get(handle_stats))
        // Channel management
        .route("/v1/channels", get(handle_channels))
        // Auth middleware on all of the above
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        // Public endpoints (auth via query param for WS)
        .route("/v1/health", get(handle_health))
        .route("/v1/ws/chat", get(handle_ws_upgrade))
        // Cross-cutting middleware
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ─── Auth middleware ────────────────────────────────────

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> impl IntoResponse {
    // If no API_KEY is set, skip auth (dev mode)
    let Some(ref expected) = state.api_key else {
        return next.run(request).await.into_response();
    };

    match headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        Some(key) if key == expected => next.run(request).await.into_response(),
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid or missing x-api-key header".into(),
            }),
        )
            .into_response(),
    }
}

// ─── Handlers ───────────────────────────────────────────

async fn handle_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Original message handler — uses the pipeline for processing.
async fn handle_message(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessageRequest>,
) -> impl IntoResponse {
    let envelope = InboundEnvelope::new(&req.channel, &req.external_id, &req.text);

    match state.pipeline.process_sync(envelope, &state.cx.db, &state.agent).await {
        Ok(result) => (
            StatusCode::OK,
            Json(MessageResponse {
                reply: result.reply,
                user_id: result.user_id,
                session_id: result.session_id,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{e}"),
            }),
        )
            .into_response(),
    }
}

/// Webhook inbound — generic webhook channel messages.
async fn handle_webhook_inbound(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WebhookInboundRequest>,
) -> impl IntoResponse {
    let mut envelope = InboundEnvelope::new(
        req.channel.as_deref().unwrap_or("webhook"),
        &req.external_id,
        &req.text,
    );
    envelope.sender_name = req.sender_name;
    envelope.callback_url = req.callback_url;
    envelope.group_id = req.group_id;

    match state.pipeline.process(envelope, &state.cx.db, &state.agent).await {
        Ok(result) => (
            StatusCode::OK,
            Json(MessageResponse {
                reply: result.reply,
                user_id: result.user_id,
                session_id: result.session_id,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{e}"),
            }),
        )
            .into_response(),
    }
}

/// List all channels and their health status.
async fn handle_channels(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let health_list = state.registry.health_all().await;
    let statuses: Vec<ChannelStatusResponse> = health_list
        .into_iter()
        .map(|(id, health)| ChannelStatusResponse { id, health })
        .collect();
    (StatusCode::OK, Json(statuses)).into_response()
}

async fn handle_sessions(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match session::list_user_sessions(&state.cx.db, &user_id).await {
        Ok(sessions) => {
            let infos: Vec<SessionInfo> = sessions
                .into_iter()
                .map(|s| SessionInfo {
                    session_id: s.node_id,
                    channel: s.channel,
                    created_at: s.created_at,
                    turn_count: s.turn_count,
                    last_active: s.last_active,
                })
                .collect();
            (StatusCode::OK, Json(infos)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to list sessions: {e}"),
            }),
        )
            .into_response(),
    }
}

async fn handle_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let graph_stats = match state.cx.stats().await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to get graph stats: {e}"),
                }),
            )
                .into_response();
        }
    };

    let (managed_sessions, total_turns) = match session::stats(&state.cx.db).await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to get session stats: {e}"),
                }),
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        Json(StatsResponse {
            nodes: graph_stats.0,
            edges: graph_stats.1,
            by_kind: graph_stats.2,
            managed_sessions,
            total_turns,
        }),
    )
        .into_response()
}

// ─── WebSocket chat ────────────────────────────────────

async fn handle_ws_upgrade(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Auth check: if API_KEY is set, verify ?api_key= query param
    if let Some(ref expected) = state.api_key {
        match params.get("api_key") {
            Some(k) if k == expected => {}
            _ => {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "Invalid or missing api_key query parameter".into(),
                    }),
                )
                    .into_response();
            }
        }
    }

    // Check connection limit
    let current = state.webchat_counter.load(Ordering::Relaxed);
    if current >= state.webchat_max {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: format!("WebChat connection limit reached ({}/{})", current, state.webchat_max),
            }),
        )
            .into_response();
    }

    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
        .into_response()
}

async fn handle_ws_connection(socket: WebSocket, state: Arc<AppState>) {
    use futures::stream::StreamExt;
    use futures::sink::SinkExt;

    let (mut sender, mut receiver) = socket.split();
    let session_token = uuid::Uuid::new_v4().to_string();

    // Increment connection counter
    state.webchat_counter.fetch_add(1, Ordering::Relaxed);

    // Send connected message
    let connected = WsChatReply::connected(&session_token);
    if let Ok(json) = serde_json::to_string(&connected) {
        let _ = sender.send(WsMessage::Text(json.into())).await;
    }

    tracing::info!(session_token = %session_token, "webchat client connected");

    // Process messages
    while let Some(Ok(msg)) = receiver.next().await {
        let text = match msg {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Close(_) => break,
            _ => continue,
        };

        // Parse the client message
        let chat_msg: WsChatMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let err = WsChatReply::error(&format!("Invalid message format: {e}"));
                if let Ok(json) = serde_json::to_string(&err) {
                    let _ = sender.send(WsMessage::Text(json.into())).await;
                }
                continue;
            }
        };

        // Send typing indicator
        let typing = WsChatReply::typing();
        if let Ok(json) = serde_json::to_string(&typing) {
            let _ = sender.send(WsMessage::Text(json.into())).await;
        }

        // Use session_token from message or from connection
        let token = chat_msg
            .session_token
            .as_deref()
            .unwrap_or(&session_token);

        // Create inbound envelope and process through pipeline
        let envelope = InboundEnvelope::new("webchat", token, &chat_msg.text);

        match state
            .pipeline
            .process_sync(envelope, &state.cx.db, &state.agent)
            .await
        {
            Ok(result) => {
                let reply = WsChatReply::reply(&result.reply);
                if let Ok(json) = serde_json::to_string(&reply) {
                    if sender.send(WsMessage::Text(json.into())).await.is_err() {
                        break;
                    }
                }
            }
            Err(e) => {
                let err = WsChatReply::error(&format!("Agent error: {e}"));
                if let Ok(json) = serde_json::to_string(&err) {
                    let _ = sender.send(WsMessage::Text(json.into())).await;
                }
            }
        }
    }

    // Decrement connection counter
    state.webchat_counter.fetch_sub(1, Ordering::Relaxed);
    tracing::info!(session_token = %session_token, "webchat client disconnected");
}
