//! Channel registry — manages the lifecycle of all channel adapters.
//!
//! The registry owns every registered [`Channel`], starts and stops them as a
//! group, and provides the shared inbound MPSC sender that channels push
//! messages into. The [`Pipeline`] reads from the other end.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, watch, RwLock};

use crate::db::Db;
use crate::error::Result;

use super::types::{ChannelContext, ChannelHealth, InboundEnvelope};
use super::Channel;

/// Manages all registered channel adapters.
pub struct ChannelRegistry {
    /// Registered channels keyed by their `id()`.
    channels: Arc<RwLock<HashMap<String, Arc<dyn Channel>>>>,
    /// The sending half — cloned to each channel on start.
    inbound_tx: mpsc::Sender<InboundEnvelope>,
    /// The receiving half — handed to the pipeline.
    inbound_rx: Option<mpsc::Receiver<InboundEnvelope>>,
    /// Shutdown broadcaster.
    shutdown_tx: watch::Sender<bool>,
    /// Shutdown receiver (cloned per channel).
    shutdown_rx: watch::Receiver<bool>,
}

impl ChannelRegistry {
    /// Create a new registry with the given inbound buffer size.
    pub fn new(buffer: usize) -> Self {
        let (inbound_tx, inbound_rx) = mpsc::channel(buffer);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
            inbound_tx,
            inbound_rx: Some(inbound_rx),
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Register a channel adapter. If one with the same `id()` already exists
    /// it is replaced (the old one is not stopped — call `stop_all` first).
    pub async fn register(&self, channel: Arc<dyn Channel>) {
        let id = channel.id().to_string();
        tracing::info!(channel = %id, "channel registered");
        self.channels.write().await.insert(id, channel);
    }

    /// Take the inbound receiver. Only call once — the pipeline needs it.
    pub fn take_inbound_rx(&mut self) -> Option<mpsc::Receiver<InboundEnvelope>> {
        self.inbound_rx.take()
    }

    /// Clone the inbound sender. Used by the notification delivery loop to
    /// inject synthetic messages into the pipeline.
    pub fn clone_inbound_tx(&self) -> mpsc::Sender<InboundEnvelope> {
        self.inbound_tx.clone()
    }

    /// Start all registered channels.
    ///
    /// Each channel receives its own [`ChannelContext`] with a cloned
    /// `inbound_tx`, the DB handle, its config section from `channel_configs`,
    /// and the shutdown watch.
    pub async fn start_all(
        &self,
        db: &Db,
        channel_configs: &HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let channels = self.channels.read().await;
        let mut started = 0usize;
        let mut failed = 0usize;

        for (id, ch) in channels.iter() {
            let config = channel_configs
                .get(id)
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            let ctx = ChannelContext {
                inbound_tx: self.inbound_tx.clone(),
                db: db.clone(),
                config,
                shutdown: self.shutdown_rx.clone(),
            };

            tracing::info!(channel = %id, "starting channel adapter");
            match ch.start(ctx).await {
                Ok(()) => {
                    started += 1;
                }
                Err(e) => {
                    // Log and skip — don't abort other channels
                    tracing::warn!(channel = %id, error = %e, "channel failed to start (skipping)");
                    failed += 1;
                }
            }
        }

        tracing::info!(started, failed, "channel startup complete");
        Ok(())
    }

    /// Stop all registered channels and signal shutdown.
    pub async fn stop_all(&self) -> Result<()> {
        let _ = self.shutdown_tx.send(true);
        let channels = self.channels.read().await;
        for (id, ch) in channels.iter() {
            tracing::info!(channel = %id, "stopping channel adapter");
            if let Err(e) = ch.stop().await {
                tracing::warn!(channel = %id, error = %e, "error stopping channel");
            }
        }
        Ok(())
    }

    /// Get a channel adapter by its ID (for outbound routing).
    pub async fn get(&self, id: &str) -> Option<Arc<dyn Channel>> {
        self.channels.read().await.get(id).cloned()
    }

    /// List all channels and their current health.
    pub async fn health_all(&self) -> Vec<(String, ChannelHealth)> {
        let channels = self.channels.read().await;
        let mut out = Vec::with_capacity(channels.len());
        for (id, ch) in channels.iter() {
            let h = ch.health().await;
            out.push((id.clone(), h));
        }
        out
    }

    /// List the IDs of all registered channels.
    pub async fn list_ids(&self) -> Vec<String> {
        self.channels.read().await.keys().cloned().collect()
    }
}
