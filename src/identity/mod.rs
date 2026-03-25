//! Identity layer — maps external channel identifiers to internal user IDs.
//!
//! A single human can interact via multiple channels (WhatsApp, Telegram, REST
//! API, CLI). Each channel has its own external identifier format. The identity
//! layer resolves all of them to a single internal `UserId`.
//!
//! Storage: a dedicated SQLite table (`identities`) alongside the graph DB.

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::Db;
use crate::error::Result;

/// A unique internal user identifier.
pub type UserId = String;

/// A channel identifier — the external handle for a user on a specific platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelId {
    /// e.g. "whatsapp", "telegram", "api", "cli"
    pub channel: String,
    /// e.g. "+447123456789", "12345678", "api-key-hash", "local"
    pub external_id: String,
}

impl ChannelId {
    pub fn new(channel: &str, external_id: &str) -> Self {
        Self {
            channel: channel.to_string(),
            external_id: external_id.to_string(),
        }
    }

    /// Canonical string form: "channel:external_id"
    pub fn canonical(&self) -> String {
        format!("{}:{}", self.channel, self.external_id)
    }
}

/// A user record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: UserId,
    pub display_name: Option<String>,
    pub created_at: i64,
}

/// Create the identities table if it doesn't exist.
pub fn create_tables(conn: &Connection) -> std::result::Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            display_name TEXT,
            created_at  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS channel_mappings (
            channel     TEXT NOT NULL,
            external_id TEXT NOT NULL,
            user_id     TEXT NOT NULL REFERENCES users(id),
            created_at  INTEGER NOT NULL,
            PRIMARY KEY (channel, external_id)
        );

        CREATE INDEX IF NOT EXISTS idx_channel_mappings_user
            ON channel_mappings(user_id);",
    )?;
    Ok(())
}

/// Look up a user by their channel identifier, or create a new one.
pub fn resolve_or_create(
    conn: &Connection,
    channel_id: &ChannelId,
) -> std::result::Result<User, rusqlite::Error> {
    // Try to find existing mapping
    let existing: Option<String> = conn
        .query_row(
            "SELECT user_id FROM channel_mappings WHERE channel = ?1 AND external_id = ?2",
            params![channel_id.channel, channel_id.external_id],
            |row| row.get(0),
        )
        .optional()?;

    if let Some(user_id) = existing {
        let user = conn.query_row(
            "SELECT id, display_name, created_at FROM users WHERE id = ?1",
            params![user_id],
            |row| {
                Ok(User {
                    id: row.get(0)?,
                    display_name: row.get(1)?,
                    created_at: row.get(2)?,
                })
            },
        )?;
        return Ok(user);
    }

    // Create new user
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let user_id = Uuid::new_v4().to_string();

    conn.execute(
        "INSERT INTO users (id, display_name, created_at) VALUES (?1, ?2, ?3)",
        params![user_id, Option::<String>::None, now],
    )?;

    conn.execute(
        "INSERT INTO channel_mappings (channel, external_id, user_id, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![channel_id.channel, channel_id.external_id, user_id, now],
    )?;

    Ok(User {
        id: user_id,
        display_name: None,
        created_at: now,
    })
}

/// Link an additional channel identifier to an existing user.
pub fn link_channel(
    conn: &Connection,
    user_id: &str,
    channel_id: &ChannelId,
) -> std::result::Result<(), rusqlite::Error> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    conn.execute(
        "INSERT OR IGNORE INTO channel_mappings (channel, external_id, user_id, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![channel_id.channel, channel_id.external_id, user_id, now],
    )?;
    Ok(())
}

/// List all channel identifiers for a user.
pub fn list_channels(
    conn: &Connection,
    user_id: &str,
) -> std::result::Result<Vec<ChannelId>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT channel, external_id FROM channel_mappings WHERE user_id = ?1",
    )?;
    let rows = stmt.query_map(params![user_id], |row| {
        Ok(ChannelId {
            channel: row.get(0)?,
            external_id: row.get(1)?,
        })
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

/// Async wrapper: resolve or create a user from a channel identifier.
pub async fn resolve_user(db: &Db, channel_id: ChannelId) -> Result<User> {
    db.call(move |conn| {
        create_tables(conn)?;
        resolve_or_create(conn, &channel_id).map_err(Into::into)
    })
    .await
    .map_err(|e| crate::error::CortexError::DbTask(e.to_string()))
}
