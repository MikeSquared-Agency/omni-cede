//! Session manager — one active session per (user_id, channel).
//!
//! In omni-cede, sessions are scoped to a specific user on a specific channel.
//! A WhatsApp conversation has its own session; the same user on Telegram gets
//! a separate one. The recency window in the engine's hybrid recall operates
//! on the session, giving each channel its own conversational flow while the
//! semantic (HNSW) layer searches the global graph — cross-channel knowledge.
//!
//! Sessions are stored both as graph nodes (for the engine's native recall)
//! and in a lightweight lookup table for fast resolution by (user_id, channel).

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::db::Db;
use crate::db::queries;
use crate::error::Result;
use crate::types::{Node, NodeId};

/// Metadata for a managed session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedSession {
    /// Graph node ID — this is also the session_id passed to `agent.run_turn()`.
    pub node_id: NodeId,
    /// Internal user ID from the identity layer.
    pub user_id: String,
    /// Channel this session belongs to (e.g. "whatsapp", "telegram", "api").
    pub channel: String,
    /// Unix timestamp when this session was created.
    pub created_at: i64,
    /// Number of turns processed in this session.
    pub turn_count: i64,
    /// Unix timestamp of the last turn.
    pub last_active: i64,
}

/// Create the session lookup table if it doesn't exist.
pub fn create_tables(conn: &Connection) -> std::result::Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS managed_sessions (
            node_id     TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            channel     TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            turn_count  INTEGER NOT NULL DEFAULT 0,
            last_active INTEGER NOT NULL,
            UNIQUE(user_id, channel)
        );

        CREATE INDEX IF NOT EXISTS idx_managed_sessions_user
            ON managed_sessions(user_id);",
    )?;
    Ok(())
}

/// Get or create the active session for a (user_id, channel) pair.
///
/// If a session already exists, returns it (and bumps `last_active`).
/// Otherwise creates a new `Node::session()` in the graph and a row in
/// the lookup table.
pub async fn get_or_create(
    db: &Db,
    user_id: &str,
    channel: &str,
) -> Result<ManagedSession> {
    let uid = user_id.to_string();
    let ch = channel.to_string();

    db.call(move |conn| {
        create_tables(conn)?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Try to find existing session
        let existing: Option<ManagedSession> = conn
            .query_row(
                "SELECT node_id, user_id, channel, created_at, turn_count, last_active
                 FROM managed_sessions
                 WHERE user_id = ?1 AND channel = ?2",
                params![uid, ch],
                |row| {
                    Ok(ManagedSession {
                        node_id: row.get(0)?,
                        user_id: row.get(1)?,
                        channel: row.get(2)?,
                        created_at: row.get(3)?,
                        turn_count: row.get(4)?,
                        last_active: row.get(5)?,
                    })
                },
            )
            .optional()?;

        if let Some(mut session) = existing {
            // Bump last_active
            conn.execute(
                "UPDATE managed_sessions SET last_active = ?1 WHERE node_id = ?2",
                params![now, session.node_id],
            )?;
            session.last_active = now;
            return Ok(session);
        }

        // Create a new session node in the graph
        let session_node = Node::session(&format!("{ch} session for {uid}"));
        let node_id = session_node.id.clone();
        queries::insert_node(conn, &session_node)?;

        // Insert into the lookup table
        conn.execute(
            "INSERT INTO managed_sessions (node_id, user_id, channel, created_at, turn_count, last_active)
             VALUES (?1, ?2, ?3, ?4, 0, ?5)",
            params![node_id, uid, ch, now, now],
        )?;

        Ok(ManagedSession {
            node_id,
            user_id: uid,
            channel: ch,
            created_at: now,
            turn_count: 0,
            last_active: now,
        })
    })
    .await
}

/// Increment the turn count for a session after a successful turn.
pub async fn record_turn(db: &Db, session_node_id: &str) -> Result<()> {
    let nid = session_node_id.to_string();
    db.call(move |conn| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        conn.execute(
            "UPDATE managed_sessions SET turn_count = turn_count + 1, last_active = ?1 WHERE node_id = ?2",
            params![now, nid],
        )?;
        Ok(())
    })
    .await
}

/// List all sessions for a user.
pub async fn list_user_sessions(db: &Db, user_id: &str) -> Result<Vec<ManagedSession>> {
    let uid = user_id.to_string();
    db.call(move |conn| {
        create_tables(conn)?;
        let mut stmt = conn.prepare(
            "SELECT node_id, user_id, channel, created_at, turn_count, last_active
             FROM managed_sessions
             WHERE user_id = ?1
             ORDER BY last_active DESC",
        )?;
        let rows = stmt.query_map(params![uid], |row| {
            Ok(ManagedSession {
                node_id: row.get(0)?,
                user_id: row.get(1)?,
                channel: row.get(2)?,
                created_at: row.get(3)?,
                turn_count: row.get(4)?,
                last_active: row.get(5)?,
            })
        })?;
        let mut result = Vec::new();
        for r in rows {
            result.push(r?);
        }
        Ok(result)
    })
    .await
}

/// Get total session count and total turn count.
pub async fn stats(db: &Db) -> Result<(i64, i64)> {
    db.call(move |conn| {
        create_tables(conn)?;
        let session_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM managed_sessions",
            [],
            |row| row.get(0),
        )?;
        let turn_count: i64 = conn.query_row(
            "SELECT COALESCE(SUM(turn_count), 0) FROM managed_sessions",
            [],
            |row| row.get(0),
        )?;
        Ok((session_count, turn_count))
    })
    .await
}
