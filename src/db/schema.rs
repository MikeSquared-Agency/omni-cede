use rusqlite::Connection;
use crate::error::Result;

/// Create all tables, indices, and pragmas. Idempotent.
pub fn create_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS nodes (
            id           TEXT PRIMARY KEY,
            kind         TEXT NOT NULL,
            title        TEXT NOT NULL,
            body         TEXT,
            importance   REAL DEFAULT 0.5,
            trust_score  REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            created_at   INTEGER NOT NULL,
            last_access  INTEGER,
            decay_rate   REAL DEFAULT 0.01,
            embedding    BLOB
        );

        CREATE TABLE IF NOT EXISTS edges (
            id         TEXT PRIMARY KEY,
            src        TEXT REFERENCES nodes(id) ON DELETE CASCADE,
            dst        TEXT REFERENCES nodes(id) ON DELETE CASCADE,
            kind       TEXT NOT NULL,
            weight     REAL DEFAULT 1.0,
            created_at INTEGER NOT NULL,
            metadata   TEXT
        );

        CREATE TABLE IF NOT EXISTS contradictions (
            node_a      TEXT REFERENCES nodes(id),
            node_b      TEXT REFERENCES nodes(id),
            detected_at INTEGER,
            resolved    INTEGER DEFAULT 0,
            PRIMARY KEY (node_a, node_b)
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_kind       ON nodes(kind);
        CREATE INDEX IF NOT EXISTS idx_nodes_importance  ON nodes(importance DESC);
        CREATE INDEX IF NOT EXISTS idx_edges_src         ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_edges_dst         ON edges(dst);
        CREATE INDEX IF NOT EXISTS idx_edges_kind        ON edges(kind);
        ",
    )?;
    Ok(())
}
