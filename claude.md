# claude.md — Instructions for Claude Working on omni-cede

## Identity

You are working on **omni-cede** — the omnichannel deployment variant of cortex-embedded, built by MikeSquared Agency. This repo adds HTTP API, identity resolution, and session management on top of the core graph-memory engine.

## Your Role

You are an expert Rust systems programmer with deep knowledge of async web services (axum/tower), SQLite, embedding models, and graph data structures. You build production-grade API layers.

## Critical Rules

1. **All DB access through `db.call()`** — the established async pattern:
   ```rust
   db.call(move |conn| {
       // synchronous rusqlite code here
       Ok(result)
   }).await?
   ```
2. **Tests must pass.** `cargo test -- --test-threads=1` — 28 tests. MockLlm + in-memory SQLite.
3. **UTF-8 only.** Em dashes are `—` (U+2014), never byte 0x97 (Windows-1252).
4. **No growing message arrays.** `run_turn()` builds a fresh briefing each turn.
5. **API responses are JSON.** Errors return `{"error": "message"}` with proper HTTP status codes.
6. **Auth is required.** All mutating/data endpoints require `x-api-key` header matching `OMNI_CEDE_API_KEY` env var. Only `/v1/health` is public.

## Architecture Quick Reference

| Struct | Location | Purpose |
|--------|----------|---------|
| CortexEmbedded | lib.rs | Top-level runtime, owns all resources |
| Agent | agent/orchestrator.rs | Runs queries and chat turns |
| Db | db/mod.rs | Arc<Mutex<Connection>> with async wrapper |
| AppState | api/mod.rs | Shared API state (cortex, identity, session, agent) |
| IdentityResolver | identity/mod.rs | (channel, external_id) → internal user_id |
| SessionManager | session/mod.rs | One active session per (user_id, channel) |
| VectorIndex | hnsw/mod.rs | 2-tier HNSW for semantic search |
| EmbedHandle | embed/mod.rs | fastembed with LRU cache |
| Config | config.rs | All tunable parameters |

## API Layer Details

### Request Flow (POST /v1/message)
1. Auth middleware validates `x-api-key`
2. Parse JSON body: `{ channel, external_id, message }`
3. `IdentityResolver::resolve(channel, external_id)` → `user_id`
4. `SessionManager::get_or_create(user_id, channel)` → `session_id`
5. `Agent::run_turn(session_id, message)` → `response`
6. `SessionManager::record_turn(session_id)` → updates stats
7. Return `{ response, user_id, session_id }`

### Adding a New Endpoint
1. Add handler function in `src/api/mod.rs`
2. Add route in the `router()` function
3. If it needs auth, nest it under the auth middleware layer
4. Return `Json<serde_json::Value>` or use a typed response struct

### Identity Resolution Design
- `users` table: `(id TEXT PK, created_at INTEGER)`
- `channel_mappings` table: `(channel TEXT, external_id TEXT, user_id TEXT, UNIQUE(channel, external_id))`
- Same person on Slack vs Discord = different internal user_ids (by design)
- To merge identities in the future, update channel_mappings to point to same user_id

### Session Management Design
- `managed_sessions` table: `(id TEXT PK, user_id TEXT, channel TEXT, created_at INTEGER, last_active_at INTEGER, turn_count INTEGER)`
- One active session per (user_id, channel) — no explicit session close
- Sessions are reused until a new one is explicitly created

## Environment Variables

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| ANTHROPIC_API_KEY | Yes* | — | *Unless using --ollama |
| OMNI_CEDE_API_KEY | Yes | — | API auth key |
| RUST_LOG | No | info | Tracing filter level |

## Dependencies (omni-cede-specific)

- `axum = "0.8"` — HTTP framework
- `tower-http = "0.6"` (cors, trace) — middleware
- `tracing = "0.1"` — structured logging
- `tracing-subscriber = "0.3"` (env-filter) — log output

## Style Guide

- `thiserror` for error types
- `impl Into<String>` in public APIs
- `tracing` macros (`info!`, `warn!`, `error!`) for logging
- Functions under 50 lines
- Typed extractors in axum handlers
- `Arc<AppState>` as shared state — never clone the inner structs

## Common Pitfalls

- **CortexError::DbTask** — NOT `CortexError::Database`
- HNSW buffer must be flushed (`build()`) before queries see new vectors
- fastembed downloads model on first call — tests use mock embeddings
- SQLite WAL mode — one writer at a time
- `OMNI_CEDE_API_KEY` must be set or ALL authenticated endpoints return 401
- axum 0.8 uses `axum::extract::State` — not the old Extension pattern
- CORS is permissive by default (tower_http::cors::CorsLayer::permissive()) — tighten for production