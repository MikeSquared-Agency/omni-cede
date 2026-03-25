# agents.md — Guide for AI Agents Working on omni-cede

You are working on **omni-cede**, the omnichannel deployment variant of the cortex-embedded cognitive engine. This file tells you how to navigate the codebase and contribute effectively.

## What This Repo Is

omni-cede extends the cortex-embedded engine with:
- **HTTP API** (axum) — stateless REST endpoints for multi-client messaging
- **Identity resolution** — maps (channel, external_id) pairs to internal user IDs
- **Session management** — one active session per (user_id, channel), automatic turn tracking

### Ecosystem Position
- **cortex-embedded** (upstream) — the frozen engine
- **cede** — forkable starter kit (no API layer)
- **omni-cede** (this repo) — production omnichannel variant

## Repository Layout

```
src/
  lib.rs              # CortexEmbedded struct, background tasks, decay, consolidation
  types.rs            # All types: Node, Edge, NodeKind, EdgeKind, Message, LlmResponse
  error.rs            # CortexError enum, Result type alias
  config.rs           # Config struct with all tunable parameters
  agent/
    mod.rs            # Re-exports Agent
    orchestrator.rs   # Agent struct, run() and run_turn() methods, tool-call loop
    subagent.rs       # Sub-agent spawning and delegation
  api/
    mod.rs            # axum Router, POST /v1/message, GET /v1/health, sessions, stats
  identity/
    mod.rs            # IdentityResolver: (channel, external_id) → internal user_id
  session/
    mod.rs            # SessionManager: one active session per (user_id, channel)
  db/
    mod.rs            # Db struct (Arc<Mutex<Connection>>), async call() wrapper
    schema.rs         # CREATE TABLE statements, migrations
    queries.rs        # All SQL queries as functions
  embed/
    mod.rs            # EmbedHandle — fastembed wrapper with LRU cache
  hnsw/
    mod.rs            # VectorIndex — 2-tier HNSW (built index + linear buffer)
  graph/
    mod.rs            # BFS traversal, graph walk scoring
  memory/
    mod.rs            # recall(), briefing(), briefing_with_kinds(), recency window
  tools/
    mod.rs            # ToolRegistry, builtin tools
  llm/
    mod.rs            # LlmClient trait, AnthropicClient, OllamaClient, MockLlm
  cli/
    mod.rs            # CLI commands including Serve { host, port }
    graph_tui.rs      # Interactive TUI graph explorer
    graph_viz.rs      # ASCII graph visualization
  bin/
    omni_cede.rs      # Binary entry point, tracing-subscriber init
tests/
  integration.rs      # 22 integration tests
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /v1/message | x-api-key | Send a message, get a response |
| GET | /v1/health | none | Health + node/edge counts |
| GET | /v1/sessions/:user_id | x-api-key | List sessions for a user |
| GET | /v1/stats | x-api-key | Global graph statistics |

### POST /v1/message
```json
{
  "channel": "web",
  "external_id": "user_abc",
  "message": "Hello"
}
```
Returns:
```json
{
  "response": "...",
  "user_id": "internal-uuid",
  "session_id": "session-uuid"
}
```

## Key Architecture (omni-cede-specific)

### Identity Resolution (`src/identity/mod.rs`)
- SQLite tables: `users` (id, created_at), `channel_mappings` (channel, external_id, user_id)
- `resolve(channel, external_id)` → returns existing or creates new internal user_id
- Same external_id on different channels = different internal users

### Session Management (`src/session/mod.rs`)
- SQLite table: `managed_sessions`
- One active session per (user_id, channel)
- `get_or_create(user_id, channel)` → session_id
- `record_turn(session_id)` → updates last_active_at, increments turn_count
- `list_user_sessions(user_id)` → all sessions across channels

### API Layer (`src/api/mod.rs`)
- axum 0.8 Router with tower-http CORS and tracing
- Auth middleware: checks `x-api-key` header against `OMNI_CEDE_API_KEY` env var
- State: `Arc<AppState>` containing CortexEmbedded, IdentityResolver, SessionManager, Agent

### Additional Dependencies vs cede
- `axum = "0.8"`, `tower-http = "0.6"` (cors, trace features)
- `tracing = "0.1"`, `tracing-subscriber = "0.3"` (env-filter feature)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| ANTHROPIC_API_KEY | Yes (unless --ollama) | Claude API key |
| OMNI_CEDE_API_KEY | Yes (for API mode) | API authentication key |
| RUST_LOG | No | Tracing filter (default: info) |

## Build and Test

```bash
cargo build
cargo test -- --test-threads=1    # 28 tests

# Run the HTTP server
OMNI_CEDE_API_KEY=secret cargo run -- serve --host 0.0.0.0 --port 3000
```

## Conventions

- Async DB: `db.call(move |conn| { ... }).await`
- Embeddings: 384-dim f32 (BAAI/bge-small-en-v1.5)
- Node IDs: UUID v4 strings
- Timestamps: Unix seconds (i64)
- Error handling: `CortexError` enum, `Result<T>` alias
- API errors: JSON `{"error": "message"}` with appropriate HTTP status

## Branch Policy

- `master` is protected: no direct push, PRs required
- Work on `dev` branch, merge via PR