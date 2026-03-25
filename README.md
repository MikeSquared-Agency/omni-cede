# omni-cede

**Omnichannel AI agent powered by embedded memory graphs. One API, every channel, one graph.**

omni-cede extends [cede](https://github.com/MikeSquared-Agency/cede) with an HTTP API, identity resolution, and per-channel session management — all backed by an embedded memory graph (single SQLite file, no external DB). Connect WhatsApp, Telegram, Slack, Discord, or any custom integration — the agent remembers across all of them because every interaction is a node in the same graph.

## Ecosystem

```
cortex-embedded          <-- embedded memory graph engine (upstream)
  |-- cede               <-- forkable starter kit
       |-- omni-cede     <-- you are here (omnichannel deployment)
```

## What omni-cede Adds

On top of everything in cede (embedded memory graph, hybrid recall, auto-linking, decay, tools, sub-agents, TUI), omni-cede adds:

| Layer | What it does |
|-------|-------------|
| **HTTP API** | `POST /v1/message` — send a message from any channel and get a reply |
| **Identity** | Maps `(channel, external_id)` pairs to internal user IDs. Same person on WhatsApp and Telegram = same user |
| **Sessions** | One active session per (user, channel). WhatsApp gets its own conversational flow; Telegram gets another. Semantic recall searches the global graph — cross-channel knowledge |
| **Auth** | `x-api-key` header middleware. Set `API_KEY` env var to enable; omit for dev mode |

## Quick Start

```bash
# Clone
git clone https://github.com/MikeSquared-Agency/omni-cede.git
cd omni-cede

# Build
cargo build --release

# Start the API server
ANTHROPIC_API_KEY=sk-ant-... omni-cede serve
# Custom host/port
omni-cede serve --host 127.0.0.1 --port 8080
# With Ollama
omni-cede --ollama llama3 serve

# Send a message
curl -X POST http://localhost:3000/v1/message \
  -H "Content-Type: application/json" \
  -d '{"channel": "whatsapp", "external_id": "+447123456789", "text": "Hello!"}'

# Health check
curl http://localhost:3000/v1/health

# List sessions for a user
curl http://localhost:3000/v1/sessions/<user_id>

# Stats
curl http://localhost:3000/v1/stats
```

### With Auth

```bash
# Start with auth enabled
API_KEY=my-secret-key ANTHROPIC_API_KEY=sk-ant-... omni-cede serve

# Requests require the header
curl -X POST http://localhost:3000/v1/message \
  -H "Content-Type: application/json" \
  -H "x-api-key: my-secret-key" \
  -d '{"channel": "telegram", "external_id": "12345678", "text": "Hello!"}'
```

## API Reference

### `POST /v1/message`

Send a message from any channel. The server resolves the user's identity, gets or creates a session, runs the agent, and returns the reply.

**Request:**
```json
{
  "channel": "whatsapp",
  "external_id": "+447123456789",
  "text": "What did we discuss yesterday?"
}
```

**Response:**
```json
{
  "reply": "Yesterday we discussed the new API design...",
  "user_id": "a1b2c3d4-...",
  "session_id": "e5f6g7h8-..."
}
```

### `GET /v1/health`

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### `GET /v1/sessions/:user_id`

```json
[
  {
    "session_id": "e5f6g7h8-...",
    "channel": "whatsapp",
    "created_at": 1711324800,
    "turn_count": 42,
    "last_active": 1711411200
  }
]
```

### `GET /v1/stats`

```json
{
  "nodes": 1234,
  "edges": 5678,
  "by_kind": {"fact": 200, "soul": 1, "session": 15, "...": "..."},
  "managed_sessions": 15,
  "total_turns": 342
}
```

## How Identity Works

```
WhatsApp +447123456789  -+
                          |-> user_id: a1b2c3d4
Telegram @johndoe       -+    (linked via identity layer)
```

When a message arrives, the identity layer:
1. Looks up `(channel, external_id)` in the `channel_mappings` table
2. If found, returns the existing internal user
3. If not, creates a new user and mapping

You can link multiple channels to one user via the identity API.

## How Sessions Work

Each (user, channel) pair gets its own session. This means:

- **Recency window is channel-scoped** — "stop using big words" on WhatsApp only affects WhatsApp's briefing
- **Semantic recall is global** — facts learned on Telegram are available when the user asks on WhatsApp
- **Sessions persist** — reconnecting to the same channel resumes the same session

## Architecture

```
+---------------------------------------------+
|                 omni-cede                    |
+-----------+-----------+---------------------+
|  HTTP API |  Identity |  Session Manager    |
| (axum)    | (channel  | (one per user +     |
|           |  mapping) |  channel pair)      |
+-----------+-----------+---------------------+
|                  cede core                   |
+---------+----------+---------+--------------+
|  recall | briefing |  tools  |    agent     |
| (HNSW + | (scored  | (custom |  (loop +    |
|  graph) |  context)|  + std) |  subagent)  |
+---------+----------+---------+--------------+
|            graph + memory                    |
|       (BFS, scoring, decay)                  |
+---------+------------------------------------+
|  HNSW   |         SQLite                     |
| (2-tier)|  (WAL, bundled rusqlite)           |
+---------+------------------------------------+
|            fastembed                          |
|      (BAAI/bge-small-en-v1.5)                |
+----------------------------------------------+
```

## CLI Commands

omni-cede retains all of cede's CLI commands and adds `serve`:

```bash
omni-cede serve                    # Start HTTP API server (0.0.0.0:3000)
omni-cede serve --port 8080        # Custom port
omni-cede chat                     # Interactive CLI chat
omni-cede ask "question"           # Single query
omni-cede graph explore            # TUI graph explorer
omni-cede graph overview           # Graph visualization
omni-cede memory stats             # Memory statistics
omni-cede memory search "query"    # Semantic search
omni-cede soul show                # View identity
omni-cede doctor                   # Health check
omni-cede consolidate              # Trust propagation
omni-cede init                     # Initialize DB + download model
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key (*or use `--ollama`) |
| `ANTHROPIC_MODEL` | No | Model override (default: `claude-sonnet-4-20250514`) |
| `API_KEY` | No | If set, requires `x-api-key` header on all requests |
| `RUST_LOG` | No | Tracing filter (default: `omni_cede=info,tower_http=info`) |

## Staying Updated

omni-cede tracks cede as `upstream`. To pull improvements:

```bash
git fetch upstream
git merge upstream/master
```

## Dependencies

Everything from cede, plus:

| Crate | Purpose |
|-------|---------|
| `axum` 0.8 | HTTP framework |
| `tower-http` 0.6 | CORS + request tracing middleware |
| `tracing` + `tracing-subscriber` | Structured logging |

## Tests

```bash
# Run all 28 tests
cargo test -- --test-threads=1
```

## License

MIT