# Omnichannel Integration Plan — omni-cede

## Inspiration

This plan takes direct inspiration from [OpenClaw](https://github.com/openclaw/openclaw), a TypeScript personal AI assistant that supports 20+ messaging channels through a **Gateway + Plugin** architecture. We adapt their core patterns to Rust and our graph-memory engine.

### What OpenClaw does right (and what we're stealing)

1. **Gateway as single control plane** — one process handles all channels, sessions, tools, and events. Channels connect TO the gateway, not the other way around.
2. **Channel = Plugin** — each channel is a self-contained extension implementing a standard contract (`channel-contract.ts`). Adding WhatsApp doesn't touch Telegram code.
3. **Plugin SDK** — shared helpers for the hard parts: pairing, allowlists, reply pipelines, typing indicators, media handling.
4. **Hooks pipeline** — lifecycle hooks (`before_dispatch`, `after_tool_call`, `session:patch`) let plugins intercept and transform messages at well-defined points.
5. **Session isolation with cross-channel knowledge** — each channel gets its own session, but the semantic search layer (their "context engine") spans everything.

### What we already have (our advantage)

- **Graph-native sessions** — our `run_turn()` already builds a fresh briefing per turn using HNSW semantic search + recency window. OpenClaw does growing message arrays then "compacts" them. We don't need that.
- **Identity resolution** — our `identity::resolve_user()` already maps (channel, external_id) → internal user_id.
- **Session manager** — our `session::get_or_create()` already scopes sessions to (user_id, channel).
- **HTTP API** — our axum server already handles `POST /v1/message` with the full identity→session→agent pipeline.

We just need to add the **channel adapter layer** — the part that connects real messaging platforms to our existing `/v1/message` pipeline.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                  omni-cede                       │
                    │                                                  │
  WhatsApp ──┐     │  ┌──────────────────────────────────────────┐   │
  Telegram ──┤     │  │            Channel Registry               │   │
  Discord  ──┤────▶│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐  │   │
  Slack    ──┤     │  │  │WhatsApp │ │Telegram │ │ Discord  │  │   │
  WebChat  ──┤     │  │  │ Adapter │ │ Adapter │ │ Adapter  │  │   │
  Webhook  ──┘     │  │  └────┬────┘ └────┬────┘ └────┬─────┘  │   │
                    │  │       │           │           │         │   │
                    │  └───────┼───────────┼───────────┼─────────┘   │
                    │          ▼           ▼           ▼              │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │          Inbound Pipeline                 │   │
                    │  │  normalize → identity → session → hooks  │   │
                    │  └──────────────────┬───────────────────────┘   │
                    │                     ▼                           │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │          Agent (run_turn)                 │   │
                    │  │  briefing → HNSW recall → LLM → tools    │   │
                    │  └──────────────────┬───────────────────────┘   │
                    │                     ▼                           │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │         Outbound Pipeline                 │   │
                    │  │  hooks → chunking → rate-limit → send    │   │
                    │  └──────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

---

## Phase 1: Channel Trait & Registry

### The `Channel` trait

Every messaging platform adapter implements one trait:

```rust
// src/channels/trait.rs

#[async_trait]
pub trait Channel: Send + Sync + 'static {
    /// Unique channel identifier, e.g. "whatsapp", "telegram", "discord"
    fn id(&self) -> &str;

    /// Human-readable name
    fn display_name(&self) -> &str;

    /// Start the channel adapter (connect to APIs, start polling/webhooks)
    async fn start(&self, ctx: ChannelContext) -> Result<()>;

    /// Stop gracefully
    async fn stop(&self) -> Result<()>;

    /// Send a message back to a user on this channel
    async fn send(&self, target: &OutboundTarget, message: OutboundMessage) -> Result<()>;

    /// Health check — is the channel connection alive?
    async fn health(&self) -> ChannelHealth;

    /// Channel-specific configuration schema (for validation)
    fn config_schema(&self) -> serde_json::Value { serde_json::json!({}) }

    /// Optional: typing indicator support
    async fn send_typing(&self, _target: &OutboundTarget) -> Result<()> { Ok(()) }

    /// Optional: message editing (for streaming responses)
    async fn edit_message(&self, _msg_id: &str, _new_text: &str) -> Result<()> {
        Err(CortexError::Unsupported("edit not supported on this channel".into()))
    }

    /// Optional: media support
    fn supports_media(&self) -> bool { false }
    async fn send_media(&self, _target: &OutboundTarget, _media: MediaPayload) -> Result<()> {
        Err(CortexError::Unsupported("media not supported".into()))
    }
}
```

### Supporting types

```rust
// src/channels/types.rs

/// Context passed to channels on startup — gives them access to the inbound pipeline
pub struct ChannelContext {
    /// Call this when a message arrives from the channel
    pub inbound_tx: tokio::sync::mpsc::Sender<InboundEnvelope>,
    /// Shared app state for identity/session resolution
    pub db: Db,
    /// Channel-specific config section
    pub config: serde_json::Value,
}

/// A normalized inbound message from any channel
pub struct InboundEnvelope {
    pub channel: String,
    pub external_id: String,
    pub sender_name: Option<String>,
    pub text: String,
    pub media: Option<MediaPayload>,
    pub reply_to: Option<String>,       // message ID being replied to
    pub group_id: Option<String>,       // if this is a group message
    pub raw: serde_json::Value,         // channel-specific raw payload
    pub timestamp: i64,
}

/// Where to send a reply
pub struct OutboundTarget {
    pub channel: String,
    pub external_id: String,
    pub group_id: Option<String>,
    pub reply_to_message_id: Option<String>,
}

/// An outbound message — text, media, or both
pub struct OutboundMessage {
    pub text: String,
    pub media: Option<MediaPayload>,
    pub metadata: serde_json::Value,
}

pub struct MediaPayload {
    pub kind: MediaKind,
    pub data: Vec<u8>,
    pub mime_type: String,
    pub filename: Option<String>,
}

pub enum MediaKind {
    Image,
    Audio,
    Video,
    Document,
}

pub enum ChannelHealth {
    Connected,
    Degraded(String),
    Disconnected(String),
}
```

### Channel Registry

```rust
// src/channels/registry.rs

pub struct ChannelRegistry {
    channels: HashMap<String, Arc<dyn Channel>>,
    inbound_tx: mpsc::Sender<InboundEnvelope>,
    inbound_rx: mpsc::Receiver<InboundEnvelope>,
}

impl ChannelRegistry {
    pub fn new() -> Self { ... }

    /// Register a channel adapter
    pub fn register(&mut self, channel: Arc<dyn Channel>) { ... }

    /// Start all registered channels
    pub async fn start_all(&self, db: &Db, config: &Config) -> Result<()> { ... }

    /// Stop all channels
    pub async fn stop_all(&self) -> Result<()> { ... }

    /// Get a channel by ID (for outbound routing)
    pub fn get(&self, id: &str) -> Option<Arc<dyn Channel>> { ... }

    /// List all registered channels with health status
    pub async fn health_all(&self) -> Vec<(String, ChannelHealth)> { ... }
}
```

### New file structure

```
src/channels/
    mod.rs          # re-exports, Channel trait
    types.rs        # InboundEnvelope, OutboundTarget, OutboundMessage, etc.
    registry.rs     # ChannelRegistry
    pipeline.rs     # inbound/outbound message processing pipeline
    webhook.rs      # Generic webhook channel (for platforms that POST to us)
    whatsapp.rs     # WhatsApp adapter (via Baileys/whatsapp-web.js sidecar or webhook)
    telegram.rs     # Telegram adapter (Bot API, long polling or webhook)
    discord.rs      # Discord adapter (serenity or webhook)
    slack.rs        # Slack adapter (Bolt-style webhook)
    webchat.rs      # Built-in WebSocket webchat (served from the gateway)
```

---

## Phase 2: Inbound / Outbound Pipeline

Inspired by OpenClaw's `before_dispatch` and reply pipeline hooks.

### Inbound Pipeline

When a message arrives from any channel:

```
InboundEnvelope
    │
    ├─ 1. Normalize: trim whitespace, detect /commands
    ├─ 2. Security: check allowlist for this (channel, sender)
    ├─ 3. Identity: resolve_user(channel, external_id) → user_id
    ├─ 4. Session: get_or_create(user_id, channel) → session_id
    ├─ 5. Hook: before_agent (plugins can modify or reject)
    ├─ 6. Agent: run_turn(session_id, text) → reply
    ├─ 7. Hook: after_agent (plugins can modify reply)
    ├─ 8. Record: session::record_turn()
    └─ 9. Outbound: route reply back to the originating channel
```

### Outbound Pipeline

```
OutboundMessage
    │
    ├─ 1. Hook: before_send (rate-limiting, logging)
    ├─ 2. Chunk: split long messages per channel limits
    │      (WhatsApp: 65536, Telegram: 4096, Discord: 2000, Slack: 40000)
    ├─ 3. Send: channel.send(target, chunk)
    ├─ 4. Typing: send typing indicator between chunks
    └─ 5. Hook: after_send (delivery tracking)
```

### Hooks System

```rust
// src/channels/hooks.rs

#[async_trait]
pub trait ChannelHook: Send + Sync {
    /// Called before the message is sent to the agent. Return Err to reject.
    async fn before_agent(&self, _env: &mut InboundEnvelope) -> Result<()> { Ok(()) }

    /// Called after the agent produces a reply. Can modify the reply text.
    async fn after_agent(&self, _env: &InboundEnvelope, _reply: &mut String) -> Result<()> { Ok(()) }

    /// Called before sending a message on a channel.
    async fn before_send(&self, _target: &OutboundTarget, _msg: &mut OutboundMessage) -> Result<()> { Ok(()) }

    /// Called after successful send.
    async fn after_send(&self, _target: &OutboundTarget, _msg: &OutboundMessage) -> Result<()> { Ok(()) }
}
```

---

## Phase 3: Channel Adapters (Priority Order)

### 3a. Webhook Channel (generic)

The simplest adapter — any platform that can POST JSON to us. Our existing `POST /v1/message` is basically this already. We generalize it:

```
POST /v1/channels/webhook/inbound
{
  "channel": "custom",
  "external_id": "user123",
  "text": "Hello",
  "callback_url": "https://my-app.com/reply"  // optional: where to POST the reply
}
```

This lets any system integrate without a dedicated adapter.

**Effort:** Small — refactor existing `/v1/message` into the pipeline pattern.

### 3b. Telegram

Telegram is the easiest real channel — clean Bot API, no unofficial hacks.

- **Inbound:** Long polling via `getUpdates` or webhook mode (Telegram POSTs to our `/v1/channels/telegram/webhook`)
- **Outbound:** `sendMessage`, `sendPhoto`, `editMessageText` (for streaming)
- **Features:** Typing indicators, inline keyboards, message editing, groups (mention gating), media
- **Auth:** `TELEGRAM_BOT_TOKEN` env var
- **Crate:** `reqwest` (just HTTP calls to `api.telegram.org`)
- **Config:**
  ```json
  {
    "channels": {
      "telegram": {
        "bot_token": "...",
        "mode": "polling",           // or "webhook"
        "webhook_url": "https://...",
        "allow_from": ["123456789"],  // telegram user IDs, "*" for all
        "groups": { "*": { "require_mention": true } }
      }
    }
  }
  ```

**Effort:** Medium — straightforward HTTP API, ~400 lines.

### 3c. Discord

Discord needs a persistent WebSocket (gateway) for real-time events.

- **Inbound:** WS gateway for `MESSAGE_CREATE` events, or slash commands via webhook
- **Outbound:** REST API `POST /channels/{id}/messages`
- **Features:** Threads, embeds, reactions, slash commands, voice channels (future)
- **Auth:** `DISCORD_BOT_TOKEN` env var
- **Crate:** Either `serenity` (full-featured) or raw WS + REST via `tokio-tungstenite` + `reqwest`
- **Config:**
  ```json
  {
    "channels": {
      "discord": {
        "token": "...",
        "allow_from": ["guild_id:channel_id"],
        "dm_policy": "pairing"
      }
    }
  }
  ```

**Effort:** Medium-High — WS gateway is more complex. Recommend `serenity` crate to handle the protocol.

### 3d. Slack

Slack uses Socket Mode (WebSocket) or Events API (webhook).

- **Inbound:** Socket Mode WS for `message` events, or HTTP webhook for Events API
- **Outbound:** `chat.postMessage`, `chat.update` (for streaming edits)
- **Features:** Threads, blocks (rich formatting), reactions, slash commands
- **Auth:** `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` env vars
- **Crate:** `reqwest` for API calls, `tokio-tungstenite` for Socket Mode
- **Config:**
  ```json
  {
    "channels": {
      "slack": {
        "bot_token": "xoxb-...",
        "app_token": "xapp-...",
        "mode": "socket",
        "allow_from": ["U12345678"],
        "dm_policy": "open"
      }
    }
  }
  ```

**Effort:** Medium — Socket Mode is simpler than Discord's gateway.

### 3e. WhatsApp

WhatsApp is the hardest — no official free API for personal accounts.

**Option A: WhatsApp Cloud API (Business)** — official, requires Meta Business account.
- Inbound: Webhook (Meta POSTs to us)
- Outbound: REST API
- Config: `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_VERIFY_TOKEN`

**Option B: Baileys sidecar** — unofficial, like OpenClaw does.
- Run a Node.js sidecar process that handles the WhatsApp Web protocol
- Communicate via local HTTP/WS between Rust and the sidecar
- More fragile, but works with personal accounts

**Recommendation:** Start with Option A (Cloud API). Add Option B later as an optional sidecar.

**Effort:** Medium (Cloud API) or High (Baileys sidecar).

### 3f. WebSocket WebChat

Built-in web interface served from the gateway itself.

- **Inbound:** WebSocket `ws://host:port/v1/ws/chat`
- **Outbound:** same WebSocket, streaming tokens
- **Features:** Real-time streaming, typing indicators, session management in the browser
- **Auth:** Session token or API key
- **Crate:** `axum` already supports WebSocket upgrades

**Effort:** Medium — WebSocket upgrade + simple web UI.

---

## Phase 4: Configuration System

Unified TOML/JSON config file at `~/.omni-cede/config.toml`:

```toml
[agent]
model = "anthropic/claude-sonnet-4-20250514"

[gateway]
host = "0.0.0.0"
port = 3000
api_key = "sk-..."  # or use OMNI_CEDE_API_KEY env var

[channels.telegram]
enabled = true
bot_token = "123456:ABCDEF"      # or TELEGRAM_BOT_TOKEN env
mode = "polling"                  # "polling" or "webhook"
allow_from = ["*"]

[channels.discord]
enabled = true
token = "MTIz..."                 # or DISCORD_BOT_TOKEN env
dm_policy = "pairing"

[channels.slack]
enabled = false

[channels.whatsapp]
enabled = false

[channels.webchat]
enabled = true                    # always-on by default

[security]
dm_policy = "pairing"             # global default: "open", "pairing", "closed"
```

**Pattern from OpenClaw:** Env vars always override config file values. Channel-specific settings override global defaults.

---

## Phase 5: Security & Access Control

Directly inspired by OpenClaw's DM pairing model:

### Pairing Flow
1. Unknown sender messages the bot on any channel
2. Bot replies with a 6-digit pairing code (stored in DB with expiry)
3. Owner approves: `omni-cede pairing approve <channel> <code>`
4. Sender is added to the persistent allowlist for that channel
5. Future messages are processed normally

### Allowlist Storage
```sql
CREATE TABLE channel_allowlist (
    channel     TEXT NOT NULL,
    external_id TEXT NOT NULL,
    approved_at INTEGER NOT NULL,
    approved_by TEXT,              -- admin user_id who approved
    PRIMARY KEY (channel, external_id)
);
```

### Policies (per-channel, cascading from global)
- `"open"` — process all inbound messages (dev/personal use)
- `"pairing"` — unknown senders get pairing code (default, safe)
- `"closed"` — only pre-approved allowlist members (production)

---

## Phase 6: Observability & Management

### CLI Commands
```
omni-cede serve                        # Start gateway + all enabled channels
omni-cede channels list                # Show all channels and their health
omni-cede channels status telegram     # Detailed status for one channel
omni-cede pairing list                 # Pending pairing requests
omni-cede pairing approve <code>       # Approve a pairing request
omni-cede sessions list                # All active sessions across channels
omni-cede doctor                       # Check config, credentials, connectivity
```

### API Endpoints (additions)
```
GET  /v1/channels                      # List channels + health
GET  /v1/channels/:id/status           # Detailed channel status
POST /v1/channels/:id/send             # Send a message TO a channel (admin)
GET  /v1/pairing                       # Pending pairing requests
POST /v1/pairing/:code/approve         # Approve pairing
```

### Metrics (via stats endpoint)
- Messages processed per channel per hour
- Average response latency per channel
- Channel uptime/reconnection count
- Session count per channel

---

## Implementation Order

| Phase | What | New Files | Est. Lines | Priority |
|-------|------|-----------|------------|----------|
| 1a | Channel trait + types | `channels/{mod,types}.rs` | ~200 | **NOW** |
| 1b | Channel registry | `channels/registry.rs` | ~150 | **NOW** |
| 1c | Inbound/outbound pipeline | `channels/pipeline.rs` | ~300 | **NOW** |
| 1d | Hooks system | `channels/hooks.rs` | ~100 | **NOW** |
| 2a | Config system (TOML) | `config_file.rs` | ~200 | **NOW** |
| 2b | Webhook channel (generic) | `channels/webhook.rs` | ~100 | **NOW** |
| 3a | Telegram adapter | `channels/telegram.rs` | ~400 | **NEXT** |
| 3b | Discord adapter | `channels/discord.rs` | ~500 | **NEXT** |
| 3c | WebSocket WebChat | `channels/webchat.rs` | ~350 | **NEXT** |
| 3d | Slack adapter | `channels/slack.rs` | ~400 | **LATER** |
| 3e | WhatsApp Cloud API | `channels/whatsapp.rs` | ~450 | **LATER** |
| 4a | Pairing/allowlist | `channels/security.rs` | ~250 | **NEXT** |
| 4b | CLI commands | `cli/mod.rs` additions | ~200 | **NEXT** |
| 5a | observability endpoints | `api/mod.rs` additions | ~150 | **LATER** |
| 5b | Doctor command | `cli/doctor.rs` | ~200 | **LATER** |

**Total new code:** ~4,000 lines across ~15 files

---

## Dependency Additions

```toml
# Phase 1 (trait + pipeline)
# No new deps — uses existing tokio, serde, axum

# Phase 2 (config)
toml = "0.8"                      # Config file parsing

# Phase 3a (Telegram)
# No new deps — uses reqwest (already have it)

# Phase 3b (Discord)
serenity = { version = "0.12", default-features = false, features = ["client", "gateway", "model"] }

# Phase 3c (WebChat)
# No new deps — axum WebSocket support is built-in

# Phase 3d (Slack)
# No new deps — uses reqwest + tokio-tungstenite
tokio-tungstenite = "0.24"        # WebSocket client for Slack Socket Mode

# Phase 3e (WhatsApp)
# No new deps for Cloud API (uses reqwest)
```

---

## Key Design Decisions

### 1. Channels run inside the gateway process (like OpenClaw)
No separate sidecar processes (except WhatsApp Baileys if needed). Each channel adapter is a Rust async task managed by the `ChannelRegistry`. This keeps deployment simple — one binary, one config file.

### 2. All channels share the same inbound pipeline
Every message, regardless of source, flows through the same normalize → identity → session → agent → outbound path. The `Channel` trait only handles platform-specific wire protocol. Business logic stays in the pipeline.

### 3. One session per (user, channel) — cross-channel knowledge via HNSW
Same as now. A WhatsApp session and a Telegram session for the same user are separate (separate recency windows). But HNSW semantic search spans the entire graph — the agent remembers what you said on WhatsApp when you talk on Telegram.

### 4. Feature flags via Cargo features (later)
Eventually, each channel can be a Cargo feature so you only compile what you need:
```toml
[features]
default = ["telegram", "webchat"]
telegram = []
discord = ["dep:serenity"]
slack = ["dep:tokio-tungstenite"]
whatsapp = []
```

### 5. Outbound chunking is channel-aware
Each channel has different message length limits. The outbound pipeline asks the channel for its limit and splits accordingly. OpenClaw does this per-channel — we should too.

---

## What We're NOT Doing (vs OpenClaw)

| OpenClaw feature | Our take |
|-----------------|----------|
| Voice Wake / Talk Mode | Out of scope — we're text-first |
| Canvas / A2UI | Out of scope — no visual workspace |
| Companion apps (macOS/iOS/Android) | Out of scope — server-only |
| Skills registry (ClawHub) | We have tools, not skills |
| Browser control | Out of scope |
| Cron / scheduled messages | Phase 6+ (future) |
| Sandboxed execution | Not needed — we don't run arbitrary code |
| Multi-agent routing | Future — could route channels to different agents |

---

## Next Steps

1. **Build Phase 1** — Channel trait, registry, pipeline, hooks (~750 lines)
2. **Build Phase 2a** — Config file system (~200 lines)
3. **Build Phase 3a** — Telegram adapter as the first real channel
4. **Test end-to-end** — Message on Telegram → agent processes → reply on Telegram
5. **Iterate** — Add Discord, WebChat, then Slack/WhatsApp
