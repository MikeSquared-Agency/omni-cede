use serde::{Deserialize, Serialize};
use std::fmt;

// ─── ID types ───────────────────────────────────────────

pub type NodeId = String;
pub type EdgeId = String;

// ─── NodeKind ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    // Knowledge
    Fact,
    Entity,
    Concept,
    Decision,
    // Identity — decay_rate: 0.0, importance: 1.0
    Soul,
    Belief,
    Goal,
    // Conversational — user inputs stored for graph-based recall
    UserInput,
    // Operational — decay fast, low importance
    Session,
    Turn,
    LlmCall,
    ToolCall,
    LoopIteration,
    // Background tasks
    BackgroundTask,
    // Scheduled tasks
    CronJob,
    CronExecution,
    // Dynamic skills / plugins
    Skill,
    // Notifications — delivered via graph, not a separate table
    Notification,
    // Self-model — medium decay
    Pattern,
    Limitation,
    Capability,
}

impl NodeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Entity => "entity",
            Self::Concept => "concept",
            Self::Decision => "decision",
            Self::UserInput => "user_input",
            Self::Soul => "soul",
            Self::Belief => "belief",
            Self::Goal => "goal",
            Self::Session => "session",
            Self::Turn => "turn",
            Self::LlmCall => "llm_call",
            Self::ToolCall => "tool_call",
            Self::LoopIteration => "loop_iteration",
            Self::BackgroundTask => "background_task",
            Self::CronJob => "cron_job",
            Self::CronExecution => "cron_execution",
            Self::Skill => "skill",
            Self::Notification => "notification",
            Self::Pattern => "pattern",
            Self::Limitation => "limitation",
            Self::Capability => "capability",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(Self::Fact),
            "entity" => Some(Self::Entity),
            "concept" => Some(Self::Concept),
            "decision" => Some(Self::Decision),
            "user_input" => Some(Self::UserInput),
            "soul" => Some(Self::Soul),
            "belief" => Some(Self::Belief),
            "goal" => Some(Self::Goal),
            "session" => Some(Self::Session),
            "turn" => Some(Self::Turn),
            "llm_call" => Some(Self::LlmCall),
            "tool_call" => Some(Self::ToolCall),
            "loop_iteration" => Some(Self::LoopIteration),
            "background_task" => Some(Self::BackgroundTask),
            "cron_job" => Some(Self::CronJob),
            "cron_execution" => Some(Self::CronExecution),
            "skill" => Some(Self::Skill),
            "notification" => Some(Self::Notification),
            "pattern" => Some(Self::Pattern),
            "limitation" => Some(Self::Limitation),
            "capability" => Some(Self::Capability),
            _ => None,
        }
    }

    /// Default decay rate for this node kind.
    pub fn default_decay_rate(&self) -> f64 {
        match self {
            // Identity nodes never decay
            Self::Soul | Self::Belief | Self::Goal => 0.0,
            // User inputs decay moderately (they're conversation context)
            Self::UserInput => 0.02,
            // Cron definitions persist like identity
            Self::CronJob | Self::Skill => 0.0,
            // Cron executions decay fast like operational nodes
            Self::CronExecution => 0.05,
            // Notifications decay fast (ephemeral once delivered)
            Self::Notification => 0.05,
            // Operational nodes decay fast
            Self::Session | Self::Turn | Self::LlmCall
            | Self::ToolCall | Self::LoopIteration => 0.05,
            // Self-model nodes decay slowly
            Self::Pattern | Self::Limitation | Self::Capability => 0.005,
            // Everything else: moderate decay
            _ => 0.01,
        }
    }

    /// Default importance for this node kind.
    pub fn default_importance(&self) -> f64 {
        match self {
            Self::Soul | Self::Belief | Self::Goal => 1.0,
            Self::CronJob | Self::Skill => 0.8,
            Self::UserInput => 0.4,
            Self::CronExecution => 0.2,
            Self::Notification => 0.3,
            Self::Session | Self::Turn | Self::LlmCall
            | Self::ToolCall | Self::LoopIteration => 0.2,
            _ => 0.5,
        }
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── EdgeKind ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    RelatesTo,
    Contradicts,
    Supports,
    DerivesFrom,
    PartOf,
    Supersedes,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RelatesTo => "relates_to",
            Self::Contradicts => "contradicts",
            Self::Supports => "supports",
            Self::DerivesFrom => "derives_from",
            Self::PartOf => "part_of",
            Self::Supersedes => "supersedes",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "relates_to" => Some(Self::RelatesTo),
            "contradicts" => Some(Self::Contradicts),
            "supports" => Some(Self::Supports),
            "derives_from" => Some(Self::DerivesFrom),
            "part_of" => Some(Self::PartOf),
            "supersedes" => Some(Self::Supersedes),
            _ => None,
        }
    }
}

impl fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── Node ───────────────────────────────────────────────

pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub title: String,
    pub body: Option<String>,
    pub importance: f64,
    pub trust_score: f64,
    pub access_count: i64,
    pub created_at: i64,
    pub last_access: Option<i64>,
    pub decay_rate: f64,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            kind: NodeKind::Fact,
            title: String::new(),
            body: None,
            importance: 0.5,
            trust_score: 1.0,
            access_count: 0,
            created_at: now_unix(),
            last_access: None,
            decay_rate: 0.01,
            embedding: None,
        }
    }
}

impl Node {
    /// Create a new node with correct defaults for its kind.
    pub fn new(kind: NodeKind, title: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            kind,
            title: title.into(),
            importance: kind.default_importance(),
            decay_rate: kind.default_decay_rate(),
            created_at: now_unix(),
            ..Default::default()
        }
    }

    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_trust(mut self, trust: f64) -> Self {
        self.trust_score = trust;
        self
    }

    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    // ── Constructor helpers ─────────────────────────────

    pub fn session(input: &str) -> Self {
        let preview: String = input.chars().take(60).collect();
        Node::new(NodeKind::Session, format!("Session: {preview}"))
            .with_body(input)
    }

    pub fn loop_iteration(iter: usize, session_id: &NodeId) -> Self {
        let ts = crate::memory::format_timestamp(now_unix());
        Node::new(NodeKind::LoopIteration, format!("[{ts}] Iteration {iter}"))
            .with_body(format!("session:{session_id}"))
    }

    pub fn notification(summary: impl Into<String>) -> Self {
        let s = summary.into();
        let ts = crate::memory::format_timestamp(now_unix());
        Node::new(NodeKind::Notification, format!("[{ts}] {s}"))
    }

    pub fn fact_from_response(text: &str, _session_id: &NodeId) -> Self {
        let title = if text.chars().count() > 80 {
            let s: String = text.chars().take(80).collect();
            format!("{s}…")
        } else {
            text.to_string()
        };
        Node::new(NodeKind::Fact, title).with_body(text)
    }

    /// Text used for embedding: title + body.
    pub fn embed_text(&self) -> String {
        match &self.body {
            Some(b) => format!("{} {}", self.title, b),
            None => self.title.clone(),
        }
    }
}

// ─── Edge ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: EdgeKind,
    pub weight: f64,
    pub created_at: i64,
    pub metadata: Option<String>,
}

impl Edge {
    pub fn new(src: NodeId, dst: NodeId, kind: EdgeKind) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            src,
            dst,
            kind,
            weight: 1.0,
            created_at: now_unix(),
            metadata: None,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_metadata(mut self, meta: impl Into<String>) -> Self {
        self.metadata = Some(meta.into());
        self
    }
}

// ─── Recall & Briefing types ────────────────────────────

#[derive(Debug, Clone)]
pub struct RecallOptions {
    pub top_k: usize,
    pub depth: usize,
    pub min_score: f64,
    pub filter_kinds: Option<Vec<NodeKind>>,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            depth: 2,
            min_score: 0.0,
            filter_kinds: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredNode {
    pub node: Node,
    pub score: f64,
    pub similarity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContradictionPair {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub detected_at: i64,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrustSummary {
    pub min_trust: f64,
    pub max_trust: f64,
    pub mean_trust: f64,
    pub low_trust_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConsolidationReport {
    pub nodes_updated: usize,
    pub contradictions_found: usize,
    pub trust_adjustments: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Briefing {
    pub nodes: Vec<ScoredNode>,
    pub contradictions: Vec<ContradictionPair>,
    pub trust_summary: TrustSummary,
    pub context_doc: String,
}

// ─── LLM types ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_call_id: Option<String>,
    /// Raw content blocks for Anthropic tool-use protocol.
    /// When present, these are sent instead of the `content` string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_blocks: Option<serde_json::Value>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), tool_call_id: None, content_blocks: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), tool_call_id: None, content_blocks: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), tool_call_id: None, content_blocks: None }
    }
    pub fn tool_result(content: impl Into<String>) -> Self {
        Self { role: Role::Tool, content: content.into(), tool_call_id: None, content_blocks: None }
    }
    /// Create an assistant message with raw content blocks (for tool_use replay).
    pub fn assistant_raw(blocks: serde_json::Value) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_call_id: None,
            content_blocks: Some(blocks),
        }
    }
    /// Create a tool_result message with proper Anthropic content blocks.
    pub fn tool_result_block(tool_use_id: &str, output: &str) -> Self {
        Self {
            role: Role::User,
            content: output.to_string(),
            tool_call_id: Some(tool_use_id.to_string()),
            content_blocks: Some(serde_json::json!([
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                }
            ])),
        }
    }
    /// Create a user message with multiple tool_result blocks.
    pub fn multi_tool_result_block(results: Vec<(String, String)>) -> Self {
        let blocks: Vec<serde_json::Value> = results
            .iter()
            .map(|(id, output)| {
                serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": output,
                })
            })
            .collect();
        Self {
            role: Role::User,
            content: "tool results".to_string(),
            tool_call_id: None,
            content_blocks: Some(serde_json::Value::Array(blocks)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub stop_reason: StopReason,
    pub tool_name: Option<String>,
    pub tool_input: Option<serde_json::Value>,
    pub tool_use_id: Option<String>,
    /// All tool_use calls in this response (for multi-tool invocations).
    pub tool_calls: Vec<ToolCall>,
    /// Raw content blocks from the API (for replaying in conversation).
    pub raw_content: Option<serde_json::Value>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// A single tool invocation from an LLM response.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

// ─── Tool types ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub output: String,
    pub success: bool,
}

// ─── Turn context ───────────────────────────────────────

/// Contextual metadata about the current turn, carried from the channel
/// pipeline into the agent so it knows *who* is talking and *where*.
#[derive(Debug, Clone)]
pub struct TurnContext {
    /// Channel name (e.g. "discord", "telegram", "webchat", "api").
    pub channel: String,
    /// Human-readable display name for the sender, if known.
    pub sender_name: Option<String>,
    /// Internal user ID (from identity resolution).
    pub user_id: String,
    /// True when the message came from a group/channel (not a DM).
    pub is_group: bool,
}

// ─── Model backend ──────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ModelBackend {
    Anthropic { model: String },
    OpenAI { model: String },
    Ollama { model: String, url: String },
    LlamaCpp { model_path: String },
}
