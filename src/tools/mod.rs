use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::process::Command as TokioCommand;
use tokio::sync::RwLock;

use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::{CortexError, Result};
use crate::hnsw::VectorIndex;
use crate::types::*;

/// A tool the agent can call. The handler is an async function that
/// takes JSON input and returns a `ToolResult`.
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub trust: f32,
    pub handler: Arc<
        dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
            + Send
            + Sync,
    >,
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    /// Execute a tool and write ToolCall + Fact nodes to the graph.
    pub async fn execute(
        &self,
        name: &str,
        input: serde_json::Value,
        iter_node: NodeId,
        db: &Db,
        auto_link_tx: &async_channel::Sender<NodeId>,
    ) -> Result<ToolResult> {
        let tool = self
            .get(name)
            .ok_or_else(|| CortexError::Tool(format!("unknown tool: {name}")))?;

        // Validate input against schema before executing
        self.validate_input(name, &input)?;

        let trust = tool.trust;
        let result = (tool.handler)(input.clone()).await?;

        // Write ToolCall node
        let tool_call_node = Node {
            kind: NodeKind::ToolCall,
            title: format!("ToolCall: {name}"),
            body: Some(serde_json::json!({
                "tool": name,
                "input": input,
                "output": &result.output,
                "success": result.success,
            }).to_string()),
            trust_score: trust as f64,
            ..Node::new(NodeKind::ToolCall, format!("ToolCall: {name}"))
        };
        let tc_id = tool_call_node.id.clone();
        db.call({
            let node = tool_call_node.clone();
            move |conn| queries::insert_node(conn, &node)
        })
        .await?;

        // Link ToolCall → LoopIteration via PartOf
        let edge = Edge::new(tc_id.clone(), iter_node, EdgeKind::PartOf);
        db.call(move |conn| queries::insert_edge(conn, &edge)).await?;

        // If success, write Fact derived from tool result
        if result.success {
            let fact = Node::new(NodeKind::Fact, format!("Result: {name}"))
                .with_body(&result.output)
                .with_trust(trust as f64);
            let fact_id = fact.id.clone();
            db.call({
                let fact = fact.clone();
                move |conn| queries::insert_node(conn, &fact)
            })
            .await?;

            let derives = Edge::new(fact_id.clone(), tc_id, EdgeKind::DerivesFrom);
            db.call(move |conn| queries::insert_edge(conn, &derives)).await?;

            // Enqueue for auto-linking
            let _ = auto_link_tx.try_send(fact_id);
        }

        Ok(result)
    }

    /// Validate tool input against its JSON schema.
    pub fn validate_input(&self, name: &str, input: &serde_json::Value) -> Result<()> {
        let tool = self
            .get(name)
            .ok_or_else(|| CortexError::Tool(format!("unknown tool: {name}")))?;

        // Skip validation for tools with no meaningful schema
        if tool.input_schema.is_null() || tool.input_schema.as_object().is_none() {
            return Ok(());
        }

        let validator = jsonschema::validator_for(&tool.input_schema).map_err(|e| {
            CortexError::Tool(format!("invalid schema for tool '{name}': {e}"))
        })?;

        if let Err(e) = validator.validate(&input) {
            return Err(CortexError::Tool(format!(
                "Input validation failed for tool '{name}': {e}"
            )));
        }

        Ok(())
    }

    /// Get a cloneable handler function for a tool (for parallel execution).
    pub fn get_handler(
        &self,
        name: &str,
    ) -> Option<
        Arc<
            dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
                + Send
                + Sync,
        >,
    > {
        self.tools.get(name).map(|t| t.handler.clone())
    }

    /// Record a tool call's graph nodes after parallel execution.
    pub async fn record_tool_call(
        &self,
        name: &str,
        result: &ToolResult,
        iter_node: NodeId,
        db: &Db,
        auto_link_tx: &async_channel::Sender<NodeId>,
    ) -> Result<()> {
        let trust = self.get(name).map(|t| t.trust).unwrap_or(0.5);

        let tool_call_node = Node {
            kind: NodeKind::ToolCall,
            title: format!("ToolCall: {name}"),
            body: Some(serde_json::json!({
                "tool": name,
                "output": &result.output,
                "success": result.success,
            }).to_string()),
            trust_score: trust as f64,
            ..Node::new(NodeKind::ToolCall, format!("ToolCall: {name}"))
        };
        let tc_id = tool_call_node.id.clone();
        db.call({
            let node = tool_call_node;
            move |conn| queries::insert_node(conn, &node)
        })
        .await?;

        let edge = Edge::new(tc_id.clone(), iter_node, EdgeKind::PartOf);
        db.call(move |conn| queries::insert_edge(conn, &edge)).await?;

        if result.success {
            let fact = Node::new(NodeKind::Fact, format!("Result: {name}"))
                .with_body(&result.output)
                .with_trust(trust as f64);
            let fact_id = fact.id.clone();
            db.call({
                let fact = fact;
                move |conn| queries::insert_node(conn, &fact)
            })
            .await?;

            let derives = Edge::new(fact_id.clone(), tc_id, EdgeKind::DerivesFrom);
            db.call(move |conn| queries::insert_edge(conn, &derives)).await?;

            let _ = auto_link_tx.try_send(fact_id);
        }

        Ok(())
    }

    /// Build a JSON schema description of all tools (for LLM system prompt).
    pub fn schema_json(&self) -> serde_json::Value {
        let tools: Vec<serde_json::Value> = self
            .tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        serde_json::Value::Array(tools)
    }

    /// Build Anthropic-format tool definitions for the API.
    pub fn anthropic_tool_defs(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect()
    }
}

// ─── Built-in tools ─────────────────────────────────────

/// Create a registry pre-loaded with the built-in cortex tools.
/// Pass `llm` to enable the `spawn_task` tool (background task loops).
pub fn builtin_registry(
    db: Db,
    embed: EmbedHandle,
    hnsw: Arc<RwLock<VectorIndex>>,
    auto_link_tx: async_channel::Sender<crate::types::NodeId>,
    llm: Option<Arc<dyn crate::llm::LlmClient>>,
    config: crate::config::Config,
) -> ToolRegistry {
    let mut reg = ToolRegistry::new();

    // ── remember: store a memory node into the graph ──
    {
        let db = db.clone();
        let embed = embed.clone();
        let hnsw = hnsw.clone();
        let auto_link_tx = auto_link_tx.clone();
        reg.register(Tool {
            name: "remember".to_string(),
            description: concat!(
                "Store a memory node into long-term graph memory. Choose the right kind:\n",
                "- Soul: core identity, purpose, values, personality traits\n",
                "- Belief: things you believe to be true about the world\n",
                "- Goal: objectives and aspirations\n",
                "- Fact: concrete information, user preferences, learned data\n",
                "- Entity: named things (people, places, projects)\n",
                "- Concept: abstract ideas, categories, frameworks\n",
                "- Decision: choices made and their rationale\n",
                "- Pattern: recurring observations about behavior/interaction\n",
                "- Limitation: known constraints or weaknesses\n",
                "- Capability: skills, abilities, things you can do\n",
                "Identity kinds (Soul, Belief, Goal) default to importance 1.0 and never decay."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the memory (e.g. 'User likes coffee')"
                    },
                    "body": {
                        "type": "string",
                        "description": "Detailed content of the memory"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["Fact", "Entity", "Concept", "Decision", "Soul", "Belief", "Goal", "Pattern", "Limitation", "Capability"],
                        "description": "The kind of memory node to create (default: Fact)"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance 0.0-1.0 (identity kinds default to 1.0, others to 0.5)"
                    }
                },
                "required": ["title", "body"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let embed = embed.clone();
                let hnsw = hnsw.clone();
                let auto_link_tx = auto_link_tx.clone();
                Box::pin(async move {
                    let title = input["title"].as_str().unwrap_or("untitled").to_string();
                    let body = input["body"].as_str().unwrap_or("").to_string();
                    let kind_str = input["kind"].as_str().unwrap_or("Fact");
                    let kind = match kind_str {
                        "Soul" => NodeKind::Soul,
                        "Belief" => NodeKind::Belief,
                        "Goal" => NodeKind::Goal,
                        "Entity" => NodeKind::Entity,
                        "Concept" => NodeKind::Concept,
                        "Decision" => NodeKind::Decision,
                        "Pattern" => NodeKind::Pattern,
                        "Limitation" => NodeKind::Limitation,
                        "Capability" => NodeKind::Capability,
                        _ => NodeKind::Fact,
                    };

                    // Identity kinds get high importance & no decay by default
                    let is_identity = matches!(kind, NodeKind::Soul | NodeKind::Belief | NodeKind::Goal);
                    let importance = input["importance"].as_f64()
                        .unwrap_or(if is_identity { 1.0 } else { 0.5 });

                    // Embed the text
                    let text = format!("{} {}", &title, &body);
                    let embedding = embed.embed(&text).await?;

                    let mut node = Node::new(kind, title.clone())
                        .with_body(&body)
                        .with_trust(0.8);
                    node.importance = importance;
                    if is_identity {
                        node.decay_rate = 0.0;
                    }
                    node.embedding = Some(embedding.clone());
                    let node_id = node.id.clone();

                    // Insert node
                    db.call({
                        let n = node.clone();
                        move |conn| queries::insert_node(conn, &n)
                    })
                    .await?;

                    // Index in HNSW
                    {
                        let mut idx = hnsw.write().await;
                        idx.insert(node_id.clone(), embedding);
                    }

                    // Enqueue for auto-linking
                    let _ = auto_link_tx.try_send(node_id.clone());

                    Ok(ToolResult {
                        output: format!("Remembered [{}]: '{}' (id: {})", kind_str, title, &node_id[..8]),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── recall: semantic search over graph memory ──
    {
        let db = db.clone();
        let embed = embed.clone();
        let hnsw = hnsw.clone();
        reg.register(Tool {
            name: "recall".to_string(),
            description: "Search long-term graph memory for relevant information. Use this to look up things you've been told before, or to find facts, preferences, and decisions. Optionally filter by node kind.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 5)"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["Fact", "Entity", "Concept", "Decision", "Soul", "Belief", "Goal", "Pattern", "Limitation", "Capability"],
                        "description": "Filter results to only this node kind (optional)"
                    }
                },
                "required": ["query"]
            }),
            trust: 1.0,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let embed = embed.clone();
                let hnsw = hnsw.clone();
                Box::pin(async move {
                    let query = input["query"].as_str().unwrap_or("").to_string();
                    let limit = input["limit"].as_u64().unwrap_or(5) as usize;
                    let filter_kinds = input["kind"].as_str()
                        .and_then(|s| NodeKind::from_str_opt(&s.to_lowercase()))
                        .map(|k| vec![k]);

                    let results = crate::memory::recall(
                        &db,
                        &embed,
                        &hnsw,
                        &crate::config::Config::default(),
                        &query,
                        RecallOptions {
                            top_k: limit,
                            filter_kinds,
                            ..Default::default()
                        },
                    )
                    .await?;

                    if results.is_empty() {
                        return Ok(ToolResult {
                            output: "No memories found matching that query.".to_string(),
                            success: true,
                        });
                    }

                    let mut out = String::new();
                    for s in &results {
                        out.push_str(&format!(
                            "- [{}] {} (id: {}, trust: {:.2})\n  {}\n",
                            s.node.kind,
                            s.node.title,
                            &s.node.id[..8],
                            s.node.trust_score,
                            s.node.body.as_deref().unwrap_or(""),
                        ));
                    }
                    Ok(ToolResult {
                        output: out,
                        success: true,
                    })
                })
            }),
        });
    }

    // ── update_memory: modify an existing node ──
    {
        let db = db.clone();
        let embed = embed.clone();
        let hnsw = hnsw.clone();
        let auto_link_tx = auto_link_tx.clone();
        reg.register(Tool {
            name: "update_memory".to_string(),
            description: "Update an existing memory node. Provide the node_id (or a unique prefix) and the fields to change. Only specified fields are updated.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Full node ID or unique prefix (at least 6 chars)"
                    },
                    "title": {
                        "type": "string",
                        "description": "New title (optional)"
                    },
                    "body": {
                        "type": "string",
                        "description": "New body content (optional)"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["Fact", "Entity", "Concept", "Decision", "Soul", "Belief", "Goal", "Pattern", "Limitation", "Capability"],
                        "description": "Change the node kind (optional)"
                    },
                    "importance": {
                        "type": "number",
                        "description": "New importance 0.0-1.0 (optional)"
                    }
                },
                "required": ["node_id"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let embed = embed.clone();
                let hnsw = hnsw.clone();
                let auto_link_tx = auto_link_tx.clone();
                Box::pin(async move {
                    let raw_id = input["node_id"].as_str().unwrap_or("").to_string();
                    if raw_id.len() < 6 {
                        return Ok(ToolResult {
                            output: "Error: node_id must be at least 6 characters.".into(),
                            success: false,
                        });
                    }

                    // Resolve prefix to full ID
                    let full_id = {
                        let rid = raw_id.clone();
                        let matches = db.call(move |conn| queries::find_nodes_by_prefix(conn, &rid)).await?;
                        match matches.len() {
                            0 => return Ok(ToolResult {
                                output: format!("No node found with prefix '{}'", raw_id),
                                success: false,
                            }),
                            1 => matches.into_iter().next().unwrap(),
                            n => return Ok(ToolResult {
                                output: format!("Ambiguous prefix '{}' matches {} nodes. Use a longer ID.", raw_id, n),
                                success: false,
                            }),
                        }
                    };

                    let new_kind = input["kind"].as_str().and_then(|s| NodeKind::from_str_opt(s));
                    let new_title = input["title"].as_str();
                    let new_body = input["body"].as_str();
                    let new_importance = input["importance"].as_f64();

                    // Update DB fields
                    {
                        let id = full_id.clone();
                        let kind = new_kind;
                        let title = new_title.map(|s| s.to_string());
                        let body = new_body.map(|s| s.to_string());
                        db.call(move |conn| {
                            queries::update_node_fields(
                                conn,
                                &id,
                                kind,
                                title.as_deref(),
                                body.as_deref(),
                                new_importance,
                                None,
                            )
                        })
                        .await?;
                    }

                    // If title or body changed, re-embed
                    if new_title.is_some() || new_body.is_some() {
                        let id_for_fetch = full_id.clone();
                        let node = db
                            .call(move |conn| queries::get_node(conn, &id_for_fetch))
                            .await?;
                        if let Some(node) = node {
                            let text = format!("{} {}", node.title, node.body.as_deref().unwrap_or(""));
                            let embedding = embed.embed(&text).await?;
                            let emb_blob: Vec<u8> =
                                bytemuck::cast_slice::<f32, u8>(&embedding).to_vec();
                            let id_for_emb = full_id.clone();
                            db.call(move |conn| {
                                conn.execute(
                                    "UPDATE nodes SET embedding = ?1 WHERE id = ?2",
                                    rusqlite::params![emb_blob, id_for_emb],
                                )?;
                                Ok(())
                            })
                            .await?;
                            let mut idx = hnsw.write().await;
                            idx.insert(full_id.clone(), embedding);
                        }

                        // Re-link after content change
                        let _ = auto_link_tx.try_send(full_id.clone());
                    }

                    Ok(ToolResult {
                        output: format!("Updated node {}", &full_id[..8]),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── delete_memory: remove a node from the graph ──
    {
        let db = db.clone();
        let hnsw = hnsw.clone();
        reg.register(Tool {
            name: "delete_memory".to_string(),
            description: "Delete a memory node and all its edges from the graph. Use the node_id or a unique prefix. This is permanent.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Full node ID or unique prefix (at least 6 chars)"
                    }
                },
                "required": ["node_id"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let hnsw = hnsw.clone();
                Box::pin(async move {
                    let raw_id = input["node_id"].as_str().unwrap_or("").to_string();
                    if raw_id.len() < 6 {
                        return Ok(ToolResult {
                            output: "Error: node_id must be at least 6 characters.".into(),
                            success: false,
                        });
                    }

                    // Resolve prefix
                    let full_id = {
                        let rid = raw_id.clone();
                        let matches = db.call(move |conn| queries::find_nodes_by_prefix(conn, &rid)).await?;
                        match matches.len() {
                            0 => return Ok(ToolResult {
                                output: format!("No node found with prefix '{}'", raw_id),
                                success: false,
                            }),
                            1 => matches.into_iter().next().unwrap(),
                            n => return Ok(ToolResult {
                                output: format!("Ambiguous prefix '{}' matches {} nodes. Use a longer ID.", raw_id, n),
                                success: false,
                            }),
                        }
                    };

                    // Fetch title for confirmation message
                    let title = {
                        let id = full_id.clone();
                        let node = db.call(move |conn| queries::get_node(conn, &id)).await?;
                        node.map(|n| n.title).unwrap_or_else(|| "(unknown)".to_string())
                    };

                    // Delete from DB
                    let id_del = full_id.clone();
                    db.call(move |conn| queries::delete_node(conn, &id_del)).await?;

                    // Remove from HNSW index
                    {
                        let mut idx = hnsw.write().await;
                        idx.remove(&full_id);
                    }

                    Ok(ToolResult {
                        output: format!("Deleted node '{}' ({})", title, &full_id[..8]),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── list_memories: enumerate nodes by kind ──
    {
        let db = db.clone();
        reg.register(Tool {
            name: "list_memories".to_string(),
            description: "List memory nodes in the graph, optionally filtered by kind. Returns node IDs, titles, and metadata. Use this to browse and audit your memory rather than relying solely on semantic search.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["Fact", "Entity", "Concept", "Decision", "Soul", "Belief", "Goal", "Pattern", "Limitation", "Capability", "Session"],
                        "description": "Filter to this node kind (optional — omit to list all)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20, max: 50)"
                    }
                },
                "required": []
            }),
            trust: 1.0,
            handler: Arc::new(move |input| {
                let db = db.clone();
                Box::pin(async move {
                    let kind = input["kind"].as_str()
                        .and_then(|s| NodeKind::from_str_opt(&s.to_lowercase()));
                    let limit = input["limit"].as_u64().unwrap_or(20).min(50) as usize;

                    let nodes = db
                        .call(move |conn| queries::list_nodes_filtered(conn, kind, limit))
                        .await?;

                    if nodes.is_empty() {
                        return Ok(ToolResult {
                            output: "No nodes found.".to_string(),
                            success: true,
                        });
                    }

                    let mut out = format!("{} node(s):\n", nodes.len());
                    for n in &nodes {
                        let ts = chrono::DateTime::from_timestamp(n.created_at, 0)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_else(|| "?".to_string());
                        out.push_str(&format!(
                            "- [{}] {} (id: {}, imp: {:.2}, created: {})\n",
                            n.kind, n.title, &n.id[..8], n.importance, ts,
                        ));
                    }
                    Ok(ToolResult { output: out, success: true })
                })
            }),
        });
    }

    // ── memory_stats: graph health overview ──
    {
        let db = db.clone();
        reg.register(Tool {
            name: "memory_stats".to_string(),
            description: "Get a summary of your memory graph: total nodes, edges, and count by node kind. Use this to understand the overall state of your memory.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            trust: 1.0,
            handler: Arc::new(move |_input| {
                let db = db.clone();
                Box::pin(async move {
                    let by_kind = db
                        .call(|conn| queries::node_count_by_kind(conn))
                        .await?;
                    let total_edges = db
                        .call(|conn| queries::edge_count(conn))
                        .await?;

                    let total_nodes: i64 = by_kind.values().sum();
                    let mut out = format!(
                        "Graph: {} nodes, {} edges\nBy kind:\n",
                        total_nodes, total_edges
                    );
                    let mut kinds: Vec<_> = by_kind.iter().collect();
                    kinds.sort_by(|a, b| b.1.cmp(a.1));
                    for (kind, count) in kinds {
                        out.push_str(&format!("  {}: {}\n", kind, count));
                    }
                    Ok(ToolResult { output: out, success: true })
                })
            }),
        });
    }

    // ── bulk_delete: delete multiple nodes at once ──
    {
        let db = db.clone();
        let hnsw = hnsw.clone();
        reg.register(Tool {
            name: "bulk_delete".to_string(),
            description: concat!(
                "Delete multiple memory nodes at once. Provide an array of node ID prefixes. ",
                "Each prefix must be at least 6 characters. All matched nodes and their edges ",
                "will be permanently removed. Use list_memories or recall first to find the IDs."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of node ID prefixes (at least 6 chars each)"
                    }
                },
                "required": ["node_ids"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let hnsw = hnsw.clone();
                Box::pin(async move {
                    let ids = input["node_ids"].as_array()
                        .ok_or_else(|| CortexError::Tool("node_ids must be an array".into()))?
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>();

                    if ids.is_empty() {
                        return Ok(ToolResult {
                            output: "No IDs provided.".into(),
                            success: false,
                        });
                    }

                    // Resolve all prefixes to full IDs
                    let mut full_ids: Vec<NodeId> = Vec::new();
                    let mut errors: Vec<String> = Vec::new();
                    for raw_id in &ids {
                        if raw_id.len() < 6 {
                            errors.push(format!("'{}' too short (min 6 chars)", raw_id));
                            continue;
                        }
                        let rid = raw_id.clone();
                        let matches = db.call(move |conn| queries::find_nodes_by_prefix(conn, &rid)).await?;
                        match matches.len() {
                            0 => errors.push(format!("'{}' not found", raw_id)),
                            1 => full_ids.push(matches.into_iter().next().unwrap()),
                            n => errors.push(format!("'{}' ambiguous ({} matches)", raw_id, n)),
                        }
                    }

                    // Delete resolved nodes
                    let count = if !full_ids.is_empty() {
                        let fids = full_ids.clone();
                        db.call(move |conn| queries::delete_nodes_bulk(conn, &fids)).await?
                    } else {
                        0
                    };

                    // Remove from HNSW
                    if !full_ids.is_empty() {
                        let mut idx = hnsw.write().await;
                        for id in &full_ids {
                            idx.remove(id);
                        }
                    }

                    let mut out = format!("Deleted {} node(s).", count);
                    if !errors.is_empty() {
                        out.push_str(&format!("\nSkipped: {}", errors.join(", ")));
                    }
                    Ok(ToolResult { output: out, success: count > 0 || errors.is_empty() })
                })
            }),
        });
    }

    // ── purge_session: delete an entire session tree ──
    {
        let db = db.clone();
        let hnsw = hnsw.clone();
        reg.register(Tool {
            name: "purge_session".to_string(),
            description: concat!(
                "Delete an entire session and all its children (iterations, LLM calls, tool calls, ",
                "and derived facts). Use list_memories with kind=Session to find session IDs first. ",
                "This is useful for cleaning up junk or test sessions."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session node ID or unique prefix (at least 6 chars)"
                    }
                },
                "required": ["session_id"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let hnsw = hnsw.clone();
                Box::pin(async move {
                    let raw_id = input["session_id"].as_str().unwrap_or("").to_string();
                    if raw_id.len() < 6 {
                        return Ok(ToolResult {
                            output: "Error: session_id must be at least 6 characters.".into(),
                            success: false,
                        });
                    }

                    // Resolve prefix
                    let full_id = {
                        let rid = raw_id.clone();
                        let matches = db.call(move |conn| queries::find_nodes_by_prefix(conn, &rid)).await?;
                        match matches.len() {
                            0 => return Ok(ToolResult {
                                output: format!("No node found with prefix '{}'", raw_id),
                                success: false,
                            }),
                            1 => matches.into_iter().next().unwrap(),
                            n => return Ok(ToolResult {
                                output: format!("Ambiguous prefix '{}' matches {} nodes.", raw_id, n),
                                success: false,
                            }),
                        }
                    };

                    // Verify it's a session node
                    let node = {
                        let id = full_id.clone();
                        db.call(move |conn| queries::get_node(conn, &id)).await?
                    };
                    match &node {
                        Some(n) if n.kind == NodeKind::Session => {},
                        Some(n) => return Ok(ToolResult {
                            output: format!("Node {} is a {}, not a session.", &full_id[..8], n.kind),
                            success: false,
                        }),
                        None => return Ok(ToolResult {
                            output: format!("Node {} not found.", &full_id[..8]),
                            success: false,
                        }),
                    }

                    // Collect entire session tree
                    let tree_ids = {
                        let sid = full_id.clone();
                        db.call(move |conn| queries::collect_session_tree(conn, &sid)).await?
                    };
                    // Delete all nodes in tree
                    let deleted = {
                        let ids = tree_ids.clone();
                        db.call(move |conn| queries::delete_nodes_bulk(conn, &ids)).await?
                    };

                    // Remove from HNSW
                    {
                        let mut idx = hnsw.write().await;
                        for id in &tree_ids {
                            idx.remove(id);
                        }
                    }

                    let title = node.unwrap().title;
                    Ok(ToolResult {
                        output: format!(
                            "Purged session '{}' ({}) — {} nodes deleted.",
                            title, &full_id[..8], deleted
                        ),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── current_time: return current date/time ──
    reg.register(Tool {
        name: "current_time".to_string(),
        description: "Get the current date and time.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        }),
        trust: 1.0,
        handler: Arc::new(|_input| {
            Box::pin(async move {
                let now = chrono::Local::now();
                Ok(ToolResult {
                    output: now.format("%Y-%m-%d %H:%M:%S %Z").to_string(),
                    success: true,
                })
            })
        }),
    });

    // ── bash: execute shell commands on the host ──
    if config.bash_enabled {
        let blocked = config.bash_blocked_patterns.clone();
        let timeout_secs = config.bash_timeout_secs;
        let max_output = config.bash_max_output_bytes;
        reg.register(Tool {
            name: "bash".to_string(),
            description: concat!(
                "Execute a shell command on the host machine and return its output. ",
                "On Linux/macOS this runs via /bin/sh -c, on Windows via cmd /C. ",
                "Use this for file operations, system inspection, running scripts, ",
                "installing packages, managing services, or any task that requires ",
                "interacting with the operating system. ",
                "Commands have a timeout and dangerous operations are blocked. ",
                "Always prefer single commands; for multi-step work, call bash multiple times."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command (optional, defaults to current dir)"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Override timeout in seconds (optional, max 300)"
                    }
                },
                "required": ["command"]
            }),
            trust: 0.7,
            handler: Arc::new(move |input| {
                let blocked = blocked.clone();
                let timeout_secs = timeout_secs;
                let max_output = max_output;
                Box::pin(async move {
                    let command = input["command"].as_str().unwrap_or("").to_string();
                    if command.is_empty() {
                        return Ok(ToolResult {
                            output: "Error: command is required.".into(),
                            success: false,
                        });
                    }

                    // Safety: check against blocked patterns
                    let cmd_lower = command.to_lowercase();
                    for pattern in &blocked {
                        if cmd_lower.contains(&pattern.to_lowercase()) {
                            return Ok(ToolResult {
                                output: format!(
                                    "Blocked: command matches safety pattern '{}'. This operation is not allowed.",
                                    pattern
                                ),
                                success: false,
                            });
                        }
                    }

                    // Resolve timeout (user override capped at 300s)
                    let timeout = std::time::Duration::from_secs(
                        input["timeout_secs"]
                            .as_u64()
                            .unwrap_or(timeout_secs)
                            .min(300),
                    );

                    // Build the OS-appropriate command
                    let mut cmd = if cfg!(target_os = "windows") {
                        let mut c = TokioCommand::new("cmd");
                        c.args(["/C", &command]);
                        c
                    } else {
                        let mut c = TokioCommand::new("/bin/sh");
                        c.args(["-c", &command]);
                        c
                    };

                    // Set working directory if provided
                    if let Some(dir) = input["working_dir"].as_str() {
                        cmd.current_dir(dir);
                    }

                    // Capture stdout + stderr
                    cmd.stdout(std::process::Stdio::piped());
                    cmd.stderr(std::process::Stdio::piped());

                    // Spawn and await with timeout
                    let child = cmd.spawn();
                    let child = match child {
                        Ok(c) => c,
                        Err(e) => {
                            return Ok(ToolResult {
                                output: format!("Failed to spawn command: {e}"),
                                success: false,
                            });
                        }
                    };

                    let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

                    match result {
                        Ok(Ok(output)) => {
                            let code = output.status.code().unwrap_or(-1);
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);

                            // Combine output, truncate if needed
                            let mut combined = String::new();
                            if !stdout.is_empty() {
                                combined.push_str(&stdout);
                            }
                            if !stderr.is_empty() {
                                if !combined.is_empty() {
                                    combined.push_str("\n--- stderr ---\n");
                                }
                                combined.push_str(&stderr);
                            }
                            if combined.is_empty() {
                                combined = "(no output)".into();
                            }

                            // Truncate to max_output bytes
                            if combined.len() > max_output {
                                combined.truncate(max_output);
                                combined.push_str(&format!(
                                    "\n... [truncated at {} bytes]",
                                    max_output
                                ));
                            }

                            let success = output.status.success();
                            Ok(ToolResult {
                                output: format!(
                                    "[exit code: {}]\n{}",
                                    code, combined
                                ),
                                success,
                            })
                        }
                        Ok(Err(e)) => Ok(ToolResult {
                            output: format!("Command execution error: {e}"),
                            success: false,
                        }),
                        Err(_) => Ok(ToolResult {
                            output: format!(
                                "Command timed out after {} seconds and was killed.",
                                timeout.as_secs()
                            ),
                            success: false,
                        }),
                    }
                })
            }),
        });
    }

    // ── spawn_task: kick off a background autonomous loop ──
    if let Some(llm) = llm {
        let db = db.clone();
        let embed = embed.clone();
        let hnsw = hnsw.clone();
        let auto_link_tx = auto_link_tx.clone();
        let config = config.clone();
        reg.register(Tool {
            name: "spawn_task".to_string(),
            description: concat!(
                "Spawn a background task that runs autonomously. The task gets its own ",
                "agent loop with full tool access (recall, remember, bash, etc.) and writes ",
                "all results directly to the graph. Returns immediately with a task ID — ",
                "the agent does NOT wait for the task to finish. ",
                "Use this for multi-step autonomous work: research, file processing, ",
                "system maintenance, report generation, or any task that would take ",
                "multiple tool calls to complete. ",
                "Results are discoverable via recall once the task finishes."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What the background task should accomplish. Be specific."
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context or constraints (optional)"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Max agent loop iterations (default: 10, max: 25)"
                    }
                },
                "required": ["task"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let db = db.clone();
                let embed = embed.clone();
                let hnsw = hnsw.clone();
                let auto_link_tx = auto_link_tx.clone();
                let llm = llm.clone();
                let config = config.clone();
                Box::pin(async move {
                    let task = input["task"].as_str().unwrap_or("").to_string();
                    let context = input["context"].as_str().unwrap_or("").to_string();
                    let max_iter = input["max_iterations"].as_u64()
                        .unwrap_or(10)
                        .min(25) as usize;

                    if task.is_empty() {
                        return Ok(ToolResult {
                            output: "Error: task is required.".into(),
                            success: false,
                        });
                    }

                    let full_task = if context.is_empty() {
                        task.clone()
                    } else {
                        format!("{task}\n\nContext: {context}")
                    };

                    let task_node = Node::new(
                        NodeKind::BackgroundTask,
                        format!("Task: {}", &task),
                    )
                    .with_body(&format!("Status: running\n\n{full_task}"))
                    .with_importance(0.6);
                    let task_id = task_node.id.clone();
                    db.call({
                        let n = task_node;
                        move |conn| queries::insert_node(conn, &n)
                    })
                    .await?;

                    let bg_task_id = task_id.clone();
                    let bg_db = db.clone();
                    let bg_embed = embed.clone();
                    let bg_hnsw = hnsw.clone();
                    let bg_auto_link_tx = auto_link_tx.clone();
                    let bg_llm = llm.clone();
                    let bg_config = config.clone();

                    tokio::spawn(async move {
                        let bg_tools = builtin_registry(
                            bg_db.clone(),
                            bg_embed.clone(),
                            bg_hnsw.clone(),
                            bg_auto_link_tx.clone(),
                            None,
                            bg_config.clone(),
                        );

                        let mut bg_agent_config = bg_config;
                        bg_agent_config.max_iterations = max_iter;

                        let agent = crate::agent::orchestrator::Agent {
                            db: bg_db.clone(),
                            embed: bg_embed,
                            hnsw: bg_hnsw,
                            config: bg_agent_config,
                            llm: bg_llm,
                            tools: bg_tools,
                            auto_link_tx: bg_auto_link_tx.clone(),
                        };

                        let result = agent.run(&full_task).await;

                        let (status, body) = match result {
                            Ok(answer) => ("completed", format!("Status: completed\n\n{answer}")),
                            Err(e) => ("failed", format!("Status: failed\n\nError: {e}")),
                        };

                        let result_fact = Node::new(
                            NodeKind::Fact,
                            format!("Task result: {}", &task),
                        )
                        .with_body(&body)
                        .with_importance(0.6);
                        let fact_id = result_fact.id.clone();
                        let _ = bg_db
                            .call({
                                let f = result_fact;
                                move |conn| queries::insert_node(conn, &f)
                            })
                            .await;

                        let edge = Edge::new(
                            fact_id.clone(),
                            bg_task_id,
                            EdgeKind::DerivesFrom,
                        );
                        let _ = bg_db
                            .call(move |conn| queries::insert_edge(conn, &edge))
                            .await;

                        let _ = bg_auto_link_tx.try_send(fact_id);

                        eprintln!("[background task {status}]: {task}");
                    });

                    Ok(ToolResult {
                        output: format!(
                            "Background task spawned (id: {}). It will run autonomously and write results to the graph. Use recall to check for results later.",
                            &task_id[..8]
                        ),
                        success: true,
                    })
                })
            }),
        });
    }

    reg
}
