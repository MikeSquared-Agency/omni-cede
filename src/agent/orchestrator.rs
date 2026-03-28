use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::task::JoinSet;

use base64::Engine as _;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::hnsw::VectorIndex;
use crate::llm::LlmClient;
use crate::memory;
use crate::memory::format_timestamp;
use crate::notification_delivery::{NotifEvent, NotifTx};
use crate::tools::ToolRegistry;
use crate::types::*;

/// Base64-encode raw bytes for Anthropic's image block format.
fn base64_encode(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// The agent. Owns the LLM client, tool registry, and a handle to the shared
/// `CortexEmbedded` infrastructure (db, embed, hnsw).
pub struct Agent {
    pub db: Db,
    pub embed: EmbedHandle,
    pub hnsw: Arc<RwLock<VectorIndex>>,
    pub config: Config,
    pub llm: Arc<dyn LlmClient>,
    pub tools: ToolRegistry,
    pub auto_link_tx: async_channel::Sender<NodeId>,
    /// Channel for event-driven notification delivery.
    /// `None` in CLI/TUI mode where proactive delivery is not available.
    pub notif_tx: Option<NotifTx>,
}

impl Agent {
    /// Run the agent loop for a single user input.
    pub async fn run(&self, input: &str) -> Result<String> {
        // Create session node
        let session = Node::session(input);
        let session_id = session.id.clone();
        self.db
            .call({
                let s = session.clone();
                move |conn| queries::insert_node(conn, &s)
            })
            .await?;

        // Build briefing for system prompt
        let now_ts = format_timestamp(crate::types::now_unix());

        // Store user input with timestamp
        let user_node = Node::new(NodeKind::UserInput, format!("[{now_ts}] {input}"))
            .with_body(input)
            .with_importance(0.4)
            .with_decay_rate(0.02);
        let user_node_id = user_node.id.clone();
        self.db
            .call({
                let n = user_node;
                move |conn| queries::insert_node(conn, &n)
            })
            .await?;
        let edge = Edge::new(user_node_id.clone(), session_id.clone(), EdgeKind::PartOf);
        self.db
            .call(move |conn| queries::insert_edge(conn, &edge))
            .await?;
        let _ = self.auto_link_tx.try_send(user_node_id);

        let brief = memory::briefing_with_kinds(
            &self.db,
            &self.embed,
            &self.hnsw,
            &self.config,
            input,
            &[
                NodeKind::Soul,
                NodeKind::Belief,
                NodeKind::Goal,
                NodeKind::Fact,
                NodeKind::Decision,
                NodeKind::Pattern,
                NodeKind::Capability,
                NodeKind::Limitation,
            ],
            self.config.briefing_max_nodes,
        )
        .await?;

        let mut messages = vec![
            Message::system(brief.context_doc),
            Message::user(input),
        ];

        let mut iter: usize = 0;

        loop {
            iter += 1;

            // Write LoopIteration node
            let iter_node = Node::loop_iteration(iter, &session_id);
            let iter_id = iter_node.id.clone();
            self.db
                .call({
                    let n = iter_node.clone();
                    move |conn| queries::insert_node(conn, &n)
                })
                .await?;

            // Link iteration to session
            let edge = Edge::new(iter_id.clone(), session_id.clone(), EdgeKind::PartOf);
            self.db
                .call(move |conn| queries::insert_edge(conn, &edge))
                .await?;

            // LLM call — send tool definitions if any are registered
            let start = Instant::now();
            let tool_defs = self.tools.anthropic_tool_defs();
            let response = if tool_defs.is_empty() {
                self.llm.complete(&messages).await?
            } else {
                self.llm.complete_with_tools(&messages, &tool_defs).await?
            };
            let latency_ms = start.elapsed().as_millis() as u64;

            // Record LlmCall node
            let llm_node = Node {
                kind: NodeKind::LlmCall,
                title: format!("LLM call iter {iter}"),
                body: Some(
                    serde_json::json!({
                        "model": self.llm.model_name(),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_ms": latency_ms,
                    })
                    .to_string(),
                ),
                ..Node::new(NodeKind::LlmCall, format!("LLM call iter {iter}"))
            };
            let llm_id = llm_node.id.clone();
            self.db
                .call({
                    let n = llm_node;
                    move |conn| queries::insert_node(conn, &n)
                })
                .await?;
            let llm_edge = Edge::new(llm_id, iter_id.clone(), EdgeKind::PartOf);
            self.db
                .call(move |conn| queries::insert_edge(conn, &llm_edge))
                .await?;

            match response.stop_reason {
                StopReason::ToolUse => {
                    // Push assistant message with raw content blocks (includes all tool_use blocks)
                    if let Some(raw) = response.raw_content.clone() {
                        messages.push(Message::assistant_raw(raw));
                    } else {
                        messages.push(Message::assistant(&response.text));
                    }

                    // Execute ALL tool calls in parallel and collect results
                    let mut tool_results: Vec<(String, String)> = Vec::new();

                    if response.tool_calls.len() == 1 {
                        let tc = &response.tool_calls[0];
                        let result = self
                            .tools
                            .execute(
                                &tc.name,
                                tc.input.clone(),
                                iter_id.clone(),
                                &self.db,
                                &self.auto_link_tx,
                            )
                            .await?;
                        tool_results.push((tc.id.clone(), result.output));
                    } else {
                        let mut set = JoinSet::new();
                        for tc in &response.tool_calls {
                            // Validate input before spawning parallel handler
                            if let Err(e) = self.tools.validate_input(&tc.name, &tc.input) {
                                tool_results.push((
                                    tc.id.clone(),
                                    format!("Validation error: {e}"),
                                ));
                                continue;
                            }
                            let handler = self.tools.get_handler(&tc.name);
                            let input = tc.input.clone();
                            let id = tc.id.clone();
                            let name = tc.name.clone();
                            if let Some(handler) = handler {
                                set.spawn(async move {
                                    let result = handler(input).await;
                                    (id, name, result)
                                });
                            } else {
                                tool_results.push((
                                    tc.id.clone(),
                                    format!("Error: unknown tool '{}'", tc.name),
                                ));
                            }
                        }
                        while let Some(res) = set.join_next().await {
                            match res {
                                Ok((id, name, Ok(result))) => {
                                    self.tools
                                        .record_tool_call(
                                            &name,
                                            &result,
                                            iter_id.clone(),
                                            &self.db,
                                            &self.auto_link_tx,
                                        )
                                        .await?;
                                    tool_results.push((id, result.output));
                                }
                                Ok((id, _name, Err(e))) => {
                                    tool_results.push((id, format!("Tool error: {e}")));
                                }
                                Err(e) => {
                                    eprintln!("Tool task panicked: {e}");
                                }
                            }
                        }
                    }

                    // Push all tool results in a single user message
                    if tool_results.len() == 1 {
                        let (id, output) = tool_results.into_iter().next().unwrap();
                        messages.push(Message::tool_result_block(&id, &output));
                    } else {
                        messages.push(Message::multi_tool_result_block(tool_results));
                    }
                }
                StopReason::EndTurn | StopReason::MaxTokens => {
                    // Store fact from response with timestamp
                    let resp_ts = format_timestamp(crate::types::now_unix());
                    let fact = Node::fact_from_response(&response.text, &session_id)
                        .with_body(format!("[{resp_ts}] {}", response.text));
                    let fact_id = fact.id.clone();
                    self.db
                        .call({
                            let f = fact;
                            move |conn| queries::insert_node(conn, &f)
                        })
                        .await?;
                    let derives = Edge::new(
                        fact_id.clone(),
                        session_id.clone(),
                        EdgeKind::DerivesFrom,
                    );
                    self.db
                        .call(move |conn| queries::insert_edge(conn, &derives))
                        .await?;
                    let _ = self.auto_link_tx.try_send(fact_id);

                    // Context compaction: extract facts from long conversations
                    if messages.len() > self.config.compaction_threshold {
                        let _ = crate::compact_session(
                            &self.db,
                            &self.embed,
                            &self.hnsw,
                            &self.config,
                            &self.auto_link_tx,
                            &session_id,
                            &messages,
                            self.llm.as_ref(),
                        )
                        .await;
                    }

                    return Ok(response.text);
                }
            }

            // Guard: max iterations
            if iter >= self.config.max_iterations {
                let limit_node = Node::new(NodeKind::Limitation, "Hit max iterations")
                    .with_body(format!(
                        "Task: {}. Stopped at {} iterations.",
                        input, iter
                    ))
                    .with_importance(0.7)
                    .with_decay_rate(0.02);
                self.db
                    .call(move |conn| queries::insert_node(conn, &limit_node))
                    .await?;
                break;
            }
        }

        // Max iterations reached — ask the LLM to summarise with full context
        messages.push(Message::user(
            "You've reached the maximum number of iterations for this task. \
             Summarise what you accomplished so far and let the user know \
             they can ask you to continue if needed. Be concise and natural."
        ));
        let wrap_up = self.llm.complete(&messages).await?;
        Ok(wrap_up.text)
    }

    /// Run a single turn within an ongoing chat session.
    ///
    /// Each turn is self-contained: the user's input is stored as a `UserInput`
    /// node in the graph, a fresh briefing is built by semantic recall (so prior
    /// turns that are relevant surface naturally), and the LLM receives only
    /// `[system(briefing), user(input)]` — no growing message history.
    ///
    /// **Non-blocking design**: The first LLM call runs synchronously so the
    /// user always gets a fast response. If the LLM requests tool calls, they
    /// are executed in a background `tokio::spawn` task which then continues
    /// the LLM loop and stores its final answer as a `BackgroundTask` node.
    /// This means the user can keep chatting while tools run.
    pub async fn run_turn(
        &self,
        session_id: &NodeId,
        input: &str,
        ctx: &TurnContext,
        media: Option<&crate::channels::types::MediaPayload>,
    ) -> Result<String> {
        // 1. Store the user's input as a UserInput node in the graph
        let now_ts = format_timestamp(crate::types::now_unix());
        let user_node = Node::new(NodeKind::UserInput, format!("[{now_ts}] {input}"))
            .with_body(input)
            .with_importance(0.4)
            .with_decay_rate(0.02);
        let user_node_id = user_node.id.clone();

        // Embed and store
        let text = user_node.embed_text();
        let embedding = self.embed.embed(&text).await?;
        let embedding_blob = bytemuck::cast_slice::<f32, u8>(&embedding).to_vec();
        let mut stored_node = user_node.clone();
        stored_node.embedding = Some(embedding.clone());

        self.db
            .call({
                let mut n = stored_node.clone();
                n.embedding = Some(bytemuck::cast_slice::<u8, f32>(&embedding_blob).to_vec());
                move |conn| queries::insert_node(conn, &n)
            })
            .await?;

        // Insert into HNSW for future recall
        {
            let mut index = self.hnsw.write().await;
            index.insert(user_node_id.clone(), embedding);
        }

        // Link UserInput → Session
        let edge = Edge::new(user_node_id.clone(), session_id.to_string(), EdgeKind::PartOf);
        self.db
            .call(move |conn| queries::insert_edge(conn, &edge))
            .await?;

        // Trigger auto-link (connects to related nodes)
        let _ = self.auto_link_tx.try_send(user_node_id);

        // 2. Build a FRESH briefing using the input as semantic query
        let brief = memory::briefing_with_kinds(
            &self.db,
            &self.embed,
            &self.hnsw,
            &self.config,
            input,
            &[
                NodeKind::Soul,
                NodeKind::Belief,
                NodeKind::Goal,
                NodeKind::Fact,
                NodeKind::UserInput,
                NodeKind::Decision,
                NodeKind::Pattern,
                NodeKind::Capability,
                NodeKind::Limitation,
            ],
            self.config.briefing_max_nodes,
        )
        .await?;

        // 3. Fetch recent session nodes (recency window)
        let recency_window = self.config.session_recency_window;
        let briefed_ids: std::collections::HashSet<String> =
            brief.nodes.iter().map(|sn| sn.node.id.clone()).collect();
        let recent_nodes = self
            .db
            .call({
                let sid = session_id.to_string();
                move |conn| queries::get_recent_session_nodes(conn, &sid, recency_window)
            })
            .await?;
        let mut recency_section = String::new();
        for node in recent_nodes.iter().rev() {
            if briefed_ids.contains(&node.id) {
                continue;
            }
            let body = node.body.as_deref().unwrap_or(&node.title);
            let label = match node.kind {
                NodeKind::UserInput => "User",
                _ => "Assistant",
            };
            let ts = format_timestamp(node.created_at);
            let meta = memory::node_metadata_label(node);
            recency_section.push_str(&format!("- [{ts}] ({meta}) {label}: {body}\n"));
        }

        let mut context_doc = brief.context_doc;
        if !recency_section.is_empty() {
            context_doc.push_str("## Session context (recent)\n");
            context_doc.push_str(&recency_section);
            context_doc.push('\n');
        }

        // ── Channel awareness ───────────────────────────
        {
            let sender = ctx.sender_name.as_deref().unwrap_or("someone");
            let where_str = if ctx.is_group { "a group chat" } else { "a direct message" };
            context_doc.push_str(&format!(
                "## Current conversation\nYou are talking to **{}** via **{}** ({}).\n\n",
                sender, ctx.channel, where_str,
            ));
            context_doc.push_str(
                "When you need to use tools to fulfil a request, always include a brief, \
                 natural acknowledgment in your response text so the user knows you're on it. \
                 Keep it short and human — e.g. \"Let me look into that\" or \"Sure, one sec.\" \
                 Your background workers will handle the tools and you'll be briefed on the \
                 results, which will then be proactively sent to the user.\n\n",
            );
        }

        // ── Pending notifications (background task results) ─
        let session_for_notif = session_id.to_string();
        let pending = self.db.call(move |conn| {
            queries::get_pending_notification_nodes(conn, &session_for_notif)
        }).await?;

        if !pending.is_empty() {
            context_doc.push_str("## Updates while you were away\n");
            context_doc.push_str("The following background tasks finished since your last message. Mention these to the user naturally:\n");
            let mut delivered_ids: Vec<String> = Vec::new();
            for node in &pending {
                let rel = memory::relative_time(node.created_at);
                context_doc.push_str(&format!("- ({}) {}\n", rel, node.title));
                delivered_ids.push(node.id.clone());
            }
            context_doc.push('\n');

            // Mark as delivered (touch increments access_count from 0 to 1+)
            if !delivered_ids.is_empty() {
                self.db.call(move |conn| {
                    queries::touch_nodes(conn, &delivered_ids)
                }).await?;
            }
        }

        // 4. Build messages — just system + user (+ optional image), no history
        let user_msg = if let Some(media) = media {
            if media.kind == crate::channels::types::MediaKind::Image {
                let b64 = base64_encode(&media.data);
                Message::user_with_image(input, &b64, &media.mime_type)
            } else {
                // Non-image media: mention it in text
                let label = format!("{} [attached {} file: {}]",
                    input,
                    format!("{:?}", media.kind).to_lowercase(),
                    media.filename.as_deref().unwrap_or("file"),
                );
                Message::user(&label)
            }
        } else {
            Message::user(input)
        };
        // Clone context_doc before it's moved into messages — needed for
        // the acknowledgment LLM call if the model returns tool calls with
        // no accompanying text.
        let context_doc_for_ack = context_doc.clone();
        let messages = vec![
            Message::system(context_doc),
            user_msg,
        ];

        // 5. First LLM call (synchronous — the user waits for this one)
        let iter: usize = 1;
        let iter_node = Node::loop_iteration(iter, session_id);
        let iter_id = iter_node.id.clone();
        self.db
            .call({
                let n = iter_node.clone();
                move |conn| queries::insert_node(conn, &n)
            })
            .await?;
        let edge = Edge::new(iter_id.clone(), session_id.to_string(), EdgeKind::PartOf);
        self.db
            .call(move |conn| queries::insert_edge(conn, &edge))
            .await?;

        let start = Instant::now();
        let tool_defs = self.tools.anthropic_tool_defs();
        let response = if tool_defs.is_empty() {
            self.llm.complete(&messages).await?
        } else {
            self.llm.complete_with_tools(&messages, &tool_defs).await?
        };
        let latency_ms = start.elapsed().as_millis() as u64;

        // Record LlmCall node
        let llm_node = Node {
            kind: NodeKind::LlmCall,
            title: format!("LLM call turn iter {iter}"),
            body: Some(
                serde_json::json!({
                    "model": self.llm.model_name(),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_ms": latency_ms,
                })
                .to_string(),
            ),
            ..Node::new(NodeKind::LlmCall, format!("LLM call turn iter {iter}"))
        };
        let llm_id = llm_node.id.clone();
        self.db
            .call({
                let n = llm_node;
                move |conn| queries::insert_node(conn, &n)
            })
            .await?;
        let llm_edge = Edge::new(llm_id, iter_id.clone(), EdgeKind::PartOf);
        self.db
            .call(move |conn| queries::insert_edge(conn, &llm_edge))
            .await?;

        match response.stop_reason {
            StopReason::EndTurn | StopReason::MaxTokens => {
                // No tools needed — store and return immediately
                let resp_ts = format_timestamp(crate::types::now_unix());
                let fact = Node::fact_from_response(&response.text, session_id)
                    .with_body(format!("[{resp_ts}] {}", response.text));
                let fact_id = fact.id.clone();
                self.db
                    .call({
                        let f = fact;
                        move |conn| queries::insert_node(conn, &f)
                    })
                    .await?;
                let derives = Edge::new(
                    fact_id.clone(),
                    session_id.to_string(),
                    EdgeKind::DerivesFrom,
                );
                self.db
                    .call(move |conn| queries::insert_edge(conn, &derives))
                    .await?;
                let _ = self.auto_link_tx.try_send(fact_id);
                return Ok(response.text);
            }
            StopReason::ToolUse => {
                // ── Return immediately, spawn tool execution in background ──
                // Use the LLM's own natural acknowledgment text. If it sent
                // tool calls with no accompanying text, make a quick LLM call
                // with the full briefing to generate a natural acknowledgment.
                let immediate_reply = if response.text.is_empty() {
                    let ack_messages = vec![
                        Message::system(context_doc_for_ack.clone()),
                        Message::user(format!(
                            "The user said: \"{}\"\n\n\
                             You are about to use tools to handle this. \
                             Write a brief, natural acknowledgment (one short sentence) \
                             so the user knows you're working on it. Do NOT describe \
                             what tools you'll use or what you're doing. Just a quick, \
                             human acknowledgment. Stay in character.",
                            input
                        )),
                    ];
                    match self.llm.complete(&ack_messages).await {
                        Ok(ack) if !ack.text.is_empty() => ack.text,
                        _ => response.text.clone(),
                    }
                } else {
                    response.text.clone()
                };

                // Clone everything needed for the background task
                let db = self.db.clone();
                let llm = self.llm.clone();
                let tool_defs = self.tools.anthropic_tool_defs();
                let tools = self.tools.clone();
                let auto_link_tx = self.auto_link_tx.clone();
                let notif_tx = self.notif_tx.clone();
                let session_id = session_id.to_string();
                let panic_session = session_id.clone();
                let pending_calls: Vec<ToolCall> = response.tool_calls.clone();
                let raw_content = response.raw_content.clone();
                let response_text = response.text.clone();
                let max_iterations = self.config.max_iterations;

                // Spawn background task for tool execution + continuation
                let handle = tokio::spawn(async move {
                    let result = Self::background_tool_loop(
                        db.clone(),
                        llm,
                        tool_defs,
                        tools,
                        pending_calls,
                        raw_content,
                        response_text,
                        messages,
                        session_id.clone(),
                        max_iterations,
                        auto_link_tx.clone(),
                    ).await;

                    // Store the final result as a BackgroundTask node
                    let bg_ts = format_timestamp(crate::types::now_unix());
                    let (bg_title, bg_body, notif_summary) = match &result {
                        Ok(text) => (
                            format!("[{bg_ts}] Background task completed"),
                            format!("[{bg_ts}] {text}"),
                            format!("Background task finished: {}", Self::truncate_summary(text, 120)),
                        ),
                        Err(e) => (
                            format!("[{bg_ts}] Background task failed"),
                            format!("[{bg_ts}] Error: {e}"),
                            format!("A background task ran into a problem: {e}"),
                        ),
                    };
                    let bg_node = Node::new(NodeKind::BackgroundTask, bg_title)
                        .with_body(&bg_body)
                        .with_importance(0.6)
                        .with_decay_rate(0.01);
                    let bg_id = bg_node.id.clone();
                    if let Err(e) = db.call({
                        let n = bg_node;
                        move |conn| queries::insert_node(conn, &n)
                    }).await {
                        tracing::error!("Failed to store background task node: {e}");
                    }
                    let edge = Edge::new(bg_id.clone(), session_id.clone(), EdgeKind::PartOf);
                    if let Err(e) = db.call(move |conn| queries::insert_edge(conn, &edge)).await {
                        tracing::error!("Failed to store background task edge: {e}");
                    }
                    let _ = auto_link_tx.try_send(bg_id.clone());

                    // Write notification node so the user gets informed on next message
                    let notif_node = Node::notification(&notif_summary);
                    let notif_id = notif_node.id.clone();
                    if let Err(e) = db.call({
                        let n = notif_node;
                        move |conn| queries::insert_node(conn, &n)
                    }).await {
                        tracing::error!("Failed to write notification node: {e}");
                    }
                    // Link notification → session via PartOf
                    let notif_edge = Edge::new(notif_id.clone(), session_id.clone(), EdgeKind::PartOf);
                    if let Err(e) = db.call(move |conn| queries::insert_edge(conn, &notif_edge)).await {
                        tracing::error!("Failed to link notification to session: {e}");
                    }
                    // Also link notification → background task node via DerivesFrom
                    let derives = Edge::new(notif_id, bg_id, EdgeKind::DerivesFrom);
                    if let Err(e) = db.call(move |conn| queries::insert_edge(conn, &derives)).await {
                        tracing::error!("Failed to link notification to bg task: {e}");
                    }

                    // Fire event for immediate delivery
                    if let Some(ref tx) = notif_tx {
                        let _ = tx.send(NotifEvent { session_id: session_id.clone() });
                    }

                    if let Err(e) = &result {
                        tracing::error!("Background tool loop failed: {e}");
                    }
                });

                // Monitor for panics in a secondary task
                let panic_db = self.db.clone();
                let panic_sid = panic_session.clone();
                let panic_notif_tx = self.notif_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle.await {
                        tracing::error!("Background task panicked: {e}");
                        let notif_node = Node::notification(
                            &format!("A background task crashed with error: {e}"),
                        );
                        let notif_id = notif_node.id.clone();
                        let _ = panic_db.call({
                            let n = notif_node;
                            move |conn| queries::insert_node(conn, &n)
                        }).await;
                        let edge = Edge::new(notif_id, panic_sid.clone(), EdgeKind::PartOf);
                        let _ = panic_db.call(move |conn| queries::insert_edge(conn, &edge)).await;
                        // Fire event for immediate delivery
                        if let Some(ref tx) = panic_notif_tx {
                            let _ = tx.send(NotifEvent { session_id: panic_sid });
                        }
                    }
                });

                return Ok(immediate_reply);
            }
        }
    }

    /// Truncate text to `max_len` chars, adding "..." if truncated.
    fn truncate_summary(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!("{}...", &text[..max_len])
        }
    }

    /// Execute tool calls and continue the LLM loop in the background.
    ///
    /// This runs after `run_turn` has returned the first response to the user.
    /// It executes all pending tool calls, feeds results back to the LLM, and
    /// continues until the LLM produces a final answer (EndTurn) or hits
    /// max_iterations.
    async fn background_tool_loop(
        db: Db,
        llm: Arc<dyn LlmClient>,
        tool_defs: Vec<serde_json::Value>,
        tools: ToolRegistry,
        pending_calls: Vec<ToolCall>,
        raw_content: Option<serde_json::Value>,
        response_text: String,
        mut messages: Vec<Message>,
        session_id: String,
        max_iterations: usize,
        auto_link_tx: async_channel::Sender<NodeId>,
    ) -> crate::error::Result<String> {
        // Push the assistant's response (with tool_use blocks)
        if let Some(raw) = raw_content {
            messages.push(Message::assistant_raw(raw));
        } else {
            messages.push(Message::assistant(&response_text));
        }

        // Execute pending tool calls using the full registry
        let tool_results = Self::execute_tool_calls(&tools, &pending_calls, &db, &auto_link_tx, &session_id).await;

        // Push tool results
        Self::push_tool_results(&mut messages, tool_results);

        // Continue LLM loop
        let mut iter: usize = 1; // already did iter 1 in run_turn
        loop {
            iter += 1;
            if iter > max_iterations {
                // Max iterations in background — ask the LLM to wrap up
                messages.push(Message::user(
                    "You've reached the maximum number of iterations for this \
                     background task. Summarise what you accomplished and what \
                     remains. Be concise and natural."
                ));
                let wrap_up = llm.complete(&messages).await?;
                return Ok(wrap_up.text);
            }

            let response = if tool_defs.is_empty() {
                llm.complete(&messages).await?
            } else {
                llm.complete_with_tools(&messages, &tool_defs).await?
            };

            match response.stop_reason {
                StopReason::EndTurn | StopReason::MaxTokens => {
                    // Store result in graph
                    let resp_ts = format_timestamp(crate::types::now_unix());
                    let fact = Node::fact_from_response(&response.text, &session_id)
                        .with_body(format!("[{resp_ts}] {}", response.text));
                    let fact_id = fact.id.clone();
                    db.call({
                        let f = fact;
                        move |conn| queries::insert_node(conn, &f)
                    }).await?;
                    let derives = Edge::new(fact_id, session_id, EdgeKind::DerivesFrom);
                    db.call(move |conn| queries::insert_edge(conn, &derives)).await?;
                    return Ok(response.text);
                }
                StopReason::ToolUse => {
                    // More tool calls — execute them and keep going
                    if let Some(raw) = response.raw_content.clone() {
                        messages.push(Message::assistant_raw(raw));
                    } else {
                        messages.push(Message::assistant(&response.text));
                    }

                    let tool_results = Self::execute_tool_calls(&tools, &response.tool_calls, &db, &auto_link_tx, &session_id).await;
                    Self::push_tool_results(&mut messages, tool_results);
                }
            }
        }
    }

    /// Execute a set of tool calls (parallel when >1) and return (id, output) pairs.
    async fn execute_tool_calls(
        tools: &ToolRegistry,
        calls: &[ToolCall],
        db: &Db,
        auto_link_tx: &async_channel::Sender<NodeId>,
        session_id: &str,
    ) -> Vec<(String, String)> {
        let mut results: Vec<(String, String)> = Vec::new();

        if calls.len() == 1 {
            let tc = &calls[0];
            match tools.execute(&tc.name, tc.input.clone(), session_id.to_string(), db, auto_link_tx).await {
                Ok(result) => {
                    tracing::debug!(tool=%tc.name, "background tool completed");
                    results.push((tc.id.clone(), result.output));
                }
                Err(e) => {
                    tracing::warn!(tool=%tc.name, error=%e, "background tool failed");
                    results.push((tc.id.clone(), format!("Tool error: {e}")));
                }
            }
        } else {
            let mut set = JoinSet::new();
            for tc in calls {
                if let Err(e) = tools.validate_input(&tc.name, &tc.input) {
                    results.push((tc.id.clone(), format!("Validation error: {e}")));
                    continue;
                }
                let handler = tools.get_handler(&tc.name);
                let input = tc.input.clone();
                let id = tc.id.clone();
                let name = tc.name.clone();
                if let Some(handler) = handler {
                    set.spawn(async move {
                        let result = handler(input).await;
                        (id, name, result)
                    });
                } else {
                    results.push((tc.id.clone(), format!("Error: unknown tool '{}'", tc.name)));
                }
            }
            while let Some(res) = set.join_next().await {
                match res {
                    Ok((id, name, Ok(result))) => {
                        tracing::debug!(tool=%name, "background tool completed");
                        results.push((id, result.output));
                    }
                    Ok((id, _name, Err(e))) => {
                        results.push((id, format!("Tool error: {e}")));
                    }
                    Err(e) => {
                        tracing::error!("Background tool task panicked: {e}");
                    }
                }
            }
        }

        results
    }

    /// Push tool results into the messages vec.
    fn push_tool_results(messages: &mut Vec<Message>, results: Vec<(String, String)>) {
        if results.len() == 1 {
            let (id, output) = results.into_iter().next().unwrap();
            messages.push(Message::tool_result_block(&id, &output));
        } else {
            messages.push(Message::multi_tool_result_block(results));
        }
    }
}
