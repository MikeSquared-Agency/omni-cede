use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::task::JoinSet;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::hnsw::VectorIndex;
use crate::llm::LlmClient;
use crate::memory;
use crate::tools::ToolRegistry;
use crate::types::*;

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
            12,
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
                    // Store fact from response
                    let fact = Node::fact_from_response(&response.text, &session_id);
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

        Ok("Reached iteration limit without final answer.".into())
    }

    /// Run a single turn within an ongoing chat session.
    ///
    /// Each turn is self-contained: the user's input is stored as a `UserInput`
    /// node in the graph, a fresh briefing is built by semantic recall (so prior
    /// turns that are relevant surface naturally), and the LLM receives only
    /// `[system(briefing), user(input)]` — no growing message history.
    ///
    /// Tool-call loops use a temporary message vec within the turn.
    pub async fn run_turn(
        &self,
        session_id: &NodeId,
        input: &str,
    ) -> Result<String> {
        // 1. Store the user's input as a UserInput node in the graph
        let user_node = Node::new(NodeKind::UserInput, input)
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
        //    Prior UserInput nodes and Fact responses that are relevant will
        //    surface naturally through HNSW recall.
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
            16, // slightly more nodes to capture conversation context
        )
        .await?;

        // 3. Fetch recent session nodes (recency window) and merge any that
        //    the semantic search didn't already return.
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
        // Reverse so we go chronological (oldest first) within the section
        for node in recent_nodes.iter().rev() {
            if briefed_ids.contains(&node.id) {
                continue; // already in semantic briefing
            }
            let body = node.body.as_deref().unwrap_or(&node.title);
            let label = match node.kind {
                NodeKind::UserInput => "User",
                _ => "Assistant",
            };
            recency_section.push_str(&format!("- {label}: {body}\n"));
        }

        let mut context_doc = brief.context_doc;
        if !recency_section.is_empty() {
            context_doc.push_str("## Session context (recent)\n");
            context_doc.push_str(&recency_section);
            context_doc.push('\n');
        }

        // 4. Build messages — just system + user, no history
        let mut messages = vec![
            Message::system(context_doc),
            Message::user(input),
        ];

        let mut iter: usize = 0;

        loop {
            iter += 1;

            // Write LoopIteration node
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

            // LLM call
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
                StopReason::ToolUse => {
                    // Tool calls stay in the temporary messages vec for this turn
                    if let Some(raw) = response.raw_content.clone() {
                        messages.push(Message::assistant_raw(raw));
                    } else {
                        messages.push(Message::assistant(&response.text));
                    }

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

                    if tool_results.len() == 1 {
                        let (id, output) = tool_results.into_iter().next().unwrap();
                        messages.push(Message::tool_result_block(&id, &output));
                    } else {
                        messages.push(Message::multi_tool_result_block(tool_results));
                    }
                }
                StopReason::EndTurn | StopReason::MaxTokens => {
                    // Store the response as a Fact node in the graph
                    let fact = Node::fact_from_response(&response.text, session_id);
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
            }

            if iter >= self.config.max_iterations {
                break;
            }
        }

        Ok("Reached iteration limit without final answer.".into())
    }
}
