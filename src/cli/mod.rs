use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

mod graph_viz;
mod graph_tui;

#[derive(Parser)]
#[command(name = "omni-cede", about = "Omnichannel self-aware agent with graph memory")]
pub struct Cli {
    /// Path to the SQLite database file.
    #[arg(long, default_value = "omni-cede.db")]
    pub db: String,

    /// Use Ollama as the LLM backend (format: model@url, e.g. llama3@http://localhost:11434)
    #[arg(long)]
    pub ollama: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Interactive chat session
    Chat,

    /// Single query
    Ask {
        query: String,
    },

    /// Memory operations
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Identity management
    Soul {
        #[command(subcommand)]
        action: SoulAction,
    },

    /// Session management
    Sessions {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Visualize the knowledge graph in the terminal
    Graph {
        #[command(subcommand)]
        action: Option<GraphAction>,
    },

    /// Run trust consolidation
    Consolidate,

    /// Check graph health
    Doctor,

    /// Start the HTTP API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value = "3000")]
        port: u16,
    },

    /// Pre-download the embedding model and initialize DB
    Init,
}

#[derive(Subcommand)]
pub enum MemoryAction {
    /// Semantic search
    Search { query: String },
    /// Show a specific node
    Show { node_id: String },
    /// Memory statistics
    Stats,
}

#[derive(Subcommand)]
pub enum SoulAction {
    /// Display identity nodes
    Show,
    /// Edit identity
    Edit,
}

#[derive(Subcommand)]
pub enum SessionAction {
    /// List sessions
    List,
    /// Show a specific session
    Show { session_id: String },
}

#[derive(Subcommand)]
pub enum GraphAction {
    /// Interactive graph explorer (TUI)
    Explore,
    /// Full graph overview
    Overview,
    /// Show ego network around a node (2 hops)
    Ego { node_id: String },
    /// Filter graph by node kind(s), comma-separated (e.g. soul,belief,fact)
    Filter { kinds: String },
}

/// Run the CLI. Called from `src/bin/cortex.rs`.
pub async fn run() -> crate::error::Result<()> {
    let cli = Cli::parse();
    let ollama_spec = cli.ollama.clone();
    let cx = crate::CortexEmbedded::open(&cli.db).await?;

    match cli.command {
        Commands::Serve { host, port } => {
            let llm = build_llm_client(&ollama_spec)?;
            cx.set_llm(llm.clone()).await;
            // Create the event-driven notification channel
            let (notif_tx, notif_rx) = tokio::sync::mpsc::unbounded_channel::<
                crate::notification_delivery::NotifEvent,
            >();

            // Start the cron scheduler with the notif_tx channel
            cx.start_scheduler(Some(notif_tx.clone()));

            let agent = crate::agent::orchestrator::Agent {
                db: cx.db.clone(),
                embed: cx.embed.clone(),
                hnsw: cx.hnsw.clone(),
                config: cx.config.clone(),
                llm: llm.clone(),
                tools: crate::tools::builtin_registry(
                    cx.db.clone(),
                    cx.embed.clone(),
                    cx.hnsw.clone(),
                    cx.auto_link_tx.clone(),
                    Some(llm),
                    cx.config.clone(),
                ).await,
                auto_link_tx: cx.auto_link_tx.clone(),
                notif_tx: Some(notif_tx.clone()),
            };

            let api_key = std::env::var("API_KEY").ok();

            // ── Build the omnichannel pipeline + registry ──
            let mut registry = crate::channels::ChannelRegistry::new(1024);

            // Register built-in passive channels (always available)
            registry.register(std::sync::Arc::new(
                crate::channels::webhook::WebhookChannel::new(),
            )).await;
            let webchat = crate::channels::webchat::WebChatChannel::new();
            let webchat_counter = webchat.connection_counter();
            let webchat_max = webchat.max_connections();
            registry.register(std::sync::Arc::new(webchat)).await;

            // Register active channels only if their tokens are configured
            if std::env::var("TELEGRAM_BOT_TOKEN").is_ok() {
                registry.register(std::sync::Arc::new(
                    crate::channels::telegram::TelegramChannel::new(),
                )).await;
            }
            if std::env::var("DISCORD_BOT_TOKEN").is_ok() {
                registry.register(std::sync::Arc::new(
                    crate::channels::discord::DiscordChannel::new(),
                )).await;
            }

            // Take inbound receiver before moving registry into Arc
            let inbound_rx = registry.take_inbound_rx();

            let registry = std::sync::Arc::new(registry);
            let pipeline = std::sync::Arc::new(
                crate::channels::Pipeline::new(std::sync::Arc::clone(&registry)),
            );

            let state = std::sync::Arc::new(crate::api::AppState {
                cx,
                agent,
                api_key,
                pipeline,
                registry,
                webchat_counter,
                webchat_max,
            });

            let app = crate::api::router(state.clone());
            let addr = format!("{host}:{port}");
            println!("omni-cede API server listening on {addr}");
            if std::env::var("API_KEY").is_err() {
                println!("  WARNING: API_KEY not set — auth disabled (dev mode)");
            }

            // ── Start channel adapters (Telegram polling, etc.) ──
            {
                let mut channel_configs = std::collections::HashMap::new();

                // Telegram: auto-enable if TELEGRAM_BOT_TOKEN is set
                if std::env::var("TELEGRAM_BOT_TOKEN").is_ok() {
                    channel_configs.insert(
                        "telegram".to_string(),
                        serde_json::json!({ "mode": "polling" }),
                    );
                    println!("  Telegram: enabled (polling mode)");
                }

                // Discord: auto-enable if DISCORD_BOT_TOKEN is set
                if std::env::var("DISCORD_BOT_TOKEN").is_ok() {
                    channel_configs.insert(
                        "discord".to_string(),
                        serde_json::json!({}),
                    );
                    println!("  Discord: enabled");
                }

                // Webhook + WebChat are always available (passive channels)
                channel_configs.insert(
                    "webhook".to_string(),
                    serde_json::json!({}),
                );
                channel_configs.insert(
                    "webchat".to_string(),
                    serde_json::json!({}),
                );

                if let Err(e) = state.registry.start_all(&state.cx.db, &channel_configs).await {
                    eprintln!("  WARNING: Channel start error: {e}");
                    eprintln!("  (server will still handle /v1/message requests)");
                }
            }

            // ── Start the pipeline inbound loop (processes messages from channels) ──
            if let Some(rx) = inbound_rx {
                let pipeline_clone = std::sync::Arc::clone(&state.pipeline);
                let db_clone = state.cx.db.clone();
                let agent_clone = std::sync::Arc::new(crate::agent::orchestrator::Agent {
                    db: state.cx.db.clone(),
                    embed: state.cx.embed.clone(),
                    hnsw: state.cx.hnsw.clone(),
                    config: state.cx.config.clone(),
                    llm: state.agent.llm.clone(),
                    tools: crate::tools::builtin_registry(
                        state.cx.db.clone(),
                        state.cx.embed.clone(),
                        state.cx.hnsw.clone(),
                        state.cx.auto_link_tx.clone(),
                        Some(state.agent.llm.clone()),
                        state.cx.config.clone(),
                    ).await,
                    auto_link_tx: state.cx.auto_link_tx.clone(),
                    notif_tx: Some(notif_tx.clone()),
                });
                tokio::spawn(async move {
                    pipeline_clone.run_inbound_loop(rx, db_clone, agent_clone).await;
                });
                println!("  Pipeline inbound loop: started");

                // ── Start event-driven notification delivery loop ──
                {
                    let notif_db = state.cx.db.clone();
                    let notif_inbound_tx = state.registry.clone_inbound_tx();
                    let notif_shutdown = state.cx.shutdown_rx();
                    tokio::spawn(async move {
                        crate::notification_delivery::run(
                            notif_db,
                            notif_inbound_tx,
                            notif_shutdown,
                            notif_rx,
                        )
                        .await;
                    });
                    println!("  Notification delivery: pipeline injection");
                }
            }

            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .map_err(|e| crate::error::CortexError::Config(format!("bind failed: {e}")))?;
            axum::serve(listener, app)
                .await
                .map_err(|e| crate::error::CortexError::Config(format!("server error: {e}")))?;
            Ok(())
        }
        Commands::Init => {
            println!("Database initialized at: {}", cli.db);
            println!("Embedding model ready.");
            println!("Soul seeded.");
            Ok(())
        }

        Commands::Memory { action } => match action {
            MemoryAction::Stats => {
                let (nodes, edges, by_kind) = cx.stats().await?;
                println!("Nodes: {nodes}");
                println!("Edges: {edges}");
                for (kind, count) in &by_kind {
                    println!("  {kind}: {count}");
                }
                Ok(())
            }
            MemoryAction::Search { query } => {
                let results = cx
                    .recall(&query, crate::types::RecallOptions::default())
                    .await?;
                for s in &results {
                    println!(
                        "  [{}] {} — score: {:.3}, trust: {:.2}",
                        s.node.kind, s.node.title, s.score, s.node.trust_score
                    );
                }
                if results.is_empty() {
                    println!("  (no results)");
                }
                Ok(())
            }
            MemoryAction::Show { node_id } => {
                let node = cx
                    .db
                    .call(move |conn| crate::db::queries::get_node(conn, &node_id))
                    .await?;
                match node {
                    Some(n) => {
                        println!("ID:         {}", n.id);
                        println!("Kind:       {}", n.kind);
                        println!("Title:      {}", n.title);
                        println!("Body:       {}", n.body.as_deref().unwrap_or(""));
                        println!("Importance: {:.3}", n.importance);
                        println!("Trust:      {:.3}", n.trust_score);
                        println!("Created:    {}", n.created_at);
                    }
                    None => println!("Node not found."),
                }
                Ok(())
            }
        },

        Commands::Soul { action } => match action {
            SoulAction::Show => {
                let nodes = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Soul))
                    .await?;
                for n in &nodes {
                    println!("[{}] {}", n.kind, n.title);
                    if let Some(ref body) = n.body {
                        println!("  {body}");
                    }
                }
                let beliefs = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Belief))
                    .await?;
                for n in &beliefs {
                    println!("[{}] {}", n.kind, n.title);
                }
                let goals = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Goal))
                    .await?;
                for n in &goals {
                    println!("[{}] {}", n.kind, n.title);
                }
                Ok(())
            }
            SoulAction::Edit => {
                // Launch the graph TUI filtered to Identity nodes with edit mode.
                // Select an identity node and press 'e' to edit.
                // Identity category is index 1 in ALL_CATEGORIES
                graph_tui::run_with_edit(
                    cx.db.clone(),
                    cx.embed.clone(),
                    cx.hnsw.clone(),
                    1, // "Identity" category
                )
                    .await
                    .map_err(|e| crate::error::CortexError::Config(format!("TUI error: {e}")))?;
                Ok(())
            }
        },

        Commands::Sessions { action } => match action {
            SessionAction::List => {
                let sessions = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Session))
                    .await?;
                for s in &sessions {
                    println!("  {} — {}", &s.id[..8], s.title);
                }
                if sessions.is_empty() {
                    println!("  (no sessions)");
                }
                Ok(())
            }
            SessionAction::Show { session_id } => {
                let node = cx
                    .db
                    .call(move |conn| crate::db::queries::get_node(conn, &session_id))
                    .await?;
                match node {
                    Some(n) => {
                        println!("Session: {}", n.title);
                        println!("Body:    {}", n.body.as_deref().unwrap_or(""));
                    }
                    None => println!("Session not found."),
                }
                Ok(())
            }
        },

        Commands::Graph { action } => match action.unwrap_or(GraphAction::Explore) {
            GraphAction::Explore => {
                match build_llm_client(&ollama_spec) {
                    Ok(llm) => {
                        cx.set_llm(llm.clone()).await;
                        let agent = crate::agent::orchestrator::Agent {
                            db: cx.db.clone(),
                            embed: cx.embed.clone(),
                            hnsw: cx.hnsw.clone(),
                            config: cx.config.clone(),
                            llm: llm.clone(),
                            tools: crate::tools::builtin_registry(
                                cx.db.clone(),
                                cx.embed.clone(),
                                cx.hnsw.clone(),
                                cx.auto_link_tx.clone(),
                                Some(llm),
                                cx.config.clone(),
                            ).await,
                            auto_link_tx: cx.auto_link_tx.clone(),
                            notif_tx: None,
                        };

                        // Create session for the TUI chat
                        let session = crate::types::Node::session("tui chat session");
                        let session_id = session.id.clone();
                        cx.db
                            .call({
                                let s = session.clone();
                                move |conn| crate::db::queries::insert_node(conn, &s)
                            })
                            .await?;

                        graph_tui::run_with_chat(
                            cx.db.clone(),
                            agent,
                            session_id,
                            Some(cx.embed.clone()),
                            Some(cx.hnsw.clone()),
                        )
                            .await
                            .map_err(|e| crate::error::CortexError::Config(format!("TUI error: {e}")))?;
                    }
                    Err(_) => {
                        // No LLM configured — fall back to graph-only TUI
                        eprintln!("No LLM configured — launching graph explorer without chat.");
                        eprintln!("Set ANTHROPIC_API_KEY or use --ollama to enable chat.\n");
                        let nodes = cx
                            .db
                            .call(|conn| crate::db::queries::get_all_nodes_light(conn))
                            .await?;
                        let edges = cx
                            .db
                            .call(|conn| crate::db::queries::get_all_edges(conn))
                            .await?;
                        graph_tui::run_interactive(nodes, edges)
                            .map_err(|e| crate::error::CortexError::Config(format!("TUI error: {e}")))?;
                    }
                }
                Ok(())
            }
            GraphAction::Overview => {
                let nodes = cx
                    .db
                    .call(|conn| crate::db::queries::get_all_nodes_light(conn))
                    .await?;
                let edges = cx
                    .db
                    .call(|conn| crate::db::queries::get_all_edges(conn))
                    .await?;
                graph_viz::render_overview(&nodes, &edges);
                Ok(())
            }
            GraphAction::Ego { node_id } => {
                // Resolve partial id
                let full_id = node_id.clone();
                let center = cx
                    .db
                    .call({
                        let id = full_id.clone();
                        move |conn| crate::db::queries::get_node(conn, &id)
                    })
                    .await?;
                let center = match center {
                    Some(n) => n,
                    None => {
                        // Try prefix match
                        let prefix = full_id.clone();
                        let nodes = cx
                            .db
                            .call(move |conn| crate::db::queries::get_all_nodes_light(conn))
                            .await?;
                        match nodes.iter().find(|n| n.id.starts_with(&prefix)) {
                            Some(n) => n.clone(),
                            None => {
                                eprintln!("Node not found: {node_id}");
                                return Ok(());
                            }
                        }
                    }
                };

                // Get 1-hop neighbors
                let center_id = center.id.clone();
                let edges = cx
                    .db
                    .call(|conn| crate::db::queries::get_all_edges(conn))
                    .await?;

                let mut neighbor_ids_1hop: Vec<(String, crate::types::EdgeKind, bool)> = Vec::new();
                for e in &edges {
                    if e.src == center_id {
                        neighbor_ids_1hop.push((e.dst.clone(), e.kind, true));
                    } else if e.dst == center_id {
                        neighbor_ids_1hop.push((e.src.clone(), e.kind, false));
                    }
                }

                // Load 1-hop neighbor nodes
                let all_nodes = cx
                    .db
                    .call(|conn| crate::db::queries::get_all_nodes_light(conn))
                    .await?;
                let node_map: std::collections::HashMap<String, crate::types::Node> = all_nodes
                    .into_iter()
                    .map(|n| (n.id.clone(), n))
                    .collect();

                let mut neighbors_1hop: Vec<(crate::types::Node, crate::types::EdgeKind, bool)> =
                    Vec::new();
                for (id, ek, is_out) in &neighbor_ids_1hop {
                    if let Some(node) = node_map.get(id) {
                        neighbors_1hop.push((node.clone(), *ek, *is_out));
                    }
                }

                // 2-hop neighbors
                let mut neighbors_2hop: std::collections::HashMap<
                    String,
                    Vec<(crate::types::Node, crate::types::EdgeKind, bool)>,
                > = std::collections::HashMap::new();
                let hop1_ids: std::collections::HashSet<&str> =
                    neighbor_ids_1hop.iter().map(|(id, _ek, _out): &(String, crate::types::EdgeKind, bool)| id.as_str()).collect();

                for (n1_id, _, _) in &neighbor_ids_1hop {
                    let mut hop2: Vec<(crate::types::Node, crate::types::EdgeKind, bool)> = Vec::new();
                    for e in &edges {
                        let (other_id, ek, is_out) = if e.src == *n1_id {
                            (&e.dst, e.kind, true)
                        } else if e.dst == *n1_id {
                            (&e.src, e.kind, false)
                        } else {
                            continue;
                        };
                        // Skip center and other 1-hop nodes
                        if *other_id == center_id || hop1_ids.contains(other_id.as_str()) {
                            continue;
                        }
                        if let Some(node) = node_map.get(other_id) {
                            hop2.push((node.clone(), ek, is_out));
                        }
                    }
                    if !hop2.is_empty() {
                        // Limit to 5 per 1-hop neighbor
                        hop2.truncate(5);
                        neighbors_2hop.insert(n1_id.clone(), hop2);
                    }
                }

                graph_viz::render_ego(&center, &neighbors_1hop, &neighbors_2hop);
                Ok(())
            }
            GraphAction::Filter { kinds } => {
                let kind_list: Vec<crate::types::NodeKind> = kinds
                    .split(',')
                    .filter_map(|s| crate::types::NodeKind::from_str_opt(s.trim()))
                    .collect();
                if kind_list.is_empty() {
                    eprintln!("No valid kinds. Options: soul, belief, goal, fact, entity, concept, decision, session, turn, pattern, ...");
                    return Ok(());
                }
                let nodes = cx
                    .db
                    .call(|conn| crate::db::queries::get_all_nodes_light(conn))
                    .await?;
                let edges = cx
                    .db
                    .call(|conn| crate::db::queries::get_all_edges(conn))
                    .await?;
                graph_viz::render_filtered(&nodes, &edges, &kind_list);
                Ok(())
            }
        },

        Commands::Consolidate => {
            let report = cx.consolidate().await?;
            println!("Consolidation complete:");
            println!("  Nodes updated:        {}", report.nodes_updated);
            println!("  Contradictions found:  {}", report.contradictions_found);
            println!("  Trust adjustments:     {}", report.trust_adjustments);
            Ok(())
        }

        Commands::Doctor => {
            let (nodes, edges, by_kind) = cx.stats().await?;
            println!("=== Graph Health ===");
            println!("Total nodes: {nodes}");
            println!("Total edges: {edges}");
            for (kind, count) in &by_kind {
                println!("  {kind}: {count}");
            }
            // Check for orphaned nodes (no edges)
            println!("\nChecks passed. Graph is healthy.");
            Ok(())
        }

        Commands::Chat => {
            let llm = build_llm_client(&ollama_spec)?;
            cx.set_llm(llm.clone()).await;
            let agent = crate::agent::orchestrator::Agent {
                db: cx.db.clone(),
                embed: cx.embed.clone(),
                hnsw: cx.hnsw.clone(),
                config: cx.config.clone(),
                llm: llm.clone(),
                tools: crate::tools::builtin_registry(
                    cx.db.clone(),
                    cx.embed.clone(),
                    cx.hnsw.clone(),
                    cx.auto_link_tx.clone(),
                    Some(llm),
                    cx.config.clone(),
                ).await,
                auto_link_tx: cx.auto_link_tx.clone(),
                notif_tx: None,
            };

            // Create a single session for the entire chat
            let session = crate::types::Node::session("chat session");
            let session_id = session.id.clone();
            cx.db
                .call({
                    let s = session.clone();
                    move |conn| crate::db::queries::insert_node(conn, &s)
                })
                .await?;

            println!("omni-cede chat — type 'exit' or Ctrl+C to quit\n");
            let stdin = io::stdin();
            loop {
                print!("> ");
                io::stdout().flush().ok();
                let mut line = String::new();
                if stdin.lock().read_line(&mut line).is_err() || line.trim().is_empty() {
                    continue;
                }
                let input = line.trim();
                if input == "exit" || input == "quit" {
                    break;
                }
                let cli_ctx = crate::types::TurnContext {
                    channel: "cli".to_string(),
                    sender_name: None,
                    user_id: "local".to_string(),
                    is_group: false,
                };
                match agent.run_turn(&session_id, input, &cli_ctx, None).await {
                    Ok(response) => println!("\n{response}\n"),
                    Err(e) => eprintln!("\nError: {e}\n"),
                }
            }
            Ok(())
        }

        Commands::Ask { query } => {
            let llm = build_llm_client(&ollama_spec)?;
            cx.set_llm(llm.clone()).await;
            let agent = crate::agent::orchestrator::Agent {
                db: cx.db.clone(),
                embed: cx.embed.clone(),
                hnsw: cx.hnsw.clone(),
                config: cx.config.clone(),
                llm: llm.clone(),
                tools: crate::tools::builtin_registry(
                    cx.db.clone(),
                    cx.embed.clone(),
                    cx.hnsw.clone(),
                    cx.auto_link_tx.clone(),
                    Some(llm),
                    cx.config.clone(),
                ).await,
                auto_link_tx: cx.auto_link_tx.clone(),
                notif_tx: None,
            };

            match agent.run(&query).await {
                Ok(response) => println!("{response}"),
                Err(e) => eprintln!("Error: {e}"),
            }
            Ok(())
        }
    }
}

/// Build an LLM client based on CLI flags and environment variables.
fn build_llm_client(ollama_spec: &Option<String>) -> crate::error::Result<Arc<dyn crate::llm::LlmClient>> {
    // Check for --ollama flag
    if let Some(ref ollama_spec) = ollama_spec {
        let (model, url) = if let Some(pos) = ollama_spec.find('@') {
            (ollama_spec[..pos].to_string(), ollama_spec[pos + 1..].to_string())
        } else {
            (ollama_spec.clone(), "http://localhost:11434".to_string())
        };
        return Ok(Arc::new(crate::llm::OllamaClient::new(model, url)));
    }

    // Check for ANTHROPIC_API_KEY
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
        return Ok(Arc::new(crate::llm::AnthropicClient::new(key, model)));
    }

    Err(crate::error::CortexError::Config(
        "No LLM backend configured. Set ANTHROPIC_API_KEY or use --ollama <model>".into(),
    ))
}
