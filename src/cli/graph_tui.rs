use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crossterm::{
    event::{Event, KeyCode, KeyEventKind, KeyModifiers, EventStream},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
};
use futures::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap, Tabs},
    Terminal,
};
use tokio::sync::mpsc;

use crate::agent::orchestrator::Agent;
use crate::db::Db;
use crate::embed::EmbedHandle;
use crate::hnsw::VectorIndex;
use crate::types::*;

// ─── Color mapping ──────────────────────────────────────

fn kind_color(kind: NodeKind) -> Color {
    match kind {
        NodeKind::Soul | NodeKind::Belief | NodeKind::Goal => Color::Magenta,
        NodeKind::Fact | NodeKind::Entity | NodeKind::Concept | NodeKind::Decision => Color::Cyan,
        NodeKind::UserInput => Color::LightCyan,
        NodeKind::Session | NodeKind::Turn | NodeKind::LlmCall
        | NodeKind::ToolCall | NodeKind::LoopIteration => Color::Yellow,
        NodeKind::Pattern | NodeKind::Limitation | NodeKind::Capability => Color::Green,
        NodeKind::BackgroundTask => Color::Blue,
    }
}

fn edge_symbol(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::RelatesTo => "───",
        EdgeKind::Supports => "══>",
        EdgeKind::Contradicts => "╳──",
        EdgeKind::DerivesFrom => "◁──",
        EdgeKind::PartOf => "⊂──",
        EdgeKind::Supersedes => "»──",
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 { return String::new(); }
    let char_count = s.chars().count();
    if char_count <= max {
        s.to_string()
    } else {
        let end: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{}…", end)
    }
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

// ─── Category helpers ───────────────────────────────────

const ALL_CATEGORIES: &[&str] = &["All", "Identity", "Knowledge", "Conversational", "Operational", "Self-Model", "Tasks"];

fn node_category(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Soul | NodeKind::Belief | NodeKind::Goal => "Identity",
        NodeKind::Fact | NodeKind::Entity | NodeKind::Concept | NodeKind::Decision => "Knowledge",
        NodeKind::UserInput => "Conversational",
        NodeKind::Session | NodeKind::Turn | NodeKind::LlmCall
        | NodeKind::ToolCall | NodeKind::LoopIteration => "Operational",
        NodeKind::Pattern | NodeKind::Limitation | NodeKind::Capability => "Self-Model",
        NodeKind::BackgroundTask => "Tasks",
    }
}

// ─── Chat message ───────────────────────────────────────

#[derive(Clone)]
struct ChatMsg {
    role: ChatRole,
    text: String,
}

#[derive(Clone, PartialEq)]
enum ChatRole {
    User,
    Assistant,
    System,
}

// ─── Focus ──────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Focus {
    NodeList,
    Detail,
    Chat,
    SoulEdit,
}

// ─── App state ──────────────────────────────────────────

pub struct App {
    // Graph data
    all_nodes: Vec<Node>,
    all_edges: Vec<Edge>,
    filtered_nodes: Vec<usize>,
    list_state: ListState,
    category_idx: usize,
    search_query: String,
    is_node_search: bool,
    outgoing: HashMap<String, Vec<(String, EdgeKind)>>,
    incoming: HashMap<String, Vec<(String, EdgeKind)>>,
    node_map: HashMap<String, usize>,
    history: Vec<usize>,
    // New-node highlighting
    new_node_ids: HashSet<String>,
    // Chat
    chat_messages: Vec<ChatMsg>,
    chat_input: String,
    chat_scroll_up: usize,
    // UI state
    focus: Focus,
    thinking: bool,
    // Stats for delta display
    prev_node_count: usize,
    prev_edge_count: usize,
    // Soul edit modal
    edit_node_id: Option<String>,
    edit_node_title: String,
    edit_input: String,
    edit_saving: bool,
    prev_focus: Focus,
}

impl App {
    pub fn new(nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
        let (outgoing, incoming, node_map) = build_lookups(&nodes, &edges);
        let filtered_nodes: Vec<usize> = (0..nodes.len()).collect();
        let mut list_state = ListState::default();
        if !filtered_nodes.is_empty() {
            list_state.select(Some(0));
        }
        let nc = nodes.len();
        let ec = edges.len();

        Self {
            all_nodes: nodes,
            all_edges: edges,
            filtered_nodes,
            list_state,
            category_idx: 0,
            search_query: String::new(),
            is_node_search: false,
            outgoing,
            incoming,
            node_map,
            history: Vec::new(),
            new_node_ids: HashSet::new(),
            chat_messages: vec![ChatMsg {
                role: ChatRole::System,
                text: "Chat ready — type a message and press Enter.".into(),
            }],
            chat_input: String::new(),
            chat_scroll_up: 0,
            focus: Focus::Chat,
            thinking: false,
            prev_node_count: nc,
            prev_edge_count: ec,
            edit_node_id: None,
            edit_node_title: String::new(),
            edit_input: String::new(),
            edit_saving: false,
            prev_focus: Focus::NodeList,
        }
    }

    fn reload_graph(&mut self, nodes: Vec<Node>, edges: Vec<Edge>) {
        let old_ids: HashSet<String> = self.all_nodes.iter().map(|n| n.id.clone()).collect();
        self.new_node_ids.clear();
        for n in &nodes {
            if !old_ids.contains(&n.id) {
                self.new_node_ids.insert(n.id.clone());
            }
        }
        self.prev_node_count = self.all_nodes.len();
        self.prev_edge_count = self.all_edges.len();

        let (outgoing, incoming, node_map) = build_lookups(&nodes, &edges);
        self.all_nodes = nodes;
        self.all_edges = edges;
        self.outgoing = outgoing;
        self.incoming = incoming;
        self.node_map = node_map;
        self.refilter();
    }

    fn refilter(&mut self) {
        let cat = ALL_CATEGORIES[self.category_idx];
        let query_lower = self.search_query.to_lowercase();

        self.filtered_nodes = self.all_nodes.iter().enumerate()
            .filter(|(_, n)| {
                if cat != "All" && node_category(n.kind) != cat {
                    return false;
                }
                if !query_lower.is_empty() {
                    let title_lower = n.title.to_lowercase();
                    let body_lower = n.body.as_deref().unwrap_or("").to_lowercase();
                    if !title_lower.contains(&query_lower) && !body_lower.contains(&query_lower) {
                        return false;
                    }
                }
                true
            })
            .map(|(i, _)| i)
            .collect();

        if self.filtered_nodes.is_empty() {
            self.list_state.select(None);
        } else {
            let current = self.list_state.selected().unwrap_or(0);
            if current >= self.filtered_nodes.len() {
                self.list_state.select(Some(self.filtered_nodes.len() - 1));
            }
        }
    }

    fn selected_node(&self) -> Option<&Node> {
        self.list_state.selected()
            .and_then(|i| self.filtered_nodes.get(i))
            .map(|&idx| &self.all_nodes[idx])
    }

    fn jump_to_node(&mut self, node_id: &str) {
        if let Some(sel) = self.list_state.selected() {
            if let Some(&idx) = self.filtered_nodes.get(sel) {
                self.history.push(idx);
            }
        }
        if let Some(pos) = self.filtered_nodes.iter().position(|&idx| self.all_nodes[idx].id == node_id) {
            self.list_state.select(Some(pos));
        } else {
            self.category_idx = 0;
            self.search_query.clear();
            self.refilter();
            if let Some(pos) = self.filtered_nodes.iter().position(|&idx| self.all_nodes[idx].id == node_id) {
                self.list_state.select(Some(pos));
            }
        }
    }

    fn go_back(&mut self) {
        if let Some(idx) = self.history.pop() {
            if idx < self.all_nodes.len() {
                let node_id = self.all_nodes[idx].id.clone();
                if let Some(pos) = self.filtered_nodes.iter().position(|&i| self.all_nodes[i].id == node_id) {
                    self.list_state.select(Some(pos));
                }
            }
        }
    }

    fn get_connections(&self, node_id: &str) -> Vec<(usize, EdgeKind, bool)> {
        let mut conns = Vec::new();
        if let Some(outs) = self.outgoing.get(node_id) {
            for (dst, ek) in outs {
                if let Some(&idx) = self.node_map.get(dst) {
                    conns.push((idx, *ek, true));
                }
            }
        }
        if let Some(ins) = self.incoming.get(node_id) {
            for (src, ek) in ins {
                if let Some(&idx) = self.node_map.get(src) {
                    conns.push((idx, *ek, false));
                }
            }
        }
        conns
    }

    fn enter_edit_mode(&mut self) {
        let info = self.selected_node().and_then(|node| {
            match node.kind {
                NodeKind::Soul | NodeKind::Belief | NodeKind::Goal => {
                    Some((node.id.clone(), node.title.clone(), node.body.clone()))
                }
                _ => None,
            }
        });
        if let Some((id, title, body)) = info {
            self.edit_node_id = Some(id);
            self.edit_node_title = title;
            self.edit_input = body.unwrap_or_default();
            self.edit_saving = false;
            self.prev_focus = self.focus;
            self.focus = Focus::SoulEdit;
        }
    }

    fn exit_edit_mode(&mut self) {
        self.edit_node_id = None;
        self.edit_input.clear();
        self.edit_node_title.clear();
        self.focus = self.prev_focus;
    }
}

fn build_lookups(nodes: &[Node], edges: &[Edge]) -> (
    HashMap<String, Vec<(String, EdgeKind)>>,
    HashMap<String, Vec<(String, EdgeKind)>>,
    HashMap<String, usize>,
) {
    let mut outgoing: HashMap<String, Vec<(String, EdgeKind)>> = HashMap::new();
    let mut incoming: HashMap<String, Vec<(String, EdgeKind)>> = HashMap::new();
    for e in edges {
        outgoing.entry(e.src.clone()).or_default().push((e.dst.clone(), e.kind));
        incoming.entry(e.dst.clone()).or_default().push((e.src.clone(), e.kind));
    }
    let node_map: HashMap<String, usize> = nodes.iter().enumerate().map(|(i, n)| (n.id.clone(), i)).collect();
    (outgoing, incoming, node_map)
}

// ─── Agent response channel ─────────────────────────────

enum AgentResult {
    Response(String),
    Error(String),
    EditSaved(String),
    EditError(String),
}

// ─── Public entry points ────────────────────────────────

/// Launch the interactive graph explorer (no chat, original behavior).
pub fn run_interactive(nodes: Vec<Node>, edges: Vec<Edge>) -> std::io::Result<()> {
    // Delegate to a minimal sync event loop (no agent needed)
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new(nodes, edges);
    app.focus = Focus::NodeList; // no chat panel active

    loop {
        terminal.draw(|f| draw(f, &mut app, false))?;
        if let Event::Key(key) = crossterm::event::read()? {
            if key.kind != KeyEventKind::Press { continue; }
            if handle_graph_keys(&mut app, key) { break; }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

/// Launch the graph explorer with editing support for identity nodes (no chat).
pub async fn run_with_edit(
    db: Db,
    embed: EmbedHandle,
    hnsw: Arc<tokio::sync::RwLock<VectorIndex>>,
    start_category: usize,
) -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let nodes = db.call(|conn| crate::db::queries::get_all_nodes_light(conn)).await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let edges = db.call(|conn| crate::db::queries::get_all_edges(conn)).await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let mut app = App::new(nodes, edges);
    app.focus = Focus::NodeList;
    // Pre-filter to specified category (e.g., Identity = 1)
    app.category_idx = start_category;
    app.refilter();

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<AgentResult>();
    let mut event_stream = EventStream::new();

    loop {
        terminal.draw(|f| draw(f, &mut app, false))?;

        tokio::select! {
            maybe_event = event_stream.next() => {
                match maybe_event {
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        if app.focus == Focus::SoulEdit {
                            match key.code {
                                KeyCode::Esc => { app.exit_edit_mode(); }
                                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    if let Some(ref node_id) = app.edit_node_id {
                                        app.edit_saving = true;
                                        let nid = node_id.clone();
                                        let new_body = app.edit_input.clone();
                                        let title = app.edit_node_title.clone();
                                        let db_c = db.clone();
                                        let embed_c = embed.clone();
                                        let hnsw_c = hnsw.clone();
                                        let tx = result_tx.clone();
                                        tokio::spawn(async move {
                                            let body_c = new_body.clone();
                                            let nid_c = nid.clone();
                                            match db_c.call(move |conn| {
                                                crate::db::queries::update_node_fields(
                                                    conn, &nid_c, None, None, Some(&body_c), None, None,
                                                )
                                            }).await {
                                                Ok(_) => {
                                                    let embed_text = format!("{} {}", title, new_body);
                                                    if let Ok(vec) = embed_c.embed(&embed_text).await {
                                                        hnsw_c.write().await.insert(nid.clone(), vec);
                                                    }
                                                    let _ = tx.send(AgentResult::EditSaved(nid));
                                                }
                                                Err(e) => {
                                                    let _ = tx.send(AgentResult::EditError(e.to_string()));
                                                }
                                            }
                                        });
                                    }
                                }
                                KeyCode::Enter => { app.edit_input.push('\n'); }
                                KeyCode::Backspace => { app.edit_input.pop(); }
                                KeyCode::Char(c) => { app.edit_input.push(c); }
                                _ => {}
                            }
                        } else if app.is_node_search {
                            match key.code {
                                KeyCode::Esc => { app.is_node_search = false; app.search_query.clear(); app.refilter(); }
                                KeyCode::Enter => { app.is_node_search = false; }
                                KeyCode::Backspace => { app.search_query.pop(); app.refilter(); }
                                KeyCode::Char(c) => { app.search_query.push(c); app.refilter(); }
                                _ => {}
                            }
                        } else {
                            // Graph keys + 'e' for edit
                            match key.code {
                                KeyCode::Char('e') => app.enter_edit_mode(),
                                _ => { if handle_graph_keys(&mut app, key) { break; } }
                            }
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                    None => break,
                }
            }
            Some(result) = result_rx.recv() => {
                match result {
                    AgentResult::EditSaved(_nid) => {
                        app.edit_saving = false;
                        app.exit_edit_mode();
                        // Reload graph
                        if let Ok(nodes) = db.call(|conn| crate::db::queries::get_all_nodes_light(conn)).await {
                            if let Ok(edges) = db.call(|conn| crate::db::queries::get_all_edges(conn)).await {
                                app.reload_graph(nodes, edges);
                            }
                        }
                    }
                    AgentResult::EditError(e) => {
                        app.edit_saving = false;
                        // Just exit edit mode on error — user sees original content
                        app.exit_edit_mode();
                        let _ = e; // logged via tracing in production
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

/// Launch the TUI with an embedded chat panel and live graph updates.
pub async fn run_with_chat(
    db: Db,
    agent: Agent,
    session_id: String,
    embed: Option<EmbedHandle>,
    hnsw: Option<Arc<tokio::sync::RwLock<VectorIndex>>>,
) -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Load initial graph
    let nodes = db.call(|conn| crate::db::queries::get_all_nodes_light(conn)).await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let edges = db.call(|conn| crate::db::queries::get_all_edges(conn)).await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let mut app = App::new(nodes, edges);

    let agent = Arc::new(agent);
    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<AgentResult>();
    let mut event_stream = EventStream::new();

    loop {
        terminal.draw(|f| draw(f, &mut app, true))?;

        tokio::select! {
            maybe_event = event_stream.next() => {
                match maybe_event {
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        if app.focus == Focus::SoulEdit {
                            match key.code {
                                KeyCode::Esc => { app.exit_edit_mode(); }
                                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    // Save the edit
                                    if let Some(ref node_id) = app.edit_node_id {
                                        app.edit_saving = true;
                                        let nid = node_id.clone();
                                        let new_body = app.edit_input.clone();
                                        let title = app.edit_node_title.clone();
                                        let db_c = db.clone();
                                        let embed_c = embed.clone();
                                        let hnsw_c = hnsw.clone();
                                        let tx = result_tx.clone();
                                        tokio::spawn(async move {
                                            let body_c = new_body.clone();
                                            let nid_c = nid.clone();
                                            match db_c.call(move |conn| {
                                                crate::db::queries::update_node_fields(
                                                    conn, &nid_c, None, None, Some(&body_c), None, None,
                                                )
                                            }).await {
                                                Ok(_) => {
                                                    // Re-embed if embed handle available
                                                    if let Some(ref emb) = embed_c {
                                                        let embed_text = format!("{} {}", title, new_body);
                                                        if let Ok(vec) = emb.embed(&embed_text).await {
                                                            if let Some(ref h) = hnsw_c {
                                                                h.write().await.insert(nid.clone(), vec);
                                                            }
                                                        }
                                                    }
                                                    let _ = tx.send(AgentResult::EditSaved(nid));
                                                }
                                                Err(e) => {
                                                    let _ = tx.send(AgentResult::EditError(e.to_string()));
                                                }
                                            }
                                        });
                                    }
                                }
                                KeyCode::Enter => { app.edit_input.push('\n'); }
                                KeyCode::Backspace => { app.edit_input.pop(); }
                                KeyCode::Char(c) => { app.edit_input.push(c); }
                                _ => {}
                            }
                        } else if app.focus == Focus::Chat && !app.is_node_search {
                            match key.code {
                                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                                KeyCode::Esc => { app.focus = Focus::NodeList; }
                                KeyCode::Up => { app.chat_scroll_up = app.chat_scroll_up.saturating_add(3); }
                                KeyCode::Down => { app.chat_scroll_up = app.chat_scroll_up.saturating_sub(3); }
                                KeyCode::PageUp => { app.chat_scroll_up = app.chat_scroll_up.saturating_add(15); }
                                KeyCode::PageDown => { app.chat_scroll_up = app.chat_scroll_up.saturating_sub(15); }
                                KeyCode::Enter => {
                                    if !app.chat_input.is_empty() && !app.thinking {
                                        let input = app.chat_input.clone();
                                        app.chat_input.clear();
                                        app.chat_messages.push(ChatMsg {
                                            role: ChatRole::User,
                                            text: input.clone(),
                                        });
                                        app.thinking = true;
                                        app.chat_scroll_up = 0;

                                        let agent_c = agent.clone();
                                        let sid = session_id.clone();
                                        let tx = result_tx.clone();
                                        tokio::spawn(async move {
                                            match agent_c.run_turn(&sid, &input).await {
                                                Ok(resp) => { let _ = tx.send(AgentResult::Response(resp)); }
                                                Err(e) => { let _ = tx.send(AgentResult::Error(e.to_string())); }
                                            }
                                        });
                                    }
                                }
                                KeyCode::Backspace => { app.chat_input.pop(); }
                                KeyCode::Char(c) => { app.chat_input.push(c); }
                                KeyCode::Tab => { app.focus = Focus::NodeList; }
                                _ => {}
                            }
                        } else if app.is_node_search {
                            match key.code {
                                KeyCode::Esc => { app.is_node_search = false; app.search_query.clear(); app.refilter(); }
                                KeyCode::Enter => { app.is_node_search = false; }
                                KeyCode::Backspace => { app.search_query.pop(); app.refilter(); }
                                KeyCode::Char(c) => { app.search_query.push(c); app.refilter(); }
                                _ => {}
                            }
                        } else {
                            if handle_graph_keys_with_chat(&mut app, key) { break; }
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                    None => break,
                }
            }
            Some(result) = result_rx.recv() => {
                app.thinking = false;
                app.chat_scroll_up = 0;
                match result {
                    AgentResult::Response(text) => {
                        app.chat_messages.push(ChatMsg { role: ChatRole::Assistant, text });
                    }
                    AgentResult::Error(e) => {
                        app.chat_messages.push(ChatMsg { role: ChatRole::System, text: format!("Error: {e}") });
                    }
                    AgentResult::EditSaved(_nid) => {
                        app.edit_saving = false;
                        app.chat_messages.push(ChatMsg {
                            role: ChatRole::System,
                            text: format!("Saved: {}", app.edit_node_title),
                        });
                        app.exit_edit_mode();
                    }
                    AgentResult::EditError(e) => {
                        app.edit_saving = false;
                        app.chat_messages.push(ChatMsg {
                            role: ChatRole::System,
                            text: format!("Edit error: {e}"),
                        });
                    }
                }
                // Reload graph to show new nodes/edges
                if let Ok(nodes) = db.call(|conn| crate::db::queries::get_all_nodes_light(conn)).await {
                    if let Ok(edges) = db.call(|conn| crate::db::queries::get_all_edges(conn)).await {
                        app.reload_graph(nodes, edges);
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

// ─── Key handling (graph mode) ──────────────────────────

/// Handle keys in graph-only mode (no chat). Returns true if should quit.
fn handle_graph_keys(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    if app.is_node_search {
        match key.code {
            KeyCode::Esc => { app.is_node_search = false; app.search_query.clear(); app.refilter(); }
            KeyCode::Enter => { app.is_node_search = false; }
            KeyCode::Backspace => { app.search_query.pop(); app.refilter(); }
            KeyCode::Char(c) => { app.search_query.push(c); app.refilter(); }
            _ => {}
        }
        return false;
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Char('Q') => return true,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Up | KeyCode::Char('k') => nav_up(app),
        KeyCode::Down | KeyCode::Char('j') => nav_down(app),
        KeyCode::PageUp => nav_page_up(app),
        KeyCode::PageDown => nav_page_down(app),
        KeyCode::Home => nav_home(app),
        KeyCode::End => nav_end(app),
        KeyCode::Tab => { app.category_idx = (app.category_idx + 1) % ALL_CATEGORIES.len(); app.refilter(); }
        KeyCode::Char('/') => { app.is_node_search = true; app.search_query.clear(); }
        KeyCode::Enter => drill_into(app),
        KeyCode::Esc | KeyCode::Backspace => { if !app.search_query.is_empty() { app.search_query.clear(); app.refilter(); } else { app.go_back(); } }
        KeyCode::Char(c @ '1'..='9') => jump_to_connection(app, c),
        _ => {}
    }
    false
}

/// Handle keys in graph mode when chat is also available. Returns true if should quit.
fn handle_graph_keys_with_chat(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Char('Q') => return true,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Char('e') => app.enter_edit_mode(),
        KeyCode::Up | KeyCode::Char('k') => nav_up(app),
        KeyCode::Down | KeyCode::Char('j') => nav_down(app),
        KeyCode::PageUp => nav_page_up(app),
        KeyCode::PageDown => nav_page_down(app),
        KeyCode::Home => nav_home(app),
        KeyCode::End => nav_end(app),
        KeyCode::Char('f') => { app.category_idx = (app.category_idx + 1) % ALL_CATEGORIES.len(); app.refilter(); }
        KeyCode::Char('/') => { app.is_node_search = true; app.search_query.clear(); }
        KeyCode::Enter => drill_into(app),
        KeyCode::Tab => { app.focus = match app.focus { Focus::NodeList => Focus::Detail, Focus::Detail => Focus::Chat, Focus::Chat => Focus::NodeList, Focus::SoulEdit => Focus::NodeList }; }
        KeyCode::BackTab => { app.focus = match app.focus { Focus::NodeList => Focus::Chat, Focus::Detail => Focus::NodeList, Focus::Chat => Focus::Detail, Focus::SoulEdit => Focus::Detail }; }
        KeyCode::Esc | KeyCode::Backspace => { if !app.search_query.is_empty() { app.search_query.clear(); app.refilter(); } else { app.go_back(); } }
        KeyCode::Char(c @ '1'..='9') => jump_to_connection(app, c),
        _ => {}
    }
    false
}

fn nav_up(app: &mut App) {
    if app.focus == Focus::NodeList {
        let i = app.list_state.selected().unwrap_or(0);
        if i > 0 { app.list_state.select(Some(i - 1)); }
    }
}
fn nav_down(app: &mut App) {
    if app.focus == Focus::NodeList {
        let i = app.list_state.selected().unwrap_or(0);
        if i + 1 < app.filtered_nodes.len() { app.list_state.select(Some(i + 1)); }
    }
}
fn nav_page_up(app: &mut App) {
    let i = app.list_state.selected().unwrap_or(0);
    app.list_state.select(Some(i.saturating_sub(20)));
}
fn nav_page_down(app: &mut App) {
    let i = app.list_state.selected().unwrap_or(0);
    let max = app.filtered_nodes.len().saturating_sub(1);
    app.list_state.select(Some((i + 20).min(max)));
}
fn nav_home(app: &mut App) {
    if !app.filtered_nodes.is_empty() { app.list_state.select(Some(0)); }
}
fn nav_end(app: &mut App) {
    if !app.filtered_nodes.is_empty() { app.list_state.select(Some(app.filtered_nodes.len() - 1)); }
}
fn drill_into(app: &mut App) {
    if let Some(node) = app.selected_node() {
        let conns = app.get_connections(&node.id);
        if let Some((idx, _, _)) = conns.first() {
            let target_id = app.all_nodes[*idx].id.clone();
            app.jump_to_node(&target_id);
        }
    }
}
fn jump_to_connection(app: &mut App, c: char) {
    if let Some(node) = app.selected_node() {
        let conns = app.get_connections(&node.id);
        let idx = (c as u8 - b'1') as usize;
        if let Some((node_idx, _, _)) = conns.get(idx) {
            let target_id = app.all_nodes[*node_idx].id.clone();
            app.jump_to_node(&target_id);
        }
    }
}

// ─── Drawing ────────────────────────────────────────────

fn draw(f: &mut ratatui::Frame, app: &mut App, show_chat: bool) {
    let size = f.area();

    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(10),   // content
            Constraint::Length(1), // help bar
        ])
        .split(size);

    draw_header(f, app, main_chunks[0]);

    if show_chat {
        // Left = node list, Right = detail (top) + chat (bottom)
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(35),
                Constraint::Percentage(65),
            ])
            .split(main_chunks[1]);

        draw_node_list(f, app, content_chunks[0]);

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(45),
                Constraint::Percentage(55),
            ])
            .split(content_chunks[1]);

        draw_detail(f, app, right_chunks[0]);
        draw_chat(f, app, right_chunks[1]);
    } else {
        // Original two-panel layout (no chat)
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Percentage(60),
            ])
            .split(main_chunks[1]);

        draw_node_list(f, app, content_chunks[0]);
        draw_detail(f, app, content_chunks[1]);
    }

    draw_help(f, app, main_chunks[2], show_chat);

    // Soul edit modal overlay
    if app.focus == Focus::SoulEdit {
        draw_soul_edit(f, app, size);
    }
}

fn draw_header(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let delta_nodes = app.all_nodes.len() as i64 - app.prev_node_count as i64;
    let delta_edges = app.all_edges.len() as i64 - app.prev_edge_count as i64;
    let delta_str = if delta_nodes > 0 || delta_edges > 0 {
        format!(" (+{} nodes, +{} edges)", delta_nodes.max(0), delta_edges.max(0))
    } else {
        String::new()
    };

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(55), Constraint::Min(20)])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" CORTEX ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(" {} nodes  {} edges ", app.all_nodes.len(), app.all_edges.len()),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(delta_str, Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(title, chunks[0]);

    let tab_titles: Vec<Line> = ALL_CATEGORIES.iter().map(|&c| Line::from(Span::raw(c))).collect();
    let tabs = Tabs::new(tab_titles)
        .select(app.category_idx)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
        .divider("│")
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(tabs, chunks[1]);
}

fn draw_node_list(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = app.filtered_nodes.iter().map(|&idx| {
        let node = &app.all_nodes[idx];
        let is_new = app.new_node_ids.contains(&node.id);
        let base_color = if is_new { Color::LightGreen } else { kind_color(node.kind) };
        let new_marker = if is_new { "★ " } else { "" };

        let line = Line::from(vec![
            Span::styled(new_marker, Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("[{:<10}] ", node.kind.as_str()),
                Style::default().fg(base_color),
            ),
            Span::styled(
                truncate(&node.title, (area.width as usize).saturating_sub(20)),
                Style::default().fg(Color::White),
            ),
        ]);
        ListItem::new(line)
    }).collect();

    let border_color = if app.focus == Focus::NodeList { Color::Cyan } else { Color::DarkGray };
    let title = if app.is_node_search {
        format!(" Nodes ({}) /{} ", app.filtered_nodes.len(), app.search_query)
    } else {
        format!(" Nodes ({}) ", app.filtered_nodes.len())
    };

    let list = List::new(items)
        .block(Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color)))
        .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn draw_detail(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let border_color = if app.focus == Focus::Detail { Color::Cyan } else { Color::DarkGray };

    let node = match app.selected_node() {
        Some(n) => n,
        None => {
            let empty = Paragraph::new("No node selected")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().title(" Details ").borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color)));
            f.render_widget(empty, area);
            return;
        }
    };

    let color = kind_color(node.kind);
    let is_new = app.new_node_ids.contains(&node.id);
    let conns = app.get_connections(&node.id);
    let mut lines: Vec<Line> = Vec::new();

    // Title
    let new_tag = if is_new {
        Span::styled(" ★NEW ", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD))
    } else { Span::raw("") };

    lines.push(Line::from(vec![
        Span::styled(format!("[{}] ", node.kind.as_str()), Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::styled(
            truncate(&node.title, (area.width as usize).saturating_sub(20)),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        new_tag,
    ]));

    // Body
    if let Some(ref body) = node.body {
        let max_w = (area.width as usize).saturating_sub(4).max(1);
        let body_clean = body.replace('\n', " ");
        let display = truncate(&body_clean, max_w * 3);
        // Word-wrap the body text at char boundaries
        let mut current = String::new();
        for ch in display.chars() {
            current.push(ch);
            if current.chars().count() >= max_w {
                lines.push(Line::from(Span::styled(
                    std::mem::take(&mut current),
                    Style::default().fg(Color::Gray),
                )));
            }
        }
        if !current.is_empty() {
            lines.push(Line::from(Span::styled(
                current,
                Style::default().fg(Color::Gray),
            )));
        }
    }

    // Stats
    let imp_filled = (node.importance * 10.0).round() as usize;
    let trust_filled = (node.trust_score * 10.0).round() as usize;
    lines.push(Line::from(vec![
        Span::styled("imp ", Style::default().fg(Color::DarkGray)),
        Span::styled("█".repeat(imp_filled), Style::default().fg(Color::Yellow)),
        Span::styled("░".repeat(10_usize.saturating_sub(imp_filled)), Style::default().fg(Color::DarkGray)),
        Span::styled("  tru ", Style::default().fg(Color::DarkGray)),
        Span::styled("█".repeat(trust_filled), Style::default().fg(Color::Green)),
        Span::styled("░".repeat(10_usize.saturating_sub(trust_filled)), Style::default().fg(Color::DarkGray)),
        Span::styled(format!("  {}", short_id(&node.id)), Style::default().fg(Color::DarkGray)),
    ]));

    // Connections
    lines.push(Line::from(""));
    if conns.is_empty() {
        lines.push(Line::from(Span::styled("(no connections)", Style::default().fg(Color::DarkGray))));
    } else {
        lines.push(Line::from(Span::styled(
            format!("Connections ({})", conns.len()),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
        for (i, (node_idx, ek, is_out)) in conns.iter().enumerate().take(12) {
            let neighbor = &app.all_nodes[*node_idx];
            let nc = kind_color(neighbor.kind);
            let arrow = if *is_out { "→" } else { "←" };
            let sym = edge_symbol(*ek);
            let marker = if app.new_node_ids.contains(&neighbor.id) { " ★" } else { "" };
            lines.push(Line::from(vec![
                Span::styled(format!("{:>2} ", i + 1), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{sym}{arrow} "), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("[{}] ", neighbor.kind.as_str()), Style::default().fg(nc)),
                Span::styled(
                    truncate(&neighbor.title, (area.width as usize).saturating_sub(25)),
                    Style::default().fg(Color::White),
                ),
                Span::styled(marker, Style::default().fg(Color::LightGreen)),
            ]));
        }
        if conns.len() > 12 {
            lines.push(Line::from(Span::styled(
                format!("   … +{} more", conns.len() - 12),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    let detail = Paragraph::new(lines)
        .block(Block::default().title(" Details ").borders(Borders::ALL)
            .border_style(Style::default().fg(border_color)))
        .wrap(Wrap { trim: false });
    f.render_widget(detail, area);
}

fn draw_chat(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let border_color = if app.focus == Focus::Chat { Color::Cyan } else { Color::DarkGray };

    let chat_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    // Messages
    let max_w = (chat_chunks[0].width as usize).saturating_sub(4);
    let mut msg_lines: Vec<Line> = Vec::new();
    for msg in &app.chat_messages {
        let (prefix, color) = match msg.role {
            ChatRole::User => ("you: ", Color::Cyan),
            ChatRole::Assistant => ("cortex: ", Color::Magenta),
            ChatRole::System => ("", Color::DarkGray),
        };
        let indent = " ".repeat(prefix.len());
        let text_w = max_w.saturating_sub(prefix.len()).max(10);
        let wrapped = word_wrap(&msg.text, text_w);
        for (i, line) in wrapped.iter().enumerate() {
            if i == 0 {
                msg_lines.push(Line::from(Span::styled(
                    format!("{}{}", prefix, line),
                    Style::default().fg(color),
                )));
            } else {
                msg_lines.push(Line::from(Span::styled(
                    format!("{}{}", indent, line),
                    Style::default().fg(color),
                )));
            }
        }
    }
    if app.thinking {
        msg_lines.push(Line::from(Span::styled("cortex: thinking…", Style::default().fg(Color::Magenta))));
    }

    let visible_h = chat_chunks[0].height.saturating_sub(2) as usize;
    let total_lines = msg_lines.len();
    let max_scroll = total_lines.saturating_sub(visible_h);
    // Clamp scroll_up so it doesn't exceed available content
    let clamped_up = app.chat_scroll_up.min(max_scroll);
    let scroll = (max_scroll.saturating_sub(clamped_up)) as u16;

    let scroll_indicator = if clamped_up > 0 {
        format!(" Chat [↑{}] ", clamped_up)
    } else {
        " Chat ".to_string()
    };

    let messages_widget = Paragraph::new(msg_lines)
        .block(Block::default().title(scroll_indicator).borders(Borders::ALL)
            .border_style(Style::default().fg(border_color)))
        .scroll((scroll, 0));
    f.render_widget(messages_widget, chat_chunks[0]);

    // Input
    let cursor = if app.focus == Focus::Chat { "█" } else { "" };
    let input_widget = Paragraph::new(Line::from(Span::styled(
        format!(" > {}{}", app.chat_input, cursor),
        Style::default().fg(Color::White),
    )))
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)));
    f.render_widget(input_widget, chat_chunks[1]);
}

fn draw_soul_edit(f: &mut ratatui::Frame, app: &App, area: Rect) {
    // Center the modal — 70% width, 60% height
    let modal_w = (area.width as f32 * 0.7) as u16;
    let modal_h = (area.height as f32 * 0.6) as u16;
    let x = area.x + (area.width.saturating_sub(modal_w)) / 2;
    let y = area.y + (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect::new(x, y, modal_w, modal_h);

    // Clear the area behind the modal
    let clear = Paragraph::new("").style(Style::default().bg(Color::Black));
    f.render_widget(clear, modal_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3), Constraint::Length(1)])
        .split(modal_area);

    // Title bar
    let title_text = format!(" Editing: {} ", app.edit_node_title);
    let title = Paragraph::new(Line::from(Span::styled(
        &title_text,
        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
    )))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );
    f.render_widget(title, chunks[0]);

    // Content area
    let status = if app.edit_saving { " (saving…)" } else { "" };
    let cursor = if !app.edit_saving { "█" } else { "" };
    let content_text = format!("{}{}", app.edit_input, cursor);
    let content = Paragraph::new(content_text)
        .block(
            Block::default()
                .title(format!(" Content{status} "))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(content, chunks[1]);

    // Help bar
    let help = Paragraph::new(Line::from(Span::styled(
        " Ctrl+S: save │ Esc: cancel │ Enter: newline",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(help, chunks[2]);
}

fn draw_help(f: &mut ratatui::Frame, app: &App, area: Rect, show_chat: bool) {
    let text = if app.is_node_search {
        " Type to search │ Enter: apply │ Esc: cancel"
    } else if show_chat {
        match app.focus {
            Focus::Chat => " Type + Enter │ ↑↓/PgUp/PgDn: scroll │ Tab/Esc: graph │ Ctrl+C: quit",
            _ => " ↑↓/jk: navigate │ f: filter │ /: search │ Enter: drill │ e: edit │ 1-9: jump │ Tab: cycle │ q: quit",
        }
    } else {
        " ↑↓/jk: navigate │ Tab: category │ /: search │ Enter: drill │ 1-9: jump │ Esc: back │ q: quit"
    };
    let help = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::DarkGray))));
    f.render_widget(help, area);
}

// ─── Helpers ────────────────────────────────────────────

fn word_wrap(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.len() + 1 + word.len() <= max_width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() { lines.push(current); }
    if lines.is_empty() { lines.push(String::new()); }
    lines
}
