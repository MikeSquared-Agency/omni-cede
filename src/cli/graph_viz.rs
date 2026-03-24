use std::collections::{BTreeMap, HashMap, HashSet};

use crate::types::*;

// ─── ANSI helpers ───────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const WHITE: &str = "\x1b[97m";

fn kind_color(kind: NodeKind) -> &'static str {
    match kind {
        // Identity → magenta
        NodeKind::Soul | NodeKind::Belief | NodeKind::Goal => "\x1b[95m",
        // Knowledge → cyan
        NodeKind::Fact | NodeKind::Entity | NodeKind::Concept | NodeKind::Decision => "\x1b[96m",
        // Operational → yellow
        NodeKind::Session | NodeKind::Turn | NodeKind::LlmCall
        | NodeKind::ToolCall | NodeKind::LoopIteration => "\x1b[93m",
        // Self-model → green
        NodeKind::Pattern | NodeKind::Limitation | NodeKind::Capability => "\x1b[92m",
        // Sub-agents → blue
        NodeKind::SubAgent | NodeKind::Delegation | NodeKind::Synthesis => "\x1b[94m",
    }
}

fn edge_label(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::RelatesTo => "relates_to",
        EdgeKind::Supports => "supports",
        EdgeKind::Contradicts => "contradicts",
        EdgeKind::DerivesFrom => "derives_from",
        EdgeKind::PartOf => "part_of",
        EdgeKind::Supersedes => "supersedes",
    }
}

fn edge_arrow(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::RelatesTo => "───",
        EdgeKind::Supports => "══>",
        EdgeKind::Contradicts => "╳──",
        EdgeKind::DerivesFrom => "◁──",
        EdgeKind::PartOf => "⊂──",
        EdgeKind::Supersedes => "»──",
    }
}

fn kind_category(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Soul | NodeKind::Belief | NodeKind::Goal => "Identity",
        NodeKind::Fact | NodeKind::Entity | NodeKind::Concept | NodeKind::Decision => "Knowledge",
        NodeKind::Session | NodeKind::Turn | NodeKind::LlmCall
        | NodeKind::ToolCall | NodeKind::LoopIteration => "Operational",
        NodeKind::Pattern | NodeKind::Limitation | NodeKind::Capability => "Self-Model",
        NodeKind::SubAgent | NodeKind::Delegation | NodeKind::Synthesis => "Sub-Agents",
    }
}

fn short_id(id: &str) -> &str {
    if id.len() > 8 { &id[..8] } else { id }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// ─── Overview ───────────────────────────────────────────

pub fn render_overview(nodes: &[Node], edges: &[Edge]) {
    // ── Header ──
    println!(
        "\n{BOLD}{WHITE}╔══════════════════════════════════════════════════════════════════╗{RESET}"
    );
    println!(
        "{BOLD}{WHITE}║               C O R T E X   K N O W L E D G E   G R A P H       ║{RESET}"
    );
    println!(
        "{BOLD}{WHITE}╠══════════════════════════════════════════════════════════════════╣{RESET}"
    );
    println!(
        "{BOLD}{WHITE}║  Nodes: {:<6}  Edges: {:<6}                                    ║{RESET}",
        nodes.len(),
        edges.len()
    );
    println!(
        "{BOLD}{WHITE}╚══════════════════════════════════════════════════════════════════╝{RESET}\n"
    );

    if nodes.is_empty() {
        println!("  {DIM}(empty graph){RESET}");
        return;
    }

    // ── Group nodes by category then kind ──
    let mut by_category: BTreeMap<&str, BTreeMap<&str, Vec<&Node>>> = BTreeMap::new();
    for n in nodes {
        let cat = kind_category(n.kind);
        by_category
            .entry(cat)
            .or_default()
            .entry(n.kind.as_str())
            .or_default()
            .push(n);
    }

    // Build lookup for edges
    let mut outgoing: HashMap<&str, Vec<&Edge>> = HashMap::new();
    for e in edges {
        outgoing.entry(e.src.as_str()).or_default().push(e);
    }

    // Node id → title map for edge rendering
    let id_title: HashMap<&str, (&str, NodeKind)> = nodes
        .iter()
        .map(|n| (n.id.as_str(), (n.title.as_str(), n.kind)))
        .collect();

    // ── Render each category ──
    for (category, kinds) in &by_category {
        // Pick representative color from first kind
        let first_kind = kinds.keys().next().unwrap();
        let color = kind_color(
            NodeKind::from_str_opt(first_kind).unwrap_or(NodeKind::Fact),
        );

        println!("  {color}{BOLD}┌─ {category} ─────────────────────────────────────────────────┐{RESET}");

        for (kind_str, kind_nodes) in kinds {
            let kind = NodeKind::from_str_opt(kind_str).unwrap_or(NodeKind::Fact);
            let kcolor = kind_color(kind);

            for n in kind_nodes {
                let title = truncate(&n.title, 42);
                let trust_bar = trust_bar(n.trust_score);
                let imp_bar = importance_bar(n.importance);
                println!(
                    "  {kcolor}│  ● [{kind_str:<12}] {title:<42} {imp_bar} {trust_bar}  {DIM}{}{RESET}",
                    short_id(&n.id),
                );
            }
        }

        println!("  {color}{BOLD}└────────────────────────────────────────────────────────────────┘{RESET}");
        println!();
    }

    // ── Edge summary ──
    let mut edge_kinds: HashMap<EdgeKind, usize> = HashMap::new();
    for e in edges {
        *edge_kinds.entry(e.kind).or_default() += 1;
    }

    println!("  {BOLD}{WHITE}Edges{RESET}");
    for (kind, count) in &edge_kinds {
        let arrow = edge_arrow(*kind);
        let label = edge_label(*kind);
        println!("    {arrow} {label}: {count}");
    }
    println!();

    // ── Show actual connections (up to 40) ──
    let display_edges: Vec<&Edge> = edges.iter().take(40).collect();
    if !display_edges.is_empty() {
        println!("  {BOLD}{WHITE}Connections{RESET} {DIM}(showing {}){RESET}", display_edges.len());
        for e in &display_edges {
            let src_title = id_title
                .get(e.src.as_str())
                .map(|(t, _)| truncate(t, 28))
                .unwrap_or_else(|| short_id(&e.src).to_string());
            let dst_title = id_title
                .get(e.dst.as_str())
                .map(|(t, _)| truncate(t, 28))
                .unwrap_or_else(|| short_id(&e.dst).to_string());
            let src_kind = id_title.get(e.src.as_str()).map(|(_, k)| *k);
            let dst_kind = id_title.get(e.dst.as_str()).map(|(_, k)| *k);
            let sc = src_kind.map(kind_color).unwrap_or(DIM);
            let dc = dst_kind.map(kind_color).unwrap_or(DIM);
            let arrow = edge_arrow(e.kind);
            println!(
                "    {sc}{src_title}{RESET} {DIM}{arrow}{RESET} {dc}{dst_title}{RESET}"
            );
        }
        if edges.len() > 40 {
            println!("    {DIM}… and {} more{RESET}", edges.len() - 40);
        }
    }
    println!();
}

// ─── Ego network (node + neighbors) ────────────────────

pub fn render_ego(
    center: &Node,
    neighbors_1hop: &[(Node, EdgeKind, bool)], // (node, edge_kind, is_outgoing)
    neighbors_2hop: &HashMap<String, Vec<(Node, EdgeKind, bool)>>,
) {
    let cc = kind_color(center.kind);

    // ── Center node box ──
    let center_title = truncate(&center.title, 50);
    let center_body = center
        .body
        .as_deref()
        .map(|b| truncate(b, 60))
        .unwrap_or_default();

    println!();
    println!(
        "  {cc}{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}"
    );
    println!(
        "  {cc}{BOLD}║  [{:<12}] {:<40} ║{RESET}",
        center.kind.as_str(),
        center_title
    );
    if !center_body.is_empty() {
        println!(
            "  {cc}{BOLD}║  {:<54} ║{RESET}",
            center_body
        );
    }
    println!(
        "  {cc}{BOLD}║  imp: {:<6.3}  trust: {:<6.3}  accesses: {:<6}       ║{RESET}",
        center.importance, center.trust_score, center.access_count
    );
    println!(
        "  {cc}{BOLD}╚═══════════════════════╦══════════════════════════════════╝{RESET}"
    );

    if neighbors_1hop.is_empty() {
        println!("  {DIM}(no connections){RESET}");
        println!();
        return;
    }

    // ── 1-hop neighbors ──
    let total_1hop = neighbors_1hop.len();
    for (i, (n, ek, is_out)) in neighbors_1hop.iter().enumerate() {
        let nc = kind_color(n.kind);
        let is_last = i == total_1hop - 1;
        let branch = if is_last { "└" } else { "├" };
        let cont = if is_last { " " } else { "│" };
        let dir = if *is_out { "→" } else { "←" };
        let arrow = edge_arrow(*ek);
        let title = truncate(&n.title, 36);

        println!(
            "  {cc}{BOLD}                        {branch}──{arrow}─{dir}{RESET} {nc}[{}] {title}{RESET} {DIM}({}){RESET}",
            n.kind.as_str(),
            short_id(&n.id),
        );

        // ── 2-hop neighbors from this node ──
        if let Some(hop2) = neighbors_2hop.get(&n.id) {
            for (j, (n2, ek2, is_out2)) in hop2.iter().enumerate() {
                let nc2 = kind_color(n2.kind);
                let is_last2 = j == hop2.len() - 1;
                let branch2 = if is_last2 { "└" } else { "├" };
                let dir2 = if *is_out2 { "→" } else { "←" };
                let arrow2 = edge_arrow(*ek2);
                let title2 = truncate(&n2.title, 30);
                println!(
                    "  {cc}{BOLD}                        {cont}       {branch2}──{arrow2}─{dir2}{RESET} {nc2}[{}] {title2}{RESET} {DIM}({}){RESET}",
                    n2.kind.as_str(),
                    short_id(&n2.id),
                );
            }
        }
    }
    println!();
}

// ─── Filtered view ──────────────────────────────────────

pub fn render_filtered(nodes: &[Node], edges: &[Edge], kinds: &[NodeKind]) {
    let kind_set: HashSet<NodeKind> = kinds.iter().cloned().collect();

    // Filter nodes
    let filtered_nodes: Vec<&Node> = nodes
        .iter()
        .filter(|n| kind_set.contains(&n.kind))
        .collect();

    // Filter edges to only those between filtered nodes
    let node_ids: HashSet<&str> = filtered_nodes.iter().map(|n| n.id.as_str()).collect();
    let filtered_edges: Vec<&Edge> = edges
        .iter()
        .filter(|e| node_ids.contains(e.src.as_str()) || node_ids.contains(e.dst.as_str()))
        .collect();

    let kind_names: Vec<&str> = kinds.iter().map(|k| k.as_str()).collect();
    println!(
        "\n  {BOLD}{WHITE}Filtered graph: {}{RESET}  ({} nodes, {} edges)\n",
        kind_names.join(", "),
        filtered_nodes.len(),
        filtered_edges.len(),
    );

    // Node id → title map
    let id_title: HashMap<&str, (&str, NodeKind)> = nodes
        .iter()
        .map(|n| (n.id.as_str(), (n.title.as_str(), n.kind)))
        .collect();

    for n in &filtered_nodes {
        let color = kind_color(n.kind);
        let title = truncate(&n.title, 50);
        let imp = importance_bar(n.importance);
        let trust = trust_bar(n.trust_score);
        println!(
            "  {color}● [{:<12}] {title:<50} {imp} {trust}{RESET}",
            n.kind.as_str()
        );
    }

    if !filtered_edges.is_empty() {
        println!();
        for e in &filtered_edges {
            let src_title = id_title
                .get(e.src.as_str())
                .map(|(t, _)| truncate(t, 28))
                .unwrap_or_else(|| short_id(&e.src).to_string());
            let dst_title = id_title
                .get(e.dst.as_str())
                .map(|(t, _)| truncate(t, 28))
                .unwrap_or_else(|| short_id(&e.dst).to_string());
            let sc = id_title.get(e.src.as_str()).map(|(_, k)| kind_color(*k)).unwrap_or(DIM);
            let dc = id_title.get(e.dst.as_str()).map(|(_, k)| kind_color(*k)).unwrap_or(DIM);
            let arrow = edge_arrow(e.kind);
            println!(
                "    {sc}{src_title}{RESET} {DIM}{arrow}{RESET} {dc}{dst_title}{RESET}"
            );
        }
    }
    println!();
}

// ─── Mini-bars ──────────────────────────────────────────

fn importance_bar(v: f64) -> String {
    let filled = (v * 5.0).round() as usize;
    let empty = 5_usize.saturating_sub(filled);
    format!(
        "{DIM}imp:{RESET}\x1b[93m{}\x1b[90m{}{RESET}",
        "█".repeat(filled),
        "░".repeat(empty),
    )
}

fn trust_bar(v: f64) -> String {
    let filled = (v * 5.0).round() as usize;
    let empty = 5_usize.saturating_sub(filled);
    format!(
        "{DIM}tru:{RESET}\x1b[92m{}\x1b[90m{}{RESET}",
        "█".repeat(filled),
        "░".repeat(empty),
    )
}
