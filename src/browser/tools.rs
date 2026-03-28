//! Browser tools registered into the agent's ToolRegistry.
//!
//! These tools give the agent the ability to:
//! - Launch/connect to a browser
//! - Navigate, click, fill, screenshot, snapshot
//! - Discover and invoke WebMCP tools
//! - Run stored browser tools and workflows

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::tools::{Tool, ToolRegistry};
use crate::types::ToolResult;

use super::cdp::BrowserSession;
use super::webmcp::WebMcpCache;

/// Shared browser state accessible by all browser tools.
pub struct BrowserState {
    /// The active browser session (None if not launched).
    pub session: Option<BrowserSession>,
    /// WebMCP tool cache.
    pub webmcp_cache: WebMcpCache,
    /// Stored tool definitions loaded from graph or config.
    pub stored_tools: HashMap<String, super::store::StoredTool>,
}

impl BrowserState {
    pub fn new() -> Self {
        Self {
            session: None,
            webmcp_cache: WebMcpCache::new(),
            stored_tools: HashMap::new(),
        }
    }
}

/// Register all browser tools into the given registry.
///
/// The browser state is shared via `Arc<Mutex<BrowserState>>`.
pub fn register_browser_tools(reg: &mut ToolRegistry) {
    let state: Arc<Mutex<BrowserState>> = Arc::new(Mutex::new(BrowserState::new()));

    // ── browser_launch: start a browser session ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_launch".to_string(),
            description: concat!(
                "Launch a Chrome browser with remote debugging, or connect to an existing one. ",
                "Must be called before any other browser_* tools. ",
                "If Chrome is already running with --remote-debugging-port, use connect_url instead."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "headless": {
                        "type": "boolean",
                        "description": "Run headless (no visible window). Default: true"
                    },
                    "port": {
                        "type": "integer",
                        "description": "Debugging port (default: 9222)"
                    },
                    "connect_url": {
                        "type": "string",
                        "description": "WebSocket URL to connect to an existing Chrome (overrides launch)"
                    }
                },
                "required": []
            }),
            trust: 0.7,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let headless = input["headless"].as_bool().unwrap_or(true);
                    let port = input["port"].as_u64().unwrap_or(9222) as u16;
                    let connect_url = input["connect_url"].as_str().map(String::from);

                    let session = if let Some(url) = connect_url {
                        BrowserSession::connect(&url).await
                    } else {
                        BrowserSession::launch(None, port, headless).await
                    };

                    match session {
                        Ok(s) => {
                            // Apply stealth
                            if let Err(e) = super::stealth::apply_stealth(&s).await {
                                tracing::warn!("stealth patches failed: {e}");
                            }
                            let mut st = state.lock().await;
                            st.session = Some(s);
                            Ok(ToolResult {
                                output: "Browser launched and ready.".to_string(),
                                success: true,
                            })
                        }
                        Err(e) => Ok(ToolResult {
                            output: format!("Failed to launch browser: {e}"),
                            success: false,
                        }),
                    }
                })
            }),
        });
    }

    // ── browser_navigate: go to a URL ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_navigate".to_string(),
            description: "Navigate the browser to a URL. Waits for page load.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to"
                    }
                },
                "required": ["url"]
            }),
            trust: 0.7,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let url = input["url"].as_str().unwrap_or("").to_string();
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session. Call browser_launch first.".into()
                        ))?;
                    session.navigate(&url).await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;

                    // Auto-discover WebMCP tools for this domain
                    drop(st);
                    if let Ok(parsed) = url::Url::parse(&url) {
                        if let Some(domain) = parsed.host_str() {
                            let mut st = state.lock().await;
                            let tools = super::webmcp::discover(domain).await;
                            if !tools.is_empty() {
                                let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
                                for tool in tools {
                                    st.webmcp_cache.cache.entry(domain.to_string())
                                        .or_default()
                                        .push(tool);
                                }
                                return Ok(ToolResult {
                                    output: format!(
                                        "Navigated to {url}\nWebMCP tools discovered: {}",
                                        tool_names.join(", ")
                                    ),
                                    success: true,
                                });
                            }
                        }
                    }

                    Ok(ToolResult {
                        output: format!("Navigated to {url}"),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_snapshot: get compact page content ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_snapshot".to_string(),
            description: concat!(
                "Get a compact snapshot of the current page. Returns interactive elements ",
                "(links, buttons, inputs) and content (headings, paragraphs) with CSS selectors. ",
                "Use this to understand page structure before clicking or filling forms."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            trust: 0.8,
            handler: Arc::new(move |_input| {
                let state = state.clone();
                Box::pin(async move {
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session. Call browser_launch first.".into()
                        ))?;
                    let url = session.current_url().await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    let elements = session.snapshot().await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    let text = super::snapshot::format_snapshot(&elements);
                    Ok(ToolResult {
                        output: format!("URL: {url}\n{} element(s):\n{text}", elements.len()),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_click: click an element ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_click".to_string(),
            description: "Click an element on the page by CSS selector. Use browser_snapshot first to find selectors.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the element to click"
                    }
                },
                "required": ["selector"]
            }),
            trust: 0.7,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let selector = input["selector"].as_str().unwrap_or("").to_string();
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session.".into()
                        ))?;
                    session.click(&selector).await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    Ok(ToolResult {
                        output: format!("Clicked: {selector}"),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_fill: fill an input field ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_fill".to_string(),
            description: "Fill an input or textarea with text. Uses CSS selector to target the element.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the input element"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to fill in"
                    }
                },
                "required": ["selector", "text"]
            }),
            trust: 0.7,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let selector = input["selector"].as_str().unwrap_or("").to_string();
                    let text = input["text"].as_str().unwrap_or("").to_string();
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session.".into()
                        ))?;
                    session.fill(&selector, &text).await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    Ok(ToolResult {
                        output: format!("Filled {selector} with text"),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_screenshot: capture page image ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_screenshot".to_string(),
            description: "Take a screenshot of the current page. Returns base64-encoded PNG.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            trust: 0.8,
            handler: Arc::new(move |_input| {
                let state = state.clone();
                Box::pin(async move {
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session.".into()
                        ))?;
                    let b64 = session.screenshot().await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    Ok(ToolResult {
                        output: format!("[screenshot: {} bytes base64]\ndata:image/png;base64,{}", b64.len(), &b64[..100.min(b64.len())]),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_evaluate: run JavaScript ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_evaluate".to_string(),
            description: "Execute JavaScript in the page context and return the result.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "JavaScript expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
            trust: 0.6,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let expr = input["expression"].as_str().unwrap_or("").to_string();
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session.".into()
                        ))?;
                    let result = session.evaluate(&expr).await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    Ok(ToolResult {
                        output: serde_json::to_string_pretty(&result).unwrap_or_default(),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_wait: wait for element ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_wait".to_string(),
            description: "Wait for a CSS selector to appear in the DOM.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to wait for"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (default: 10000)"
                    }
                },
                "required": ["selector"]
            }),
            trust: 0.8,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let selector = input["selector"].as_str().unwrap_or("").to_string();
                    let timeout = input["timeout_ms"].as_u64().unwrap_or(10000);
                    let st = state.lock().await;
                    let session = st.session.as_ref()
                        .ok_or_else(|| crate::error::CortexError::Tool(
                            "No browser session.".into()
                        ))?;
                    session.wait_for_selector(&selector, timeout).await
                        .map_err(|e| crate::error::CortexError::Tool(e))?;
                    Ok(ToolResult {
                        output: format!("Element found: {selector}"),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_close: close the browser ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_close".to_string(),
            description: "Close the browser session.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            trust: 0.8,
            handler: Arc::new(move |_input| {
                let state = state.clone();
                Box::pin(async move {
                    let mut st = state.lock().await;
                    if let Some(session) = st.session.as_ref() {
                        let _ = session.close().await;
                    }
                    st.session = None;
                    Ok(ToolResult {
                        output: "Browser closed.".to_string(),
                        success: true,
                    })
                })
            }),
        });
    }

    // ── browser_webmcp: discover and call WebMCP tools ──
    {
        let state = state.clone();
        reg.register(Tool {
            name: "browser_webmcp".to_string(),
            description: concat!(
                "Interact with WebMCP tools exposed by the current website. ",
                "Use action='discover' to find available tools, or action='invoke' ",
                "to call a specific tool by name."
            ).to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["discover", "invoke"],
                        "description": "discover: list tools from current site. invoke: call a tool."
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain to discover tools from (default: current page's domain)"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the WebMCP tool to invoke (required for action=invoke)"
                    },
                    "input": {
                        "type": "object",
                        "description": "Input parameters for the tool (required for action=invoke)"
                    }
                },
                "required": ["action"]
            }),
            trust: 0.7,
            handler: Arc::new(move |input| {
                let state = state.clone();
                Box::pin(async move {
                    let action = input["action"].as_str().unwrap_or("discover");
                    let mut st = state.lock().await;

                    match action {
                        "discover" => {
                            let domain = if let Some(d) = input["domain"].as_str() {
                                d.to_string()
                            } else if let Some(session) = st.session.as_ref() {
                                let url = session.current_url().await
                                    .map_err(|e| crate::error::CortexError::Tool(e))?;
                                url::Url::parse(&url)
                                    .ok()
                                    .and_then(|u| u.host_str().map(String::from))
                                    .unwrap_or_default()
                            } else {
                                return Ok(ToolResult {
                                    output: "No domain specified and no browser session.".into(),
                                    success: false,
                                });
                            };

                            let tools = st.webmcp_cache.get_or_discover(&domain).await;
                            if tools.is_empty() {
                                Ok(ToolResult {
                                    output: format!("No WebMCP tools found at {domain}"),
                                    success: true,
                                })
                            } else {
                                let mut out = format!("{} WebMCP tool(s) from {domain}:\n", tools.len());
                                for t in tools {
                                    out.push_str(&format!("- {}: {}\n", t.name, t.description));
                                }
                                Ok(ToolResult { output: out, success: true })
                            }
                        }
                        "invoke" => {
                            let tool_name = input["tool_name"].as_str().unwrap_or("");
                            let tool_input = input.get("input").cloned().unwrap_or(serde_json::json!({}));

                            // Find the tool across all cached domains
                            let tool = st.webmcp_cache.cache.values()
                                .flat_map(|tools: &Vec<super::webmcp::WebMcpTool>| tools.iter())
                                .find(|t| t.name == tool_name)
                                .cloned();

                            let tool = match tool {
                                Some(t) => t,
                                None => return Ok(ToolResult {
                                    output: format!("WebMCP tool '{tool_name}' not found. Use action=discover first."),
                                    success: false,
                                }),
                            };

                            // Try imperative first, fall back to declarative
                            let result = if tool.endpoint.is_some() {
                                super::webmcp::invoke_imperative(&tool, &tool_input).await
                            } else if tool.form_selector.is_some() {
                                if let Some(session) = st.session.as_ref() {
                                    super::webmcp::invoke_declarative(session, &tool, &tool_input).await
                                } else {
                                    Err("No browser session for declarative WebMCP tool.".to_string())
                                }
                            } else {
                                Err("Tool has neither endpoint nor form_selector.".to_string())
                            };

                            match result {
                                Ok(output) => Ok(ToolResult { output, success: true }),
                                Err(e) => Ok(ToolResult {
                                    output: format!("WebMCP invoke error: {e}"),
                                    success: false,
                                }),
                            }
                        }
                        _ => Ok(ToolResult {
                            output: format!("Unknown action: {action}. Use 'discover' or 'invoke'."),
                            success: false,
                        }),
                    }
                })
            }),
        });
    }
}
