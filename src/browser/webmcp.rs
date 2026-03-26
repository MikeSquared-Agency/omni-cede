//! WebMCP client — discover and invoke structured tools exposed by websites.
//!
//! Chrome's WebMCP (early preview) lets websites declare tools via
//! `/.well-known/webmcp.json`. This module discovers those declarations
//! and converts them into callable tool definitions for the agent.
//!
//! Reference: https://developer.chrome.com/blog/webmcp-epp

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A WebMCP tool descriptor as declared by a website.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebMcpTool {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// The URL endpoint to POST to (for imperative tools).
    #[serde(default)]
    pub endpoint: Option<String>,
    /// CSS selector for the form element (for declarative tools).
    #[serde(default)]
    pub form_selector: Option<String>,
    /// The originating domain.
    #[serde(skip_deserializing, default)]
    pub domain: String,
}

/// WebMCP manifest (/.well-known/webmcp.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebMcpManifest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tools: Vec<WebMcpTool>,
    #[serde(default)]
    pub version: String,
}

/// Discover WebMCP tools from a website.
///
/// Fetches `https://{domain}/.well-known/webmcp.json` and parses the manifest.
/// Returns an empty vec if the site doesn't support WebMCP.
pub async fn discover(domain: &str) -> Vec<WebMcpTool> {
    let url = format!("https://{domain}/.well-known/webmcp.json");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<WebMcpManifest>().await {
                Ok(manifest) => {
                    let mut tools = manifest.tools;
                    for tool in &mut tools {
                        tool.domain = domain.to_string();
                    }
                    tracing::info!(
                        "WebMCP: discovered {} tool(s) from {domain}",
                        tools.len()
                    );
                    tools
                }
                Err(e) => {
                    tracing::debug!("WebMCP: invalid manifest from {domain}: {e}");
                    vec![]
                }
            }
        }
        Ok(_) => {
            tracing::debug!("WebMCP: no manifest at {domain}");
            vec![]
        }
        Err(e) => {
            tracing::debug!("WebMCP: fetch failed for {domain}: {e}");
            vec![]
        }
    }
}

/// Invoke a WebMCP tool via its endpoint (imperative mode).
///
/// POSTs the input JSON to the tool's endpoint and returns the response.
pub async fn invoke_imperative(
    tool: &WebMcpTool,
    input: &serde_json::Value,
) -> Result<String, String> {
    let endpoint = tool
        .endpoint
        .as_deref()
        .ok_or_else(|| "tool has no endpoint (declarative only)".to_string())?;

    let url = if endpoint.starts_with("http") {
        endpoint.to_string()
    } else {
        format!("https://{}{}", tool.domain, endpoint)
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    let resp = client
        .post(&url)
        .json(input)
        .send()
        .await
        .map_err(|e| format!("WebMCP invoke error: {e}"))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("WebMCP response error: {e}"))?;

    if !status.is_success() {
        return Err(format!("WebMCP tool returned {status}: {body}"));
    }

    Ok(body)
}

/// Invoke a WebMCP tool via form filling (declarative mode).
///
/// Uses the browser session to fill and submit the form identified
/// by the tool's `form_selector`.
pub async fn invoke_declarative(
    session: &super::cdp::BrowserSession,
    tool: &WebMcpTool,
    input: &serde_json::Value,
) -> Result<String, String> {
    let form_selector = tool
        .form_selector
        .as_deref()
        .ok_or_else(|| "tool has no form_selector (imperative only)".to_string())?;

    // Fill each input field in the form
    if let Some(params) = input.as_object() {
        for (key, value) in params {
            let val_str = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };

            // Try to fill by name attribute within the form
            let selector = format!("{form_selector} [name=\"{key}\"]");
            if let Err(_) = session.fill(&selector, &val_str).await {
                // Fall back to aria-label
                let selector = format!("{form_selector} [aria-label=\"{key}\"]");
                let _ = session.fill(&selector, &val_str).await;
            }
        }
    }

    // Submit the form
    let submit_js = format!(
        r#"(() => {{
            const form = document.querySelector({sel});
            if (!form) return 'FORM_NOT_FOUND';
            const submit = form.querySelector('[type="submit"], button');
            if (submit) {{ submit.click(); return 'CLICKED'; }}
            form.submit();
            return 'SUBMITTED';
        }})()"#,
        sel = serde_json::to_string(form_selector).unwrap(),
    );

    let result = session.evaluate(&submit_js).await?;
    if result.as_str() == Some("FORM_NOT_FOUND") {
        return Err(format!("form not found: {form_selector}"));
    }

    // Wait briefly for response
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Take a snapshot of the result page
    let snapshot = session.snapshot().await?;
    let snapshot_text = super::snapshot::format_snapshot(&snapshot);

    Ok(format!("Form submitted. Page snapshot:\n{snapshot_text}"))
}

/// Cache of discovered WebMCP tools, keyed by domain.
pub struct WebMcpCache {
    pub cache: HashMap<String, Vec<WebMcpTool>>,
}

impl WebMcpCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get cached tools for a domain, or discover them.
    pub async fn get_or_discover(&mut self, domain: &str) -> &[WebMcpTool] {
        if !self.cache.contains_key(domain) {
            let tools = discover(domain).await;
            self.cache.insert(domain.to_string(), tools);
        }
        self.cache.get(domain).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// List all cached domains and their tool counts.
    pub fn summary(&self) -> Vec<(String, usize)> {
        self.cache
            .iter()
            .map(|(domain, tools)| (domain.clone(), tools.len()))
            .collect()
    }
}
