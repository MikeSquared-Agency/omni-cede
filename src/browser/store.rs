//! Stored tool definitions — reusable browser interaction patterns.
//!
//! Inspired by Xbot's stored tool system. A stored tool captures a
//! repeatable browser interaction as a JSON definition that can be
//! replayed on demand.

use serde::{Deserialize, Serialize};

/// A stored browser tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTool {
    /// Unique name (e.g. "twitter_post", "google_search").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// The URL to navigate to before executing steps.
    pub start_url: String,
    /// Ordered list of interaction steps.
    pub steps: Vec<ToolStep>,
    /// Input parameters this tool accepts.
    #[serde(default)]
    pub parameters: Vec<ToolParameter>,
    /// Domain briefing — context about the site for the LLM.
    #[serde(default)]
    pub domain_briefing: Option<String>,
    /// Whether to apply stealth patches before execution.
    #[serde(default = "default_true")]
    pub stealth: bool,
}

fn default_true() -> bool {
    true
}

/// A single step in a stored tool's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum ToolStep {
    /// Navigate to a URL (supports {{param}} interpolation).
    #[serde(rename = "navigate")]
    Navigate { url: String },

    /// Click an element by CSS selector.
    #[serde(rename = "click")]
    Click { selector: String },

    /// Fill an input with text (supports {{param}} interpolation).
    #[serde(rename = "fill")]
    Fill { selector: String, value: String },

    /// Wait for a selector to appear.
    #[serde(rename = "wait")]
    Wait {
        selector: String,
        #[serde(default = "default_wait_ms")]
        timeout_ms: u64,
    },

    /// Wait a fixed duration.
    #[serde(rename = "delay")]
    Delay {
        #[serde(default = "default_delay_ms")]
        ms: u64,
    },

    /// Take a snapshot and return it as the step's output.
    #[serde(rename = "snapshot")]
    Snapshot,

    /// Evaluate JavaScript and capture the result.
    #[serde(rename = "evaluate")]
    Evaluate { expression: String },

    /// Take a screenshot (returned as base64 PNG).
    #[serde(rename = "screenshot")]
    Screenshot,

    /// Scroll the page.
    #[serde(rename = "scroll")]
    Scroll {
        #[serde(default)]
        x: i32,
        #[serde(default = "default_scroll_y")]
        y: i32,
    },

    /// Press a key (Enter, Tab, Escape, etc.).
    #[serde(rename = "key")]
    Key { key: String },

    /// Conditional: only execute inner steps if selector exists.
    #[serde(rename = "if_exists")]
    IfExists {
        selector: String,
        then: Vec<ToolStep>,
        #[serde(default)]
        otherwise: Vec<ToolStep>,
    },
}

fn default_wait_ms() -> u64 { 5000 }
fn default_delay_ms() -> u64 { 1000 }
fn default_scroll_y() -> i32 { 500 }

/// A parameter that a stored tool accepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    #[serde(default = "default_param_type")]
    pub param_type: String,
    pub description: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub default_value: Option<String>,
}

fn default_param_type() -> String {
    "string".to_string()
}

impl StoredTool {
    /// Interpolate `{{param}}` placeholders in a string with actual values.
    pub fn interpolate(
        template: &str,
        params: &std::collections::HashMap<String, String>,
    ) -> String {
        let mut result = template.to_string();
        for (key, value) in params {
            result = result.replace(&format!("{{{{{key}}}}}"), value);
        }
        result
    }

    /// Execute this stored tool using a browser session.
    pub async fn execute(
        &self,
        session: &super::cdp::BrowserSession,
        params: &std::collections::HashMap<String, String>,
    ) -> Result<Vec<StepResult>, String> {
        // Apply stealth if requested
        if self.stealth {
            super::stealth::apply_stealth(session).await?;
        }

        // Navigate to start URL
        let url = Self::interpolate(&self.start_url, params);
        session.navigate(&url).await?;

        // Wait a moment for page load
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Execute steps
        let mut results = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            match Self::execute_step(session, step, params).await {
                Ok(r) => results.push(r),
                Err(e) => {
                    results.push(StepResult {
                        step_index: i,
                        success: false,
                        output: format!("Step {i} failed: {e}"),
                    });
                    break; // Stop on failure
                }
            }
        }

        Ok(results)
    }

    fn execute_step<'a>(
        session: &'a super::cdp::BrowserSession,
        step: &'a ToolStep,
        params: &'a std::collections::HashMap<String, String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<StepResult, String>> + Send + 'a>> {
        Box::pin(async move {
        let i = 0; // Simplified — real index tracked by caller
        match step {
            ToolStep::Navigate { url } => {
                let url = Self::interpolate(url, params);
                session.navigate(&url).await?;
                Ok(StepResult { step_index: i, success: true, output: format!("Navigated to {url}") })
            }
            ToolStep::Click { selector } => {
                let sel = Self::interpolate(selector, params);
                session.click(&sel).await?;
                Ok(StepResult { step_index: i, success: true, output: format!("Clicked {sel}") })
            }
            ToolStep::Fill { selector, value } => {
                let sel = Self::interpolate(selector, params);
                let val = Self::interpolate(value, params);
                session.fill(&sel, &val).await?;
                Ok(StepResult { step_index: i, success: true, output: format!("Filled {sel}") })
            }
            ToolStep::Wait { selector, timeout_ms } => {
                let sel = Self::interpolate(selector, params);
                session.wait_for_selector(&sel, *timeout_ms).await?;
                Ok(StepResult { step_index: i, success: true, output: format!("Found {sel}") })
            }
            ToolStep::Delay { ms } => {
                tokio::time::sleep(std::time::Duration::from_millis(*ms)).await;
                Ok(StepResult { step_index: i, success: true, output: format!("Waited {ms}ms") })
            }
            ToolStep::Snapshot => {
                let elements = session.snapshot().await?;
                let text = super::snapshot::format_snapshot(&elements);
                Ok(StepResult { step_index: i, success: true, output: text })
            }
            ToolStep::Evaluate { expression } => {
                let expr = Self::interpolate(expression, params);
                let result = session.evaluate(&expr).await?;
                Ok(StepResult { step_index: i, success: true, output: result.to_string() })
            }
            ToolStep::Screenshot => {
                let b64 = session.screenshot().await?;
                Ok(StepResult { step_index: i, success: true, output: format!("[screenshot: {} bytes base64]", b64.len()) })
            }
            ToolStep::Scroll { x, y } => {
                session.scroll(*x, *y).await?;
                Ok(StepResult { step_index: i, success: true, output: format!("Scrolled ({x}, {y})") })
            }
            ToolStep::Key { key } => {
                session
                    .send(
                        "Input.dispatchKeyEvent",
                        serde_json::json!({
                            "type": "keyDown",
                            "key": key,
                        }),
                    )
                    .await?;
                session
                    .send(
                        "Input.dispatchKeyEvent",
                        serde_json::json!({
                            "type": "keyUp",
                            "key": key,
                        }),
                    )
                    .await?;
                Ok(StepResult { step_index: i, success: true, output: format!("Pressed {key}") })
            }
            ToolStep::IfExists { selector, then, otherwise } => {
                let sel = Self::interpolate(selector, params);
                let exists = session
                    .evaluate(&format!(
                        "document.querySelector({}) !== null",
                        serde_json::to_string(&sel).unwrap(),
                    ))
                    .await?;

                let steps = if exists.as_bool() == Some(true) {
                    then
                } else {
                    otherwise
                };

                let mut last_result = StepResult {
                    step_index: i,
                    success: true,
                    output: format!("Condition: {sel} = {exists}"),
                };
                for sub_step in steps {
                    last_result = Self::execute_step(session, sub_step, params).await?;
                }
                Ok(last_result)
            }
        }
        })
    }
}

/// Result of a single step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_index: usize,
    pub success: bool,
    pub output: String,
}
