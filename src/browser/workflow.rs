//! Workflow engine — execute multi-step browser workflows with conditionals.
//!
//! A workflow is an ordered sequence of stored tool invocations,
//! with conditional branching based on page state.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::store::StoredTool;

/// A workflow definition — a sequence of stored tool calls with conditionals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow name.
    pub name: String,
    /// Description of what this workflow accomplishes.
    pub description: String,
    /// Ordered steps in the workflow.
    pub steps: Vec<WorkflowStep>,
    /// Global parameters passed to all tool calls.
    #[serde(default)]
    pub parameters: Vec<super::store::ToolParameter>,
}

/// A single step in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkflowStep {
    /// Execute a stored tool by name.
    #[serde(rename = "tool")]
    RunTool {
        tool_name: String,
        /// Parameter overrides for this invocation.
        #[serde(default)]
        params: HashMap<String, String>,
    },

    /// Conditional based on current page URL pattern.
    #[serde(rename = "if_url")]
    IfUrl {
        pattern: String,
        then: Vec<WorkflowStep>,
        #[serde(default)]
        otherwise: Vec<WorkflowStep>,
    },

    /// Wait for a specific page state before continuing.
    #[serde(rename = "wait_for")]
    WaitFor {
        selector: String,
        #[serde(default = "default_timeout")]
        timeout_ms: u64,
    },

    /// Log a message to the workflow output.
    #[serde(rename = "log")]
    Log { message: String },
}

fn default_timeout() -> u64 { 10000 }

/// Result of executing a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_name: String,
    pub steps_executed: usize,
    pub success: bool,
    pub outputs: Vec<WorkflowStepOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStepOutput {
    pub step_index: usize,
    pub step_type: String,
    pub success: bool,
    pub output: String,
}

/// Execute a workflow using a browser session and a tool registry.
pub async fn execute_workflow(
    session: &super::cdp::BrowserSession,
    workflow: &Workflow,
    tools: &HashMap<String, StoredTool>,
    params: &HashMap<String, String>,
) -> WorkflowResult {
    let mut outputs = Vec::new();
    let mut success = true;

    for (i, step) in workflow.steps.iter().enumerate() {
        match execute_step(session, step, tools, params).await {
            Ok(output) => {
                outputs.push(WorkflowStepOutput {
                    step_index: i,
                    step_type: step_type_name(step),
                    success: true,
                    output,
                });
            }
            Err(e) => {
                outputs.push(WorkflowStepOutput {
                    step_index: i,
                    step_type: step_type_name(step),
                    success: false,
                    output: format!("Error: {e}"),
                });
                success = false;
                break;
            }
        }
    }

    WorkflowResult {
        workflow_name: workflow.name.clone(),
        steps_executed: outputs.len(),
        success,
        outputs,
    }
}

fn execute_step<'a>(
    session: &'a super::cdp::BrowserSession,
    step: &'a WorkflowStep,
    tools: &'a HashMap<String, StoredTool>,
    params: &'a HashMap<String, String>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send + 'a>> {
    Box::pin(async move {
    match step {
        WorkflowStep::RunTool { tool_name, params: extra_params } => {
            let tool = tools
                .get(tool_name)
                .ok_or_else(|| format!("stored tool not found: {tool_name}"))?;

            let mut merged_params = params.clone();
            merged_params.extend(extra_params.clone());

            let results = tool.execute(session, &merged_params).await?;
            let output: Vec<String> = results.iter().map(|r| r.output.clone()).collect();
            Ok(output.join("\n"))
        }
        WorkflowStep::IfUrl { pattern, then, otherwise } => {
            let current_url = session.current_url().await?;
            let matches = current_url.contains(pattern);

            let steps = if matches { then } else { otherwise };
            let mut last_output = format!("URL check: '{}' {} '{}'",
                current_url,
                if matches { "matches" } else { "does not match" },
                pattern,
            );

            for sub_step in steps {
                last_output = execute_step(session, sub_step, tools, params).await?;
            }
            Ok(last_output)
        }
        WorkflowStep::WaitFor { selector, timeout_ms } => {
            session.wait_for_selector(selector, *timeout_ms).await?;
            Ok(format!("Found: {selector}"))
        }
        WorkflowStep::Log { message } => {
            let interpolated = StoredTool::interpolate(message, params);
            Ok(format!("[log] {interpolated}"))
        }
    }
    })
}

fn step_type_name(step: &WorkflowStep) -> String {
    match step {
        WorkflowStep::RunTool { tool_name, .. } => format!("tool:{tool_name}"),
        WorkflowStep::IfUrl { .. } => "if_url".to_string(),
        WorkflowStep::WaitFor { .. } => "wait_for".to_string(),
        WorkflowStep::Log { .. } => "log".to_string(),
    }
}
