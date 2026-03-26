use crate::error::Result;
use crate::types::*;

// ─── LLM Client trait ───────────────────────────────────

/// Abstraction over LLM backends. Implement this trait for Anthropic, Ollama,
/// OpenAI, or a mock client for testing.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse>;

    /// Complete with tool definitions available.
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        // Default: ignore tools, just call complete
        let _ = tools;
        self.complete(messages).await
    }

    /// Return the model name for recording in LlmCall nodes.
    fn model_name(&self) -> &str;
}

// ─── Mock client for testing ────────────────────────────

/// A mock LLM client that returns pre-scripted responses in FIFO order.
pub struct MockLlmClient {
    pub responses: std::sync::Mutex<std::collections::VecDeque<LlmResponse>>,
    pub name: String,
}

impl MockLlmClient {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses.into()),
            name: "mock".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, _messages: &[Message]) -> Result<LlmResponse> {
        let mut queue = self.responses.lock().unwrap();
        queue
            .pop_front()
            .ok_or_else(|| crate::error::CortexError::Llm("no more mock responses".into()))
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

// ─── Anthropic client (Phase 5) ─────────────────────────

pub struct AnthropicClient {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model: String,
}

impl AnthropicClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse> {
        self.call_api(messages, &[]).await
    }

    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        self.call_api(messages, tools).await
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

impl AnthropicClient {
    fn build_messages(messages: &[Message]) -> (String, Vec<serde_json::Value>) {
        let system_msg = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let chat_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                let role = match m.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "assistant",
                    Role::System => unreachable!(),
                };
                // Use raw content blocks if present (for tool_use/tool_result)
                if let Some(ref blocks) = m.content_blocks {
                    serde_json::json!({
                        "role": role,
                        "content": blocks,
                    })
                } else {
                    serde_json::json!({
                        "role": role,
                        "content": m.content,
                    })
                }
            })
            .collect();

        (system_msg, chat_messages)
    }

    async fn call_api(&self, messages: &[Message], tools: &[serde_json::Value]) -> Result<LlmResponse> {
        let (system_msg, chat_messages) = Self::build_messages(messages);

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system_msg,
            "messages": chat_messages,
        });

        // Add tools if any are registered
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools.to_vec());
        }

        self.do_request(body).await
    }

    async fn do_request(&self, body: serde_json::Value) -> Result<LlmResponse> {

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("request: {e}")))?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("parse: {e}")))?;

        if !status.is_success() {
            return Err(crate::error::CortexError::Llm(format!(
                "API error {status}: {}",
                json
            )));
        }

        // Parse response
        let stop = json["stop_reason"].as_str().unwrap_or("end_turn");
        let stop_reason = match stop {
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let mut text = String::new();
        let mut tool_name = None;
        let mut tool_input = None;
        let mut tool_use_id = None;
        let mut tool_calls = Vec::new();
        let raw_content = json.get("content").cloned();

        if let Some(content) = json["content"].as_array() {
            for block in content {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(t) = block["text"].as_str() {
                            text.push_str(t);
                        }
                    }
                    Some("tool_use") => {
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input = block["input"].clone();
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        // Keep first tool_use for backward compat
                        if tool_name.is_none() {
                            tool_name = Some(name.clone());
                            tool_input = Some(input.clone());
                            tool_use_id = Some(id.clone());
                        }
                        tool_calls.push(ToolCall { id, name, input });
                    }
                    _ => {}
                }
            }
        }

        let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
        let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(LlmResponse {
            text,
            stop_reason,
            tool_name,
            tool_input,
            tool_use_id,
            tool_calls,
            raw_content,
            input_tokens,
            output_tokens,
        })
    }
}

// ─── Ollama client (Phase 5) ────────────────────────────

pub struct OllamaClient {
    pub client: reqwest::Client,
    pub url: String,
    pub model: String,
}

impl OllamaClient {
    pub fn new(model: String, url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
            model,
        }
    }

    fn build_messages(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    },
                    "content": m.content,
                })
            })
            .collect()
    }

    async fn do_request(&self, body: serde_json::Value) -> Result<LlmResponse> {
        let resp = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("ollama: {e}")))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("ollama parse: {e}")))?;

        let text = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Parse tool calls from Ollama response
        let mut tool_calls = Vec::new();
        let mut tool_name = None;
        let mut tool_input = None;
        let mut tool_use_id = None;

        if let Some(calls) = json["message"]["tool_calls"].as_array() {
            for (i, call) in calls.iter().enumerate() {
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let arguments = call["function"]["arguments"].clone();
                let id = format!("ollama_tc_{i}");

                if tool_name.is_none() {
                    tool_name = Some(name.clone());
                    tool_input = Some(arguments.clone());
                    tool_use_id = Some(id.clone());
                }
                tool_calls.push(ToolCall {
                    id,
                    name,
                    input: arguments,
                });
            }
        }

        let stop_reason = if tool_calls.is_empty() {
            StopReason::EndTurn
        } else {
            StopReason::ToolUse
        };

        Ok(LlmResponse {
            text,
            stop_reason,
            tool_name,
            tool_input,
            tool_use_id,
            tool_calls,
            raw_content: None,
            input_tokens: 0,
            output_tokens: 0,
        })
    }
}

#[async_trait::async_trait]
impl LlmClient for OllamaClient {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse> {
        let msgs = Self::build_messages(messages);
        let body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "stream": false,
        });
        self.do_request(body).await
    }

    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        let msgs = Self::build_messages(messages);
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "stream": false,
        });
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools.to_vec());
        }
        self.do_request(body).await
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
