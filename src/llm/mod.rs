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
                        tool_calls.push(ToolCall { id, name, input, thought_signature: None });
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
                    thought_signature: None,
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

// ─── Gemini client (Google) ─────────────────────────────

pub struct GeminiClient {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model: String,
}

impl GeminiClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    fn build_payload(messages: &[Message], tools: &[serde_json::Value]) -> serde_json::Value {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        for m in messages {
            match m.role {
                Role::System => {
                    system_instruction = Some(serde_json::json!({
                        "parts": [{"text": m.content}]
                    }));
                }
                Role::User => {
                    // For tool_results, Gemini expects functionResponse
                    if let Some(blocks) = &m.content_blocks {
                        if let Some(arr) = blocks.as_array() {
                            let mut parts = Vec::new();
                            for block in arr {
                                if block["type"] == "tool_result" {
                                    let content_str = block["content"].as_str().unwrap_or("{}");
                                    // Gemini expects functionResponse to have the name, but our Message only has tool_use_id.
                                    // For simplicity we try to map it, or default to parsing it as text response.
                                    let parsed_content: serde_json::Value = serde_json::from_str(content_str)
                                        .unwrap_or(serde_json::json!({"output": content_str}));
                                        
                                    parts.push(serde_json::json!({
                                        "functionResponse": {
                                            "name": block["tool_use_id"], // using id as name is a fallback if name is lost
                                            "response": { "name": block["tool_use_id"], "content": parsed_content }
                                        }
                                    }));
                                } else {
                                    parts.push(serde_json::json!({"text": m.content}));
                                }
                            }
                            contents.push(serde_json::json!({ "role": "user", "parts": parts }));
                        }
                    } else {
                        contents.push(serde_json::json!({
                            "role": "user",
                            "parts": [{"text": m.content}]
                        }));
                    }
                }
                Role::Assistant => {
                    // Gemini assistant is "model"
                    if let Some(blocks) = &m.content_blocks {
                        if let Some(arr) = blocks.as_array() {
                            let mut parts = Vec::new();
                            for block in arr {
                                if block["type"] == "tool_use" {
                                    let mut func_call = serde_json::json!({
                                        "name": block["name"],
                                        "args": block["input"]
                                    });
                                    if let Some(id) = block.get("id") {
                                        func_call["id"] = id.clone();
                                    }
                                    let mut part_obj = serde_json::json!({
                                        "functionCall": func_call
                                    });
                                    if let Some(ts) = block.get("thought_signature") {
                                        part_obj["thoughtSignature"] = ts.clone();
                                    }
                                    parts.push(part_obj);
                                } else if block["type"] == "text" {
                                    parts.push(serde_json::json!({"text": block["text"]}));
                                }
                            }
                            contents.push(serde_json::json!({ "role": "model", "parts": parts }));
                        }
                    } else {
                        contents.push(serde_json::json!({
                            "role": "model",
                            "parts": [{"text": m.content}]
                        }));
                    }
                }
                Role::Tool => {
                    // Tool results map to user role with functionResponse in Gemini
                    let parsed: serde_json::Value = serde_json::from_str(&m.content).unwrap_or(serde_json::json!({"output": &m.content}));
                    let name = m.tool_call_id.clone().unwrap_or_else(|| "unknown_tool".to_string());
                    contents.push(serde_json::json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": name.clone(),
                                "response": { "name": name, "content": parsed }
                            }
                        }]
                    }));
                }
            }
        }

        let mut payload = serde_json::json!({
            "contents": contents,
        });

        if let Some(sys) = system_instruction {
            payload["systemInstruction"] = sys;
        }

        // Convert Anthropic tools format to Gemini function declarations
        if !tools.is_empty() {
            let function_declarations: Vec<serde_json::Value> = tools.iter().map(|t| {
                serde_json::json!({
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"]
                })
            }).collect();

            payload["tools"] = serde_json::json!([
                { "functionDeclarations": function_declarations }
            ]);
        }

        payload
    }

    async fn do_request(&self, body: serde_json::Value) -> Result<LlmResponse> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("gemini request: {e}")))?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("gemini parse: {e}")))?;

        if !status.is_success() {
            return Err(crate::error::CortexError::Llm(format!(
                "Gemini API error {status}: {}",
                json
            )));
        }

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut tool_name = None;
        let mut tool_input = None;
        let mut tool_use_id = None;

        if let Some(candidates) = json["candidates"].as_array() {
            if let Some(candidate) = candidates.first() {
                if let Some(parts) = candidate["content"]["parts"].as_array() {
                    for (i, part) in parts.iter().enumerate() {
                        if let Some(t) = part["text"].as_str() {
                            text.push_str(t);
                        } else if let Some(func) = part.get("functionCall") {
                            let name = func["name"].as_str().unwrap_or("").to_string();
                            let args = func["args"].clone();
                            let thought_signature = part.get("thoughtSignature").and_then(|v| v.as_str()).map(|s| s.to_string());
                            
                            // Try to get ID from Gemini 3.1, fallback to generated ID
                            let id = func.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
                                .unwrap_or_else(|| format!("gemini_tc_{}_{i}", uuid::Uuid::new_v4().simple()));
                            
                            if tool_name.is_none() {
                                tool_name = Some(name.clone());
                                tool_input = Some(args.clone());
                                tool_use_id = Some(id.clone());
                            }
                            
                            tool_calls.push(ToolCall {
                                id,
                                name,
                                input: args,
                                thought_signature,
                            });
                        }
                    }
                }
            }
        }

        let stop_reason = if tool_calls.is_empty() {
            StopReason::EndTurn
        } else {
            StopReason::ToolUse
        };

        let input_tokens = json["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0) as usize;
        let output_tokens = json["usageMetadata"]["candidatesTokenCount"].as_u64().unwrap_or(0) as usize;

        // Construct raw content blocks mapped to Anthropic style for replayability in the unified agent
        let raw_content = if !tool_calls.is_empty() {
            let mut blocks = Vec::new();
            if !text.is_empty() {
                blocks.push(serde_json::json!({ "type": "text", "text": text }));
            }
            for tc in &tool_calls {
                let mut tc_block = serde_json::json!({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input
                });
                if let Some(ts) = &tc.thought_signature {
                    tc_block["thought_signature"] = serde_json::json!(ts);
                }
                blocks.push(tc_block);
            }
            Some(serde_json::Value::Array(blocks))
        } else {
            None
        };

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

#[async_trait::async_trait]
impl LlmClient for GeminiClient {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse> {
        let payload = Self::build_payload(messages, &[]);
        self.do_request(payload).await
    }

    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        let payload = Self::build_payload(messages, tools);
        self.do_request(payload).await
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
