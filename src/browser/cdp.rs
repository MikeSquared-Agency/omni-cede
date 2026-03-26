//! Chrome DevTools Protocol (CDP) client over WebSocket.
//!
//! Communicates with a running Chrome instance via its debugging WebSocket.
//! Supports navigation, DOM queries, JavaScript evaluation, screenshots,
//! input events, and network interception.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock, oneshot};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type WsWriter = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;
type WsReader = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

/// A CDP session wrapping a WebSocket connection to Chrome.
pub struct BrowserSession {
    writer: Arc<Mutex<WsWriter>>,
    /// Pending request callbacks keyed by message ID.
    pending: Arc<RwLock<HashMap<u64, oneshot::Sender<Value>>>>,
    /// Monotonic message counter.
    next_id: AtomicU64,
    /// Chrome process handle (if we spawned it).
    _chrome: Option<tokio::process::Child>,
    /// Event listeners keyed by method name.
    event_listeners: Arc<RwLock<HashMap<String, Vec<tokio::sync::mpsc::Sender<Value>>>>>,
}

/// Response from calling navigate.
#[derive(Debug, Clone)]
pub struct NavigateResult {
    pub frame_id: String,
    pub loader_id: Option<String>,
}

/// A compact representation of a page element.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PageElement {
    pub tag: String,
    pub text: String,
    pub attributes: HashMap<String, String>,
    pub selector: String,
}

impl BrowserSession {
    /// Launch Chrome with remote debugging and connect via CDP.
    ///
    /// `chrome_path` — path to Chrome executable (None = auto-detect).
    /// `port` — debugging port (default 9222).
    /// `headless` — whether to run headless.
    pub async fn launch(
        chrome_path: Option<&str>,
        port: u16,
        headless: bool,
    ) -> Result<Self, String> {
        let chrome = find_chrome(chrome_path)?;

        let mut args = vec![
            format!("--remote-debugging-port={port}"),
            "--no-first-run".to_string(),
            "--no-default-browser-check".to_string(),
            "--disable-background-networking".to_string(),
            "--disable-component-update".to_string(),
            "--disable-features=TranslateUI".to_string(),
        ];

        if headless {
            args.push("--headless=new".to_string());
        }

        // Apply stealth flags
        args.extend(super::stealth::chrome_flags());

        let child = tokio::process::Command::new(&chrome)
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to launch Chrome: {e}"))?;

        // Wait for the debugger to come up
        let ws_url = wait_for_debugger(port, 15).await?;

        let mut session = Self::connect(&ws_url).await?;
        session._chrome = Some(child);

        Ok(session)
    }

    /// Connect to an already-running Chrome debugger at the given WebSocket URL.
    pub async fn connect(ws_url: &str) -> Result<Self, String> {
        let (ws, _) = tokio_tungstenite::connect_async(ws_url)
            .await
            .map_err(|e| format!("CDP connect failed: {e}"))?;

        let (writer, reader) = ws.split();
        let pending: Arc<RwLock<HashMap<u64, oneshot::Sender<Value>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let event_listeners: Arc<RwLock<HashMap<String, Vec<tokio::sync::mpsc::Sender<Value>>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let session = Self {
            writer: Arc::new(Mutex::new(writer)),
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            _chrome: None,
            event_listeners: event_listeners.clone(),
        };

        // Spawn reader task
        tokio::spawn(Self::reader_loop(reader, pending, event_listeners));

        Ok(session)
    }

    /// Background loop that reads CDP responses and events.
    async fn reader_loop(
        mut reader: WsReader,
        pending: Arc<RwLock<HashMap<u64, oneshot::Sender<Value>>>>,
        event_listeners: Arc<RwLock<HashMap<String, Vec<tokio::sync::mpsc::Sender<Value>>>>>,
    ) {
        while let Some(Ok(msg)) = reader.next().await {
            if let WsMessage::Text(text) = msg {
                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                    // CDP response (has "id" field)
                    if let Some(id) = json["id"].as_u64() {
                        let mut map = pending.write().await;
                        if let Some(tx) = map.remove(&id) {
                            let _ = tx.send(json);
                        }
                    }
                    // CDP event (has "method" field, no "id")
                    else if let Some(method) = json["method"].as_str() {
                        let listeners = event_listeners.read().await;
                        if let Some(senders) = listeners.get(method) {
                            let params = json["params"].clone();
                            for tx in senders {
                                let _ = tx.try_send(params.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    /// Send a CDP command and wait for the response.
    pub async fn send(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let msg = serde_json::json!({
            "id": id,
            "method": method,
            "params": params,
        });

        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending.write().await;
            map.insert(id, tx);
        }

        {
            let mut writer = self.writer.lock().await;
            writer
                .send(WsMessage::Text(msg.to_string()))
                .await
                .map_err(|e| format!("CDP send error: {e}"))?;
        }

        let response = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
            .await
            .map_err(|_| "CDP response timeout (30s)".to_string())?
            .map_err(|_| "CDP response channel closed".to_string())?;

        if let Some(err) = response.get("error") {
            return Err(format!("CDP error: {}", err));
        }

        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Subscribe to a CDP event. Returns a receiver that yields event params.
    pub async fn on_event(&self, method: &str) -> tokio::sync::mpsc::Receiver<Value> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut listeners = self.event_listeners.write().await;
        listeners
            .entry(method.to_string())
            .or_default()
            .push(tx);
        rx
    }

    // ─── High-level helpers ───────────────────────────────

    /// Navigate to a URL and wait for load.
    pub async fn navigate(&self, url: &str) -> Result<NavigateResult, String> {
        // Enable Page domain
        self.send("Page.enable", serde_json::json!({})).await?;

        let result = self
            .send("Page.navigate", serde_json::json!({ "url": url }))
            .await?;

        let frame_id = result["frameId"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let loader_id = result["loaderId"].as_str().map(String::from);

        // Wait for loadEventFired
        let mut rx = self.on_event("Page.loadEventFired").await;
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            rx.recv(),
        )
        .await;

        Ok(NavigateResult {
            frame_id,
            loader_id,
        })
    }

    /// Get the current page URL.
    pub async fn current_url(&self) -> Result<String, String> {
        let result = self
            .send(
                "Runtime.evaluate",
                serde_json::json!({
                    "expression": "window.location.href",
                    "returnByValue": true,
                }),
            )
            .await?;
        Ok(result["result"]["value"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    /// Evaluate JavaScript and return the result as a string.
    pub async fn evaluate(&self, expression: &str) -> Result<Value, String> {
        let result = self
            .send(
                "Runtime.evaluate",
                serde_json::json!({
                    "expression": expression,
                    "returnByValue": true,
                    "awaitPromise": true,
                }),
            )
            .await?;

        if let Some(exc) = result.get("exceptionDetails") {
            return Err(format!("JS error: {}", exc));
        }

        Ok(result["result"]["value"].clone())
    }

    /// Click on an element matching a CSS selector.
    pub async fn click(&self, selector: &str) -> Result<(), String> {
        let js = format!(
            r#"(() => {{
                const el = document.querySelector({sel});
                if (!el) return 'NOT_FOUND';
                el.click();
                return 'OK';
            }})()"#,
            sel = serde_json::to_string(selector).unwrap(),
        );
        let result = self.evaluate(&js).await?;
        if result.as_str() == Some("NOT_FOUND") {
            return Err(format!("element not found: {selector}"));
        }
        Ok(())
    }

    /// Type text into the focused element, character by character.
    pub async fn type_text(&self, text: &str) -> Result<(), String> {
        for ch in text.chars() {
            self.send(
                "Input.dispatchKeyEvent",
                serde_json::json!({
                    "type": "keyDown",
                    "text": ch.to_string(),
                }),
            )
            .await?;
            self.send(
                "Input.dispatchKeyEvent",
                serde_json::json!({
                    "type": "keyUp",
                    "text": ch.to_string(),
                }),
            )
            .await?;
        }
        Ok(())
    }

    /// Fill an input element matching a selector with text.
    pub async fn fill(&self, selector: &str, text: &str) -> Result<(), String> {
        // Focus the element
        let focus_js = format!(
            r#"(() => {{
                const el = document.querySelector({sel});
                if (!el) return 'NOT_FOUND';
                el.focus();
                el.value = '';
                return 'OK';
            }})()"#,
            sel = serde_json::to_string(selector).unwrap(),
        );
        let result = self.evaluate(&focus_js).await?;
        if result.as_str() == Some("NOT_FOUND") {
            return Err(format!("element not found: {selector}"));
        }
        self.type_text(text).await?;

        // Trigger input event
        let trigger_js = format!(
            r#"(() => {{
                const el = document.querySelector({sel});
                if (el) {{
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            }})()"#,
            sel = serde_json::to_string(selector).unwrap(),
        );
        self.evaluate(&trigger_js).await?;
        Ok(())
    }

    /// Take a screenshot (PNG), return as base64.
    pub async fn screenshot(&self) -> Result<String, String> {
        let result = self
            .send(
                "Page.captureScreenshot",
                serde_json::json!({ "format": "png" }),
            )
            .await?;
        result["data"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| "no screenshot data".to_string())
    }

    /// Get a compact text snapshot of the page DOM.
    pub async fn snapshot(&self) -> Result<Vec<PageElement>, String> {
        super::snapshot::take_snapshot(self).await
    }

    /// Get page HTML content.
    pub async fn get_html(&self) -> Result<String, String> {
        let result = self
            .send(
                "Runtime.evaluate",
                serde_json::json!({
                    "expression": "document.documentElement.outerHTML",
                    "returnByValue": true,
                }),
            )
            .await?;
        Ok(result["result"]["value"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    /// Wait for a selector to appear in the DOM, with timeout.
    pub async fn wait_for_selector(
        &self,
        selector: &str,
        timeout_ms: u64,
    ) -> Result<(), String> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms);

        loop {
            let js = format!(
                "document.querySelector({}) !== null",
                serde_json::to_string(selector).unwrap(),
            );
            let result = self.evaluate(&js).await?;
            if result.as_bool() == Some(true) {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(format!(
                    "timeout waiting for selector '{selector}' ({timeout_ms}ms)"
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }

    /// Scroll the page by (x, y) pixels.
    pub async fn scroll(&self, x: i32, y: i32) -> Result<(), String> {
        self.evaluate(&format!("window.scrollBy({x}, {y})")).await?;
        Ok(())
    }

    /// Get all cookies.
    pub async fn get_cookies(&self) -> Result<Value, String> {
        self.send("Network.getCookies", serde_json::json!({})).await
    }

    /// Set a cookie.
    pub async fn set_cookie(&self, cookie: Value) -> Result<(), String> {
        self.send("Network.setCookie", cookie).await?;
        Ok(())
    }

    /// Close the browser session.
    pub async fn close(&self) -> Result<(), String> {
        let _ = self.send("Browser.close", serde_json::json!({})).await;
        Ok(())
    }
}

// ─── Chrome discovery ───────────────────────────────────

fn find_chrome(explicit: Option<&str>) -> Result<String, String> {
    if let Some(p) = explicit {
        return Ok(p.to_string());
    }

    let candidates = if cfg!(target_os = "windows") {
        vec![
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
    } else if cfg!(target_os = "macos") {
        vec!["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    } else {
        vec![
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ]
    };

    for c in &candidates {
        if std::path::Path::new(c).exists() {
            return Ok(c.to_string());
        }
    }

    Err("Chrome not found. Set chrome_path explicitly or install Chrome.".to_string())
}

/// Poll the Chrome debugger endpoint until it responds with a WebSocket URL.
async fn wait_for_debugger(port: u16, max_secs: u64) -> Result<String, String> {
    let url = format!("http://127.0.0.1:{port}/json/version");
    let client = reqwest::Client::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(max_secs);

    loop {
        if std::time::Instant::now() > deadline {
            return Err(format!(
                "Chrome debugger did not respond on port {port} within {max_secs}s"
            ));
        }

        match client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<Value>().await {
                    if let Some(ws) = json["webSocketDebuggerUrl"].as_str() {
                        return Ok(ws.to_string());
                    }
                }
            }
            Err(_) => {}
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}
