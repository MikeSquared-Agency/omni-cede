//! Browser automation module (feature-gated behind `browser`).
//!
//! Provides:
//! - CDP (Chrome DevTools Protocol) client over WebSocket
//! - Compact page snapshots (DOM → structured text)
//! - Stealth / anti-detection helpers
//! - WebMCP client (Chrome's agent-website structured tool API)
//! - Stored tool definitions and workflow engine
//! - Browser tools registered in the agent's ToolRegistry

#[cfg(feature = "browser")]
pub mod cdp;
#[cfg(feature = "browser")]
pub mod snapshot;
#[cfg(feature = "browser")]
pub mod stealth;
#[cfg(feature = "browser")]
pub mod webmcp;
#[cfg(feature = "browser")]
pub mod store;
#[cfg(feature = "browser")]
pub mod workflow;
#[cfg(feature = "browser")]
pub mod tools;

/// Re-export the browser session for convenience.
#[cfg(feature = "browser")]
pub use cdp::BrowserSession;
