//! Server-Sent Events (SSE) client for streaming LLM responses
//!
//! This module re-exports the platform-agnostic streaming implementation.
//! - Web: Uses ReadableStream via web_sys
//! - Native: Uses reqwest with streaming response

// Re-export streaming functionality from platform module
pub use crate::platform::streaming::stream_chat;
