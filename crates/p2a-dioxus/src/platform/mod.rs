//! Platform abstraction layer for cross-platform support
//!
//! This module provides abstractions for platform-specific functionality:
//! - Storage: Persistent settings storage (localStorage on web, file-based on native)
//! - HTTP: HTTP client for API requests
//! - Streaming: SSE streaming for LLM responses

pub mod http;
pub mod storage;
pub mod streaming;

pub use http::*;
pub use storage::*;
// Note: streaming is imported directly via crate::platform::streaming::stream_chat

/// Returns true if running on a native platform (desktop/mobile)
#[inline]
pub const fn is_native() -> bool {
    !cfg!(target_arch = "wasm32")
}

/// Returns true if running on web (WASM)
#[inline]
pub const fn is_web() -> bool {
    cfg!(target_arch = "wasm32")
}

/// Get the platform name for logging/display
pub fn platform_name() -> &'static str {
    if cfg!(target_arch = "wasm32") {
        "web"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    }
}
