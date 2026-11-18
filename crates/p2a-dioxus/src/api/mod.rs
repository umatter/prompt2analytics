//! API client module for communicating with p2a-mcp backend

mod client;
mod sse;
pub mod types;

pub use client::*;
pub use sse::*;
pub use types::*;

/// The one formatter for REST and SSE errors.
pub use crate::platform::error_text;
