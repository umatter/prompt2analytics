//! MCP (Model Context Protocol) client for subprocess communication.

pub mod client;
pub mod protocol;

pub use client::{McpClient, McpError};
pub use protocol::{ImageData, ToolResult};
