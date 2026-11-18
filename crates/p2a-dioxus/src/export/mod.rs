//! Export functionality for conversations
//!
//! Supports exporting conversations in multiple formats:
//! - JSON: shareable archive of the transcript, tool calls and charts (there is no import)
//! - Markdown: Human-readable, suitable for GitHub/docs
//! - HTML: Self-contained shareable report

mod download;
mod html;
mod json;
mod markdown;
mod types;

pub use download::{copy_to_clipboard, generate_filename, trigger_download};
pub use html::export_to_html;
pub use json::export_to_json;
pub use markdown::export_to_markdown;
pub use types::*;
