//! Database persistence layer using SurrealDB
//!
//! This module provides persistent storage for sessions, conversations,
//! messages, and settings using SurrealDB with an embedded RocksDB backend.
//!
//! # Features
//!
//! This module is only available when the `db` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! p2a-mcp = { version = "*", features = ["db"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use p2a_mcp::db::{DbConnection, Conversation, Message};
//!
//! // Connect to database
//! let db = DbConnection::connect_rocksdb("./data/p2a.db").await?;
//!
//! // Create a conversation
//! let conv = db.create_conversation("session-123", "My Chat").await?;
//!
//! // Add messages
//! db.add_message(&conv.id, "user", "Hello!").await?;
//! db.add_message(&conv.id, "assistant", "Hi there!").await?;
//!
//! // Load conversation with messages
//! let full = db.get_conversation_with_messages(&conv.id).await?;
//! ```

mod connection;
mod conversations;
mod dataset_meta;
mod models;
mod sessions;
mod tool_calls;

#[cfg(test)]
mod tests;

pub use connection::{DbConnection, DbError, DbStats};
pub use models::{
    Conversation, ConversationWithMessages, DatasetMeta, DbSession, Message, MessageRole, Settings,
    ToolCall, ToolCallStatus,
};
